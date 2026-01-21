# algos/kf_ukf_pf/ekf_nonlinear_toy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class EKFNonlinearToy:
    """
    EKF for the toy nonlinear model (component-wise), with diagonal Jacobians.

    f_i(x) = alpha*sin(beta*x_i + phi) + delta
    => df_i/dx_i = alpha*beta*cos(beta*x_i + phi)

    h_i(x) = a*(b*x_i + c)^2
    => dh_i/dx_i = 2*a*b*(b*x_i + c)
    """
    params_design: Dict[str, float]
    Q: np.ndarray  # (2,2)
    R: np.ndarray  # (2,2)

    def f(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        alpha, beta, phi, delta = p["alpha"], p["beta"], p["phi"], p["delta"]
        return alpha * np.sin(beta * x + phi) + delta

    def h(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        a, b, c = p["a"], p["b"], p["c"]
        return a * (b * x + c) ** 2

    def Jf(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        alpha, beta, phi = p["alpha"], p["beta"], p["phi"]
        d = alpha * beta * np.cos(beta * x + phi)  # component-wise
        return np.diag(d)

    def Jh(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        a, b, c = p["a"], p["b"], p["c"]
        d = 2.0 * a * b * (b * x + c)
        return np.diag(d)

    def run(
        self,
        Y: np.ndarray,                       # (T,2)
        x0_hat: Optional[np.ndarray] = None, # (2,)
        P0: Optional[np.ndarray] = None      # (2,2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]
        assert Y.shape[1] == 2

        if x0_hat is None:
            x_hat = np.zeros((2,), dtype=float)
        else:
            x_hat = np.asarray(x0_hat, dtype=float).reshape(2,)

        if P0 is None:
            P = np.eye(2, dtype=float)
        else:
            P = np.asarray(P0, dtype=float).reshape(2, 2)

        X_hat = np.zeros((T, 2), dtype=float)
        Ps = np.zeros((T, 2, 2), dtype=float)
        I = np.eye(2, dtype=float)

        for t in range(T):
            # Predict
            x_pred = self.f(x_hat)
            Fk = self.Jf(x_hat)
            P_pred = Fk @ P @ Fk.T + self.Q

            # Innovation
            y_pred = self.h(x_pred)
            Hk = self.Jh(x_pred)
            dy = Y[t] - y_pred

            S = Hk @ P_pred @ Hk.T + self.R
            K = (P_pred @ Hk.T) @ np.linalg.solve(S, np.eye(2))

            # Update
            x_hat = x_pred + K @ dy
            P = (I - K @ Hk) @ P_pred @ (I - K @ Hk).T + K @ self.R @ K.T  # Joseph

            X_hat[t] = x_hat
            Ps[t] = P

        return X_hat, Ps
