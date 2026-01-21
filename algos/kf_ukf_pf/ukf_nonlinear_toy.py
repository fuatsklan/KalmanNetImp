# algos/kf_ukf_pf/ukf_nonlinear_toy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class UKFNonlinearToy:
    """
    Unscented Kalman Filter (UKF) for the toy nonlinear model in R^2.

    Uses Julier sigma points with parameters (alpha, beta, kappa).
    """
    params_design: Dict[str, float]
    Q: np.ndarray  # (2,2)
    R: np.ndarray  # (2,2)

    alpha: float = 1e-1
    beta: float = 2.0
    kappa: float = 0.0

    def f(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        return p["alpha"] * np.sin(p["beta"] * x + p["phi"]) + p["delta"]

    def h(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        return p["a"] * (p["b"] * x + p["c"]) ** 2

    def sigma_points(self, x: np.ndarray, P: np.ndarray):
        n = x.shape[0]
        lam = (self.alpha ** 2) * (n + self.kappa) - n
        c = n + lam

        # sqrt matrix
        S = np.linalg.cholesky(c * P + 1e-12 * np.eye(n))

        X = np.zeros((2 * n + 1, n), dtype=float)
        X[0] = x
        for i in range(n):
            X[i + 1] = x + S[:, i]
            X[n + i + 1] = x - S[:, i]

        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
        Wm[0] = lam / c
        Wc[0] = lam / c + (1 - self.alpha ** 2 + self.beta)
        return X, Wm, Wc

    def run(self, Y: np.ndarray, x0_hat: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]

        if x0_hat is None:
            x = np.zeros(2, dtype=float)
        else:
            x = np.asarray(x0_hat, dtype=float).reshape(2,)

        if P0 is None:
            P = np.eye(2, dtype=float)
        else:
            P = np.asarray(P0, dtype=float).reshape(2, 2)

        X_hat = np.zeros((T, 2), dtype=float)
        I = np.eye(2)

        for t in range(T):
            # --- Predict ---
            Xsig, Wm, Wc = self.sigma_points(x, P)
            Xsig_pred = np.array([self.f(xi) for xi in Xsig])

            x_pred = np.sum(Wm[:, None] * Xsig_pred, axis=0)
            P_pred = self.Q.copy()
            for i in range(Xsig_pred.shape[0]):
                dx = (Xsig_pred[i] - x_pred).reshape(2, 1)
                P_pred += Wc[i] * (dx @ dx.T)

            # --- Update ---
            Ysig = np.array([self.h(xi) for xi in Xsig_pred])
            y_pred = np.sum(Wm[:, None] * Ysig, axis=0)

            S = self.R.copy()
            for i in range(Ysig.shape[0]):
                dy = (Ysig[i] - y_pred).reshape(2, 1)
                S += Wc[i] * (dy @ dy.T)

            Cxy = np.zeros((2, 2), dtype=float)
            for i in range(Ysig.shape[0]):
                dx = (Xsig_pred[i] - x_pred).reshape(2, 1)
                dy = (Ysig[i] - y_pred).reshape(2, 1)
                Cxy += Wc[i] * (dx @ dy.T)

            K = Cxy @ np.linalg.solve(S, I)

            innov = Y[t] - y_pred
            x = x_pred + K @ innov
            P = P_pred - K @ S @ K.T

            X_hat[t] = x

        return X_hat
