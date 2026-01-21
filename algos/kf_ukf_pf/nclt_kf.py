# algos/kf_ukf_pf/nclt_kf.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def make_F(dt=1.0):
    return np.array([[1.0, dt],
                     [0.0, 1.0]], dtype=float)

def make_H():
    return np.array([[0.0, 1.0]], dtype=float)

def make_Q(dt: float, q2: float):
    # Eq.(23): q^2 * [[1/3 dt^3, 1/2 dt^2], [1/2 dt^2, dt]]
    return q2 * np.array([[ (dt**3)/3.0, (dt**2)/2.0 ],
                          [ (dt**2)/2.0, dt ]], dtype=float)

def make_stacked(F1, H1, Q1, r2):
    F = np.zeros((4,4), dtype=float)
    H = np.zeros((2,4), dtype=float)
    Q = np.zeros((4,4), dtype=float)
    R = r2 * np.eye(2, dtype=float)

    F[0:2,0:2] = F1
    F[2:4,2:4] = F1
    H[0:1,0:2] = H1
    H[1:2,2:4] = H1
    Q[0:2,0:2] = Q1
    Q[2:4,2:4] = Q1
    return F, H, Q, R


@dataclass
class KFLinear:
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    def run(self, Y: np.ndarray, x0: np.ndarray, P0: np.ndarray | None = None):
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]
        m = self.F.shape[0]
        n = self.H.shape[0]

        x = np.asarray(x0, dtype=float).reshape(m, 1)
        P = np.eye(m) if P0 is None else np.asarray(P0, dtype=float).reshape(m, m)

        X_hat = np.zeros((T, m), dtype=float)
        I = np.eye(m)

        for t in range(T):
            # predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # update
            y = Y[t].reshape(n, 1)
            y_pred = self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            x = x_pred + K @ (y - y_pred)
            P = (I - K @ self.H) @ P_pred

            X_hat[t] = x.squeeze(-1)

        return X_hat
