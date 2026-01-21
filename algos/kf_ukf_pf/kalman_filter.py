# algos/kf_ukf_pf/kalman_filter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KalmanFilter:
    F: np.ndarray  # (m,m)
    H: np.ndarray  # (n,m)
    Q: np.ndarray  # (m,m)
    R: np.ndarray  # (n,n)

    def __post_init__(self) -> None:
        self.m = self.F.shape[0]
        self.n = self.H.shape[0]
        assert self.F.shape == (self.m, self.m)
        assert self.H.shape == (self.n, self.m)
        assert self.Q.shape == (self.m, self.m)
        assert self.R.shape == (self.n, self.n)

    def run(
        self,
        Y: np.ndarray,                       # (T,n)
        x0_hat: Optional[np.ndarray] = None, # (m,)
        P0: Optional[np.ndarray] = None      # (m,m)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          X_hat: (T,m) posterior means
          P:     (T,m,m) posterior covariances
        """
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]
        assert Y.shape[1] == self.n

        if x0_hat is None:
            x_hat = np.zeros((self.m,), dtype=float)
        else:
            x_hat = np.asarray(x0_hat, dtype=float).reshape(self.m,)

        if P0 is None:
            P = np.eye(self.m, dtype=float)
        else:
            P = np.asarray(P0, dtype=float).reshape(self.m, self.m)

        X_hat = np.zeros((T, self.m), dtype=float)
        Ps = np.zeros((T, self.m, self.m), dtype=float)

        I = np.eye(self.m, dtype=float)

        for t in range(T):
            # 1) Predict
            x_pred = self.F @ x_hat
            P_pred = self.F @ P @ self.F.T + self.Q

            # 2) Innovation
            y_pred = self.H @ x_pred
            dy = Y[t] - y_pred

            S = self.H @ P_pred @ self.H.T + self.R

            # 3) Kalman Gain (use solve instead of explicit inverse for stability)
            # K = P_pred H^T S^{-1}
            K = (P_pred @ self.H.T) @ np.linalg.solve(S, np.eye(self.n))

            # 4) Update
            x_hat = x_pred + K @ dy
            P = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T  # Joseph form

            X_hat[t] = x_hat
            Ps[t] = P

        return X_hat, Ps
