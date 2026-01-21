# algos/kf_ukf_pf/nonlinear_filters_generic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


def jacobian_fd(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Finite-difference Jacobian of func at x.
    func: R^m -> R^n
    returns J: (n,m)
    """
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(func(x), dtype=float)
    n = y0.size
    m = x.size
    J = np.zeros((n, m), dtype=float)

    for i in range(m):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        yp = np.asarray(func(xp), dtype=float)
        ym = np.asarray(func(xm), dtype=float)
        J[:, i] = (yp - ym) / (2 * eps)
    return J


@dataclass
class EKFGeneric:
    f: Callable[[np.ndarray], np.ndarray]  # R^m -> R^m
    h: Callable[[np.ndarray], np.ndarray]  # R^m -> R^n
    Q: np.ndarray                          # (m,m)
    R: np.ndarray                          # (n,n)
    eps_jac: float = 1e-5

    def run(self, Y: np.ndarray, x0_hat: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None):
        Y = np.asarray(Y, dtype=float)
        T, n = Y.shape
        m = self.Q.shape[0]

        x = np.zeros(m) if x0_hat is None else np.asarray(x0_hat, dtype=float).reshape(m,)
        P = np.eye(m) if P0 is None else np.asarray(P0, dtype=float).reshape(m, m)

        X_hat = np.zeros((T, m), dtype=float)
        I_m = np.eye(m)

        for t in range(T):
            # Predict
            Fk = jacobian_fd(self.f, x, eps=self.eps_jac)
            x_pred = self.f(x)
            P_pred = Fk @ P @ Fk.T + self.Q

            # Update
            Hk = jacobian_fd(self.h, x_pred, eps=self.eps_jac)
            y_pred = self.h(x_pred)
            dy = Y[t] - y_pred

            S = Hk @ P_pred @ Hk.T + self.R
            K = (P_pred @ Hk.T) @ np.linalg.solve(S, np.eye(n))

            x = x_pred + K @ dy
            P = (I_m - K @ Hk) @ P_pred @ (I_m - K @ Hk).T + K @ self.R @ K.T

            X_hat[t] = x

        return X_hat


@dataclass
class UKFGeneric:
    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    alpha: float = 1e-1
    beta: float = 2.0
    kappa: float = 0.0

    def sigma_points(self, x: np.ndarray, P: np.ndarray):
        n = x.size
        lam = (self.alpha ** 2) * (n + self.kappa) - n
        c = n + lam
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

    def run(self, Y: np.ndarray, x0_hat: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None):
        Y = np.asarray(Y, dtype=float)
        T, ny = Y.shape
        nx = self.Q.shape[0]

        x = np.zeros(nx) if x0_hat is None else np.asarray(x0_hat, dtype=float).reshape(nx,)
        P = np.eye(nx) if P0 is None else np.asarray(P0, dtype=float).reshape(nx, nx)
        X_hat = np.zeros((T, nx), dtype=float)
        I_y = np.eye(ny)

        for t in range(T):
            # Predict
            Xsig, Wm, Wc = self.sigma_points(x, P)
            Xsig_pred = np.array([self.f(xi) for xi in Xsig])

            x_pred = np.sum(Wm[:, None] * Xsig_pred, axis=0)
            P_pred = self.Q.copy()
            for i in range(Xsig_pred.shape[0]):
                dx = (Xsig_pred[i] - x_pred).reshape(nx, 1)
                P_pred += Wc[i] * (dx @ dx.T)

            # Update
            Ysig = np.array([self.h(xi) for xi in Xsig_pred])
            y_pred = np.sum(Wm[:, None] * Ysig, axis=0)

            S = self.R.copy()
            for i in range(Ysig.shape[0]):
                dy = (Ysig[i] - y_pred).reshape(ny, 1)
                S += Wc[i] * (dy @ dy.T)

            Cxy = np.zeros((nx, ny), dtype=float)
            for i in range(Ysig.shape[0]):
                dx = (Xsig_pred[i] - x_pred).reshape(nx, 1)
                dy = (Ysig[i] - y_pred).reshape(ny, 1)
                Cxy += Wc[i] * (dx @ dy.T)

            K = Cxy @ np.linalg.solve(S, I_y)
            innov = Y[t] - y_pred
            x = x_pred + K @ innov
            P = P_pred - K @ S @ K.T

            X_hat[t] = x

        return X_hat


@dataclass
class ParticleFilterGeneric:
    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    n_particles: int = 100
    resample_threshold: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.nx = self.Q.shape[0]

    def loglik(self, y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        d = y - mean
        inv = np.linalg.inv(cov)
        return -0.5 * float(d.T @ inv @ d)

    def systematic_resample(self, w: np.ndarray) -> np.ndarray:
        N = len(w)
        positions = (self.rng.random() + np.arange(N)) / N
        idx = np.zeros(N, dtype=int)
        cumsum = np.cumsum(w)
        i = j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1
        return idx

    def run(self, Y: np.ndarray, x0_hat: Optional[np.ndarray] = None):
        Y = np.asarray(Y, dtype=float)
        T, ny = Y.shape

        x0 = np.zeros(self.nx) if x0_hat is None else np.asarray(x0_hat, dtype=float).reshape(self.nx,)
        particles = self.rng.multivariate_normal(mean=x0, cov=np.eye(self.nx), size=self.n_particles)
        w = np.ones(self.n_particles) / self.n_particles
        X_hat = np.zeros((T, self.nx), dtype=float)

        for t in range(T):
            noise = self.rng.multivariate_normal(mean=np.zeros(self.nx), cov=self.Q, size=self.n_particles)
            particles = np.array([self.f(particles[i]) for i in range(self.n_particles)]) + noise

            logw = np.array([self.loglik(Y[t], self.h(particles[i]), self.R) for i in range(self.n_particles)])
            logw -= np.max(logw)
            w = np.exp(logw)
            w /= np.sum(w)

            X_hat[t] = np.sum(w[:, None] * particles, axis=0)

            ess = 1.0 / np.sum(w ** 2)
            if ess < self.resample_threshold * self.n_particles:
                idx = self.systematic_resample(w)
                particles = particles[idx]
                w = np.ones(self.n_particles) / self.n_particles

        return X_hat
