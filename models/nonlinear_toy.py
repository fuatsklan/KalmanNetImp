# models/nonlinear_toy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


def make_noise_from_nu(r2: float, nu_db: float):
    nu = 10.0 ** (nu_db / 10.0)
    q2 = nu * r2
    return q2, r2


@dataclass
class NonlinearToySSM:
    """
    x_t = f(x_{t-1}) + e_t,  e_t ~ N(0, q2 I)
    y_t = h(x_t)     + v_t,  v_t ~ N(0, r2 I)

    x, y are R^2 and f,h are applied component-wise:
      f(x) = alpha*sin(beta*x + phi) + delta
      h(x) = a*(b*x + c)^2
    """
    params_true: Dict[str, float]
    q2: float
    r2: float
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.m = 2
        self.n = 2
        self.Q = self.q2 * np.eye(self.m, dtype=float)
        self.R = self.r2 * np.eye(self.n, dtype=float)

    def f(self, x: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        p = self.params_true if params is None else params
        alpha, beta, phi, delta = p["alpha"], p["beta"], p["phi"], p["delta"]
        return alpha * np.sin(beta * x + phi) + delta

    def h(self, x: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        p = self.params_true if params is None else params
        a, b, c = p["a"], p["b"], p["c"]
        return a * (b * x + c) ** 2

    def sample(self, T: int, x0: Optional[np.ndarray] = None):
        if x0 is None:
            x0 = self.rng.normal(size=(2,))
        else:
            x0 = np.asarray(x0, dtype=float).reshape(2,)

        X = np.zeros((T + 1, 2), dtype=float)
        Y = np.zeros((T, 2), dtype=float)
        X[0] = x0

        for t in range(1, T + 1):
            e = self.rng.multivariate_normal(mean=np.zeros(2), cov=self.Q)
            X[t] = self.f(X[t - 1]) + e

            v = self.rng.multivariate_normal(mean=np.zeros(2), cov=self.R)
            Y[t - 1] = self.h(X[t]) + v

        return X, Y
