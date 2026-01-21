# algos/kf_ukf_pf/particle_filter_nonlinear_toy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class ParticleFilterNonlinearToy:
    params_design: Dict[str, float]
    Q: np.ndarray
    R: np.ndarray
    n_particles: int = 100
    resample_threshold: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def f(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        return p["alpha"] * np.sin(p["beta"] * x + p["phi"]) + p["delta"]

    def h(self, x: np.ndarray) -> np.ndarray:
        p = self.params_design
        return p["a"] * (p["b"] * x + p["c"]) ** 2

    def gaussian_logpdf(self, y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        # y, mean shape (2,)
        d = y - mean
        inv = np.linalg.inv(cov)
        return -0.5 * (d.T @ inv @ d)  # ignore constant; cancels in weights

    def systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        N = len(weights)
        positions = (self.rng.random() + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)

        cumsum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def run(self, Y: np.ndarray, x0_hat: Optional[np.ndarray] = None) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]

        if x0_hat is None:
            x0_hat = np.zeros(2, dtype=float)
        else:
            x0_hat = np.asarray(x0_hat, dtype=float).reshape(2,)

        # init particles around x0_hat
        particles = self.rng.multivariate_normal(mean=x0_hat, cov=np.eye(2), size=self.n_particles)
        weights = np.ones(self.n_particles, dtype=float) / self.n_particles

        X_hat = np.zeros((T, 2), dtype=float)

        for t in range(T):
            # propagate
            noise = self.rng.multivariate_normal(mean=np.zeros(2), cov=self.Q, size=self.n_particles)
            particles = np.array([self.f(particles[i]) for i in range(self.n_particles)]) + noise

            # weight by likelihood
            logw = np.array([self.gaussian_logpdf(Y[t], self.h(particles[i]), self.R) for i in range(self.n_particles)])
            logw -= np.max(logw)
            w = np.exp(logw)
            w /= np.sum(w)
            weights = w

            # estimate
            X_hat[t] = np.sum(weights[:, None] * particles, axis=0)

            # resample if needed
            ess = 1.0 / np.sum(weights ** 2)
            if ess < self.resample_threshold * self.n_particles:
                idx = self.systematic_resample(weights)
                particles = particles[idx]
                weights = np.ones(self.n_particles, dtype=float) / self.n_particles

        return X_hat
