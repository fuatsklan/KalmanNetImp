# models/linear_ssm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np


CanonicalForm = Literal["controllable_canonical", "random_stable"]
HForm = Literal["identity", "inverse_canonical", "last_state"]


def rotation_2d(deg: float) -> np.ndarray:
    """2D rotation matrix Rxy_alpha."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def companion_from_roots(roots: np.ndarray) -> np.ndarray:
    """
    Build a controllable companion matrix whose eigenvalues are the given roots.
    Uses polynomial coefficients from roots, then forms companion matrix.
    """
    # poly gives coefficients for: z^m + a1 z^(m-1) + ... + am
    poly = np.poly(roots)  # length m+1, leading coefficient = 1
    # companion matrix uses the NEGATIVE of polynomial coefficients (excluding leading 1)
    a = -poly[1:]  # length m
    m = len(a)
    F = np.zeros((m, m), dtype=float)
    F[:-1, 1:] = np.eye(m - 1)
    F[-1, :] = a[::-1]  # reverse so it matches canonical companion form
    return F


def make_F(m: int, form: CanonicalForm = "controllable_canonical", seed: Optional[int] = None) -> np.ndarray:
    """
    Default: a stable controllable canonical (companion) matrix.
    """
    rng = np.random.default_rng(seed)

    if m == 1:
        # simple stable scalar
        return np.array([[0.9]], dtype=float)

    if form == "controllable_canonical":
        # Sample stable roots inside unit circle.
        # For real matrix: create conjugate pairs when using complex roots.
        roots = []
        i = 0
        while i < m:
            if i <= m - 2 and rng.uniform() < 0.5:
                # complex conjugate pair
                r = rng.uniform(0.2, 0.95)
                theta = rng.uniform(0, np.pi)
                z = r * (np.cos(theta) + 1j * np.sin(theta))
                roots.append(z)
                roots.append(np.conj(z))
                i += 2
            else:
                roots.append(rng.uniform(0.2, 0.95))
                i += 1
        roots = np.array(roots[:m])
        F = companion_from_roots(roots)
        return F

    if form == "random_stable":
        A = rng.normal(size=(m, m))
        # scale to have spectral radius < 1
        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        A = A / (1.1 * rho)
        return A

    raise ValueError(f"Unknown F form: {form}")


def make_H(n: int, m: int, form: HForm = "inverse_canonical") -> np.ndarray:
    """
    Observation matrix. The paper says "inverse canonical form" in the linear full-info case.
    There's some ambiguity across literature; here are practical options:

    - identity: if n==m, H=I
    - inverse_canonical: reversed identity-like mapping (commonly used as "inverse" / flipped canonical)
    - last_state: observe last state only (useful for n=1)
    """
    if form == "identity":
        if n != m:
            raise ValueError("identity H requires n == m")
        return np.eye(m, dtype=float)

    if form == "inverse_canonical":
        if n == m:
            return np.fliplr(np.eye(m, dtype=float))  # reversed identity
        # rectangular: take first n rows of flipped identity on m
        return np.fliplr(np.eye(m, dtype=float))[:n, :]

    if form == "last_state":
        if n != 1:
            raise ValueError("last_state form is intended for n==1")
        H = np.zeros((1, m), dtype=float)
        H[0, -1] = 1.0
        return H

    raise ValueError(f"Unknown H form: {form}")


@dataclass
class LinearSSM:
    """
    x_t = F x_{t-1} + e_t,  e_t ~ N(0, Q)
    y_t = H x_t       + v_t, v_t ~ N(0, R)

    Q = q2 I, R = r2 I (matches the paper’s simplified noise setting)
    """
    F: np.ndarray  # (m,m)
    H: np.ndarray  # (n,m)
    q2: float
    r2: float
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

        m = self.F.shape[0]
        n = self.H.shape[0]

        assert self.F.shape == (m, m)
        assert self.H.shape[1] == m

        self.m = m
        self.n = n

        self.Q = self.q2 * np.eye(m, dtype=float)
        self.R = self.r2 * np.eye(n, dtype=float)

    def sample(self, T: int, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          X: (T+1, m)  includes x0
          Y: (T, n)
        """
        if x0 is None:
            x0 = self.rng.normal(size=(self.m,))
        else:
            x0 = np.asarray(x0, dtype=float).reshape(self.m,)

        X = np.zeros((T + 1, self.m), dtype=float)
        Y = np.zeros((T, self.n), dtype=float)
        X[0] = x0

        for t in range(1, T + 1):
            e = self.rng.multivariate_normal(mean=np.zeros(self.m), cov=self.Q)
            X[t] = self.F @ X[t - 1] + e

            v = self.rng.multivariate_normal(mean=np.zeros(self.n), cov=self.R)
            Y[t - 1] = self.H @ X[t] + v

        return X, Y


def make_noise_from_nu(r2: float, nu_db: float) -> Tuple[float, float]:
    """
    The paper uses nu = q^2 / r^2 (often in dB).
    Given r2 and nu_db, compute q2.
    """
    nu = 10.0 ** (nu_db / 10.0)
    q2 = nu * r2
    return q2, r2
