# models/lorenz.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np


ObsType = Literal["identity", "spherical", "rotated_identity"]


def make_noise_from_nu(r2: float, nu_db: float) -> tuple[float, float]:
    nu = 10.0 ** (nu_db / 10.0)
    q2 = nu * r2
    return q2, r2


def A_lorenz(x: np.ndarray) -> np.ndarray:
    """
    A(x) as in paper Eq.(18):
      dx/dt = A(x) x
    where x = [x1,x2,x3].
    """
    x1 = float(x[0])
    return np.array([
        [-10.0, 10.0, 0.0],
        [28.0, -1.0, -x1],
        [0.0, x1, -(8.0/3.0)]
    ], dtype=float)


def F_taylor(x: np.ndarray, dt: float, J: int) -> np.ndarray:
    """
    F(x) = exp(A(x) dt) approximated by Taylor series of order J:
      I + sum_{j=1..J} (A dt)^j / j!
    """
    A = A_lorenz(x)
    Adt = A * dt
    F = np.eye(3, dtype=float)
    term = np.eye(3, dtype=float)
    fact = 1.0
    for j in range(1, J + 1):
        term = term @ Adt
        fact *= j
        F = F + term / fact
    return F


def f_lorenz_discrete(x: np.ndarray, dt: float, J: int) -> np.ndarray:
    """
    x_{t+1} = F(x_t) x_t  (paper Eq.(21))
    """
    return F_taylor(x, dt, J) @ x


def spherical_from_cart(x: np.ndarray) -> np.ndarray:
    """
    Cartesian (x,y,z) -> spherical (r, theta, phi)
      r = sqrt(x^2+y^2+z^2)
      theta = atan2(y, x)      azimuth
      phi = acos(z/r)         inclination
    """
    x1, x2, x3 = x
    r = np.sqrt(x1*x1 + x2*x2 + x3*x3) + 1e-12
    theta = np.arctan2(x2, x1)
    phi = np.arccos(np.clip(x3 / r, -1.0, 1.0))
    return np.array([r, theta, phi], dtype=float)


def rot3_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class LorenzSSM:
    """
    Discrete Lorenz generator:
      x_{t+1} = f(x_t) + e_t,  e_t ~ N(0, q2 I)
      y_t     = h(x_t) + v_t,  v_t ~ N(0, r2 I)

    obs_type:
      - identity: y = x + v
      - spherical: y = spherical(x) + v
      - rotated_identity: y = R x + v (R is rotation matrix), used for rotation mismatch experiment
    """
    dt: float = 0.02
    J: int = 5
    q2: float = 1e-6
    r2: float = 1e-3
    obs_type: ObsType = "identity"
    rot_deg: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.m = 3
        self.n = 3
        self.Q = self.q2 * np.eye(3, dtype=float)
        self.R = self.r2 * np.eye(3, dtype=float)
        self.Rot = rot3_z(self.rot_deg) if self.obs_type == "rotated_identity" else np.eye(3)

    def f(self, x: np.ndarray) -> np.ndarray:
        return f_lorenz_discrete(x, dt=self.dt, J=self.J)

    def h(self, x: np.ndarray) -> np.ndarray:
        if self.obs_type == "identity":
            return x
        if self.obs_type == "rotated_identity":
            return self.Rot @ x
        if self.obs_type == "spherical":
            return spherical_from_cart(x)
        raise ValueError(f"Unknown obs_type={self.obs_type}")

    def sample(self, T: int, x0: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        if x0 is None:
            x0 = self.rng.normal(size=(3,))
        else:
            x0 = np.asarray(x0, dtype=float).reshape(3,)

        X = np.zeros((T + 1, 3), dtype=float)
        Y = np.zeros((T, 3), dtype=float)
        X[0] = x0

        for t in range(1, T + 1):
            e = self.rng.multivariate_normal(mean=np.zeros(3), cov=self.Q)
            X[t] = self.f(X[t - 1]) + e

            v = self.rng.multivariate_normal(mean=np.zeros(3), cov=self.R)
            Y[t - 1] = self.h(X[t]) + v

        return X, Y


def sample_dense_then_decimate(
    T: int,
    dt_dense: float,
    decim: int,
    x0: np.ndarray,
    process_noise_q2: float,
    obs_noise_r2: float,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sampling mismatch experiment:
    - simulate continuous-time approximately with very small dt_dense using Euler on Lorenz ODE
    - then decimate by factor 'decim' to get dt = dt_dense*decim
    - observations: identity + noise
    - optionally add process noise at the decimated steps (paper says no process noise in that experiment)
    """
    rng = np.random.default_rng(seed)

    def lorenz_ode(x: np.ndarray) -> np.ndarray:
        # classic Lorenz: sigma=10, rho=28, beta=8/3
        x1, x2, x3 = x
        dx1 = 10.0 * (x2 - x1)
        dx2 = x1 * (28.0 - x3) - x2
        dx3 = x1 * x2 - (8.0/3.0) * x3
        return np.array([dx1, dx2, dx3], dtype=float)

    # simulate at dense rate for T * decim steps
    steps = T * decim
    X_dense = np.zeros((steps + 1, 3), dtype=float)
    X_dense[0] = x0

    for k in range(1, steps + 1):
        X_dense[k] = X_dense[k - 1] + dt_dense * lorenz_ode(X_dense[k - 1])

    # decimate
    X = X_dense[::decim].copy()  # shape (T+1,3)

    # add process noise on decimated states if requested
    if process_noise_q2 > 0:
        Q = process_noise_q2 * np.eye(3)
        for t in range(1, T + 1):
            X[t] = X[t] + rng.multivariate_normal(mean=np.zeros(3), cov=Q)

    # observations
    R = obs_noise_r2 * np.eye(3)
    Y = np.zeros((T, 3), dtype=float)
    for t in range(1, T + 1):
        Y[t - 1] = X[t] + rng.multivariate_normal(mean=np.zeros(3), cov=R)

    return X, Y
