# algos/kalmannet/lorenz_sysmodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import torch
import numpy as np

ObsType = Literal["identity", "spherical", "rotated_identity"]


def A_lorenz_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,1]
    returns A(x): [B,3,3]
    """
    x1 = x[:, 0, 0]
    B = x.shape[0]
    A = torch.zeros((B, 3, 3), device=x.device, dtype=x.dtype)
    A[:, 0, 0] = -10.0
    A[:, 0, 1] = 10.0
    A[:, 1, 0] = 28.0
    A[:, 1, 1] = -1.0
    A[:, 1, 2] = -x1
    A[:, 2, 1] = x1
    A[:, 2, 2] = -(8.0 / 3.0)
    return A


def F_taylor_torch(x: torch.Tensor, dt: float, J: int) -> torch.Tensor:
    """
    Taylor approx of exp(A(x) dt) for each batch item.
    Returns [B,3,3].
    """
    A = A_lorenz_torch(x)
    Adt = A * dt
    B = x.shape[0]
    I = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
    F = I.clone()
    term = I.clone()
    fact = 1.0
    for j in range(1, J + 1):
        term = torch.bmm(term, Adt)
        fact *= j
        F = F + term / fact
    return F


def spherical_from_cart_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,1] -> y: [B,3,1] = (r, theta, phi)
    """
    x1 = x[:, 0, 0]
    x2 = x[:, 1, 0]
    x3 = x[:, 2, 0]
    r = torch.sqrt(x1*x1 + x2*x2 + x3*x3 + 1e-12)
    theta = torch.atan2(x2, x1)
    phi = torch.acos(torch.clamp(x3 / r, -1.0, 1.0))
    y = torch.stack([r, theta, phi], dim=1).unsqueeze(-1)
    return y


def rot3_z_torch(deg: float, device, dtype) -> torch.Tensor:
    a = torch.tensor(np.deg2rad(deg), device=device, dtype=dtype)
    c = torch.cos(a); s = torch.sin(a)
    R = torch.stack([
        torch.stack([c, -s, torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([s,  c, torch.tensor(0.0, device=device, dtype=dtype)]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype),
                     torch.tensor(0.0, device=device, dtype=dtype),
                     torch.tensor(1.0, device=device, dtype=dtype)])
    ])
    return R


@dataclass
class LorenzSysModelTorch:
    dt: float
    J: int
    obs_type: ObsType = "identity"
    rot_deg: float = 0.0
    device: Optional[torch.device] = None

    m: int = 3
    n: int = 3

    prior_Q: torch.Tensor = None
    prior_Sigma: torch.Tensor = None
    prior_S: torch.Tensor = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cpu")
        self.prior_Sigma = torch.eye(3, device=self.device).float()
        self._Rot = None

    @staticmethod
    def build(dt: float, J: int, prior_q2: float, prior_r2: float, obs_type: ObsType,
              rot_deg: float, device: torch.device):
        obj = LorenzSysModelTorch(dt=dt, J=J, obs_type=obs_type, rot_deg=rot_deg, device=device)
        obj.prior_Q = (prior_q2 * torch.eye(3, device=device)).float()
        obj.prior_S = (prior_r2 * torch.eye(3, device=device)).float()
        obj.prior_Sigma = torch.eye(3, device=device).float()
        return obj

    def f(self, x: torch.Tensor) -> torch.Tensor:
        F = F_taylor_torch(x, dt=self.dt, J=self.J)
        return torch.bmm(F, x)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        if self.obs_type == "identity":
            return x
        if self.obs_type == "spherical":
            return spherical_from_cart_torch(x)
        if self.obs_type == "rotated_identity":
            if self._Rot is None:
                self._Rot = rot3_z_torch(self.rot_deg, device=x.device, dtype=x.dtype).unsqueeze(0)
            return torch.bmm(self._Rot.repeat(x.shape[0], 1, 1), x)
        raise ValueError(f"Unknown obs_type={self.obs_type}")
