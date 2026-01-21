# algos/kalmannet/nclt_sysmodel.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np


def make_wiener_F_H(dt: float = 1.0, device: torch.device | None = None):
    """
    Builds 4D stacked model:
      state: [px,vx,py,vy]
      y: [vx, vy]
    """
    if device is None:
        device = torch.device("cpu")

    F1 = torch.tensor([[1.0, dt],
                       [0.0, 1.0]], dtype=torch.float32, device=device)
    H1 = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)

    F = torch.zeros((4, 4), dtype=torch.float32, device=device)
    H = torch.zeros((2, 4), dtype=torch.float32, device=device)

    F[0:2, 0:2] = F1
    F[2:4, 2:4] = F1
    H[0:1, 0:2] = H1
    H[1:2, 2:4] = H1

    return F, H


@dataclass
class NCLTSysModelTorch:
    """
    Torch-compatible SysModel for KalmanNetArch1.
    f(x) = F x
    h(x) = H x
    """
    F: torch.Tensor
    H: torch.Tensor
    m: int = 4
    n: int = 2

    prior_Q: torch.Tensor = None
    prior_Sigma: torch.Tensor = None
    prior_S: torch.Tensor = None

    @staticmethod
    def build(prior_q2: float, prior_r2: float, dt: float, device: torch.device):
        F, H = make_wiener_F_H(dt=dt, device=device)

        obj = NCLTSysModelTorch(F=F, H=H)
        obj.prior_Q = (prior_q2 * torch.eye(4, device=device)).float()
        obj.prior_Sigma = torch.eye(4, device=device).float()
        obj.prior_S = (prior_r2 * torch.eye(2, device=device)).float()
        return obj

    def f(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,1]
        return torch.bmm(self.F.unsqueeze(0).repeat(x.shape[0], 1, 1), x)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,1]
        return torch.bmm(self.H.unsqueeze(0).repeat(x.shape[0], 1, 1), x)
