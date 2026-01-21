# algos/kalmannet/sysmodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class LinearSysModelTorch:
    """
    Torch wrapper for a linear SSM with functions f/h and prior moments used to init KalmanNet hidden states.

    x_t = F x_{t-1} + e_t
    y_t = H x_t + v_t
    """
    F: torch.Tensor  # (m,m)
    H: torch.Tensor  # (n,m)
    m: int
    n: int

    prior_Q: torch.Tensor       # (m,m)
    prior_Sigma: torch.Tensor   # (m,m)
    prior_S: torch.Tensor       # (n,n)

    device: torch.device

    @staticmethod
    def from_numpy(F, H, m, n, prior_q2=1e-3, prior_r2=1e-2, device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cpu")

        # Ensure arrays are contiguous (fixes negative stride issue)
        F = np.ascontiguousarray(F)
        H = np.ascontiguousarray(H)

        F_t = torch.tensor(F, dtype=torch.float32, device=device)
        H_t = torch.tensor(H, dtype=torch.float32, device=device)

        prior_Q = (prior_q2 * torch.eye(m, device=device)).float()
        prior_Sigma = (torch.eye(m, device=device)).float()
        prior_S = (prior_r2 * torch.eye(n, device=device)).float()

        return LinearSysModelTorch(
            F=F_t, H=H_t, m=m, n=n,
            prior_Q=prior_Q, prior_Sigma=prior_Sigma, prior_S=prior_S,
            device=device
        )

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, m, 1]
        returns: [B, m, 1]
        """
        return torch.bmm(self.F.unsqueeze(0).expand(x.shape[0], -1, -1), x)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, m, 1]
        returns: [B, n, 1]
        """
        return torch.bmm(self.H.unsqueeze(0).expand(x.shape[0], -1, -1), x)
