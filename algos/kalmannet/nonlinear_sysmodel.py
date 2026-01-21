# algos/kalmannet/nonlinear_sysmodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch


@dataclass
class NonlinearToySysModelTorch:
    """
    Torch wrapper with f/h for KalmanNet. x in [B,2,1], y in [B,2,1]
    """
    params_design: Dict[str, float]
    device: torch.device

    m: int = 2
    n: int = 2

    prior_Q: torch.Tensor = None
    prior_Sigma: torch.Tensor = None
    prior_S: torch.Tensor = None

    @staticmethod
    def from_numpy_params(params_design: Dict[str, float], prior_q2: float, prior_r2: float, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")

        prior_Q = (prior_q2 * torch.eye(2, device=device)).float()
        prior_Sigma = (torch.eye(2, device=device)).float()
        prior_S = (prior_r2 * torch.eye(2, device=device)).float()

        return NonlinearToySysModelTorch(
            params_design=params_design,
            device=device,
            prior_Q=prior_Q,
            prior_Sigma=prior_Sigma,
            prior_S=prior_S
        )

    def f(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,2,1]
        p = self.params_design
        alpha = float(p["alpha"]); beta = float(p["beta"]); phi = float(p["phi"]); delta = float(p["delta"])
        return alpha * torch.sin(beta * x + phi) + delta

    def h(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,2,1]
        p = self.params_design
        a = float(p["a"]); b = float(p["b"]); c = float(p["c"])
        return a * (b * x + c) ** 2
