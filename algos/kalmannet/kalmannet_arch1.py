# algos/kalmannet/kalmannet_arch1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn


@dataclass
class KNetArch1Args:
    use_cuda: bool = False
    n_batch: int = 32
    hidden_mult: int = 10
    n_gru_layers: int = 1
    # ✅ critical for NCLT: avoid 0/0 amplification
    feat_norm_eps: float = 1e-3
    enable_nan_guards: bool = False
    kgain_tanh_scale: float = 1.0


def safe_unit_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Normalize along dim=1 with a *safe* epsilon that prevents
    huge amplification when x is near-zero (common in NCLT at 1Hz).
    """
    denom = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / (denom + eps)


class KalmanNetArch1(nn.Module):
    """
    KalmanNet Architecture #1 (paper Fig.3 style):
      FC_in -> GRU -> FC_out -> K_t
    """
    def __init__(self, feature_set: List[str]):
        super().__init__()
        self.feature_set = feature_set

    def NNBuild(self, SysModel, args: KNetArch1Args):
        self.device = torch.device("cuda") if (args.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.args = args
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        d_in = 0
        for f in self.feature_set:
            if f in ("F1", "F2"):
                d_in += self.n
            elif f in ("F3", "F4"):
                d_in += self.m
            else:
                raise ValueError(f"Unknown feature {f}")
        self.d_in = d_in

        self.d_hidden = args.hidden_mult * (self.m**2 + self.n**2)
        self.d_out = self.m * self.n

        self.fc_in = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden),
            nn.ReLU()
        ).to(self.device)

        self.gru = nn.GRU(
            input_size=self.d_hidden,
            hidden_size=self.d_hidden,
            num_layers=args.n_gru_layers,
        ).to(self.device)

        self.fc_out = nn.Linear(self.d_hidden, self.d_out).to(self.device)

        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        self.to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.h = h
        self.m = m
        self.n = n

    def init_hidden(self, batch_size: int):
        self.batch_size = batch_size
        self.h_gru = torch.zeros(self.args.n_gru_layers, batch_size, self.d_hidden, device=self.device)

    def InitSequence(self, x0_hat: torch.Tensor, T: int):
        self.T = T
        self.m1x_posterior = x0_hat.to(self.device).detach()
        self.m1x_posterior_prev = self.m1x_posterior.clone().detach()
        self.m1x_prior_prev = self.m1x_posterior.clone().detach()
        self.y_prev = self.h(self.m1x_posterior).clone().detach()
        self.y_pred = self.y_prev.clone().detach()

    def step_prior(self):
        # For linear NCLT this is stable even with grad, but keep no_grad for consistency
        with torch.no_grad():
            self.m1x_prior = self.f(self.m1x_posterior)
            self.y_pred = self.h(self.m1x_prior)
        self.m1x_prior = self.m1x_prior.detach()
        self.y_pred = self.y_pred.detach()

    def _build_features(self, y: torch.Tensor) -> torch.Tensor:
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_prev, 2)          # F1
        innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_pred, 2)        # F2
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_prev, 2)  # F3
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_prev, 2)    # F4

        eps = float(self.args.feat_norm_eps)
        obs_diff = safe_unit_norm(obs_diff, eps)
        innov_diff = safe_unit_norm(innov_diff, eps)
        fw_evol_diff = safe_unit_norm(fw_evol_diff, eps)
        fw_update_diff = safe_unit_norm(fw_update_diff, eps)

        blocks = []
        for ftr in self.feature_set:
            if ftr == "F1":
                blocks.append(obs_diff)
            elif ftr == "F2":
                blocks.append(innov_diff)
            elif ftr == "F3":
                blocks.append(fw_evol_diff)
            elif ftr == "F4":
                blocks.append(fw_update_diff)

        return torch.cat(blocks, dim=1)  # [B,d_in]

    def step_KGain(self, y: torch.Tensor):
        feats = self._build_features(y)              # [B,d_in]
        z = self.fc_in(feats)                        # [B,d_hidden]
        z = z.unsqueeze(0)                           # [1,B,d_hidden]
        out, self.h_gru = self.gru(z, self.h_gru)    # [1,B,d_hidden]
        kg_vec = self.fc_out(out[0])  # [B, m*n]
        s = float(self.args.kgain_tanh_scale)
        if s and s > 0:
            kg_vec = torch.tanh(kg_vec) * s
                    # [B,m*n]
        self.KGain = kg_vec.view(self.batch_size, self.m, self.n)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)

        self.step_prior()
        self.step_KGain(y)

        dy = y - self.y_pred                          # [B,n,1]
        inov = torch.bmm(self.KGain, dy)              # [B,m,1]

        if self.args.enable_nan_guards:
            if not torch.isfinite(dy).all():
                raise RuntimeError("NaN/Inf in innovation dy")
            if not torch.isfinite(self.KGain).all():
                raise RuntimeError("NaN/Inf in KGain")
            if not torch.isfinite(inov).all():
                raise RuntimeError("NaN/Inf in inov")
            if not torch.isfinite(self.m1x_prior).all():
                raise RuntimeError("NaN/Inf in m1x_prior")

        self.m1x_posterior_prev = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y

        return self.m1x_posterior
