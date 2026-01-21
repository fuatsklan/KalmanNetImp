# algos/improvements/lorenz_kalmannet_arch1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn


@dataclass
class ImpKNetArch1Args:
    use_cuda: bool = False
    n_batch: int = 32
    hidden_mult: int = 10
    n_rnn_layers: int = 1
    rnn_type: str = "lstm"          # {"gru","lstm"}
    feat_norm_eps: float = 1e-3
    enable_nan_guards: bool = False


def safe_unit_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / (denom + eps)


class ImprovedKalmanNetArch1(nn.Module):
    """
    Arch1 but with GRU/LSTM selectable.
    Same KF-like flow:
      x_prior = f(x_post)
      y_pred  = h(x_prior)
      x_post  = x_prior + K(y - y_pred)
    """
    def __init__(self, feature_set: List[str]):
        super().__init__()
        self.feature_set = feature_set

    def NNBuild(self, SysModel, args: ImpKNetArch1Args):
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

        rnn_type = args.rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.d_hidden,
                hidden_size=self.d_hidden,
                num_layers=args.n_rnn_layers,
            ).to(self.device)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.d_hidden,
                hidden_size=self.d_hidden,
                num_layers=args.n_rnn_layers,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown rnn_type={args.rnn_type}. Use 'gru' or 'lstm'.")

        self.fc_out = nn.Linear(self.d_hidden, self.d_out).to(self.device)

        # start with K ~ 0
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        self.to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.h = h
        self.m = m
        self.n = n

    def _is_lstm(self) -> bool:
        return self.args.rnn_type.lower() == "lstm"

    def init_hidden(self, batch_size: int):
        self.batch_size = batch_size
        nl = self.args.n_rnn_layers
        h = torch.zeros(nl, batch_size, self.d_hidden, device=self.device)
        if self._is_lstm():
            c = torch.zeros(nl, batch_size, self.d_hidden, device=self.device)
            self.h_rnn = (h, c)
        else:
            self.h_rnn = h

    def InitSequence(self, x0_hat: torch.Tensor, T: int):
        self.T = T
        self.m1x_posterior = x0_hat.to(self.device).detach()
        self.m1x_posterior_prev = self.m1x_posterior.clone().detach()
        self.m1x_prior_prev = self.m1x_posterior.clone().detach()
        self.y_prev = self.h(self.m1x_posterior).clone().detach()
        self.y_pred = self.y_prev.clone().detach()
        self.KGain = torch.zeros((self.batch_size, self.m, self.n), device=self.device)

    def step_prior(self):
        # keeping no_grad is OK for stability on Lorenz
        with torch.no_grad():
            self.m1x_prior = self.f(self.m1x_posterior)
            self.y_pred = self.h(self.m1x_prior)
        self.m1x_prior = self.m1x_prior.detach()
        self.y_pred = self.y_pred.detach()

    def _build_features(self, y: torch.Tensor) -> torch.Tensor:
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_prev, 2)        # F1
        innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_pred, 2)      # F2
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
        return torch.cat(blocks, dim=1)

    def step_KGain(self, y: torch.Tensor):
        feats = self._build_features(y)        # [B, d_in]
        z = self.fc_in(feats)                  # [B, d_hidden]
        z = z.unsqueeze(0)                     # [1, B, d_hidden]

        if self._is_lstm():
            out, self.h_rnn = self.rnn(z, self.h_rnn)
        else:
            out, self.h_rnn = self.rnn(z, self.h_rnn)

        kg_vec = self.fc_out(out[0])           # [B, m*n]
        self.KGain = kg_vec.view(self.batch_size, self.m, self.n)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)

        self.step_prior()
        self.step_KGain(y)

        dy = y - self.y_pred
        inov = torch.bmm(self.KGain, dy)

        if self.args.enable_nan_guards:
            if not torch.isfinite(self.KGain).all():
                raise RuntimeError("NaN/Inf in KGain")
            if not torch.isfinite(inov).all():
                raise RuntimeError("NaN/Inf in inov")

        self.m1x_posterior_prev = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y
        return self.m1x_posterior
