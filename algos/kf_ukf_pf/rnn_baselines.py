# algos/kf_ukf_pf/rnn_baselines.py
from __future__ import annotations
import torch
import torch.nn as nn


class GRUSeq(nn.Module):
    """
    Simple GRU sequence model:
      input:  [B, T, d_in]
      output: [B, T, d_out]
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=d_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_out)

    def forward(self, x_seq: torch.Tensor):
        out, _ = self.gru(x_seq)
        y = self.fc(out)
        return y


class VanillaRNNStateEstimator(nn.Module):
    """
    Vanilla RNN: x_hat_t = RNN(y_1..y_t) (implemented as seq->seq).
    """
    def __init__(self, n: int, m: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.net = GRUSeq(d_in=n, d_hidden=hidden, d_out=m, n_layers=n_layers)

    def forward(self, Y: torch.Tensor, x0_hat: torch.Tensor | None = None):
        # Y: [B,T,n] -> [B,T,m]
        return self.net(Y)


class MBRNNIncrementEstimator(nn.Module):
    """
    MB-RNN:
      prior: x_prior_t = F x_post_{t-1}
      RNN outputs delta: Δx_t
      posterior: x_post_t = x_prior_t + Δx_t

    RNN input each t: concat([y_t, x_prior_t])  -> dimension (n + m)
    RNN output each t: Δx_t -> dimension m
    """
    def __init__(self, F_mat: torch.Tensor, n: int, m: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.register_buffer("F_mat", F_mat.float())  # [m,m]
        self.m = m
        self.n = n
        self.net = GRUSeq(d_in=n + m, d_hidden=hidden, d_out=m, n_layers=n_layers)

    def forward(self, Y: torch.Tensor, x0_hat: torch.Tensor | None = None):
        """
        Y: [B,T,n]
        x0_hat: [B,m] (optional)
        returns X_hat: [B,T,m]
        """
        B, T, _ = Y.shape
        device = Y.device

        if x0_hat is None:
            x_post_prev = torch.zeros(B, self.m, device=device)
        else:
            x_post_prev = x0_hat

        priors = []
        inp = []
        for t in range(T):
            x_prior = (self.F_mat @ x_post_prev.unsqueeze(-1)).squeeze(-1)  # [B,m]
            priors.append(x_prior)
            inp.append(torch.cat([Y[:, t, :], x_prior], dim=1))
            x_post_prev = x_prior  # placeholder; actual update after delta is predicted

        inp_seq = torch.stack(inp, dim=1)      # [B,T,n+m]
        dX = self.net(inp_seq)                 # [B,T,m]
        X_prior = torch.stack(priors, dim=1)   # [B,T,m]
        X_hat = X_prior + dX
        return X_hat
