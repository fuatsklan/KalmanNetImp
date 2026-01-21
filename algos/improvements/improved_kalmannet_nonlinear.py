from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImprovedNLKNetArgs:
    use_cuda: bool = False
    n_batch: int = 32
    in_mult: int = 5
    out_mult: int = 2

    # improvement knobs
    rnn_type: str = "lstm"  # {"gru","lstm"}


class ImprovedNonlinearKalmanNet(nn.Module):
    """
    Nonlinear-compatible "Improved KalmanNet":
      - same KF-style flow as KalmanNet (prior via f, predicted measurement via h)
      - RNN predicts K_t directly (no analytic KF gain, since no fixed F/H)
      - RNN blocks can be GRU or LSTM
    """
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args: ImprovedNLKNetArgs):
        self.device = torch.device("cuda") if (args.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.args = args

        self.f = SysModel.f
        self.h = SysModel.h
        self.m = SysModel.m
        self.n = SysModel.n

        self._init_gain_net(args)
        self.to(self.device)

    # -------------------
    # Gain net
    # -------------------
    def _is_lstm(self):
        return self.args.rnn_type.lower() == "lstm"

    def _make_rnn(self, input_dim: int, hidden_dim: int):
        t = self.args.rnn_type.lower()
        if t == "gru":
            return nn.GRU(input_dim, hidden_dim)
        if t == "lstm":
            return nn.LSTM(input_dim, hidden_dim)
        raise ValueError(f"Unknown rnn_type={self.args.rnn_type}")

    def _init_gain_net(self, args: ImprovedNLKNetArgs):
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        # RNN to track "Q-like" hidden (paper-style)
        self.d_input_Q = self.m * args.in_mult
        self.d_hidden_Q = self.m ** 2
        self.RNN_Q = self._make_rnn(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # RNN to track "Sigma-like" hidden
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.RNN_Sigma = self._make_rnn(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # RNN to track "S-like" hidden
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult
        self.d_hidden_S = self.n ** 2
        self.RNN_S = self._make_rnn(self.d_input_S, self.d_hidden_S).to(self.device)

        # FC heads
        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, self.n ** 2), nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.n * self.m),
        ).to(self.device)

        # feature encoders
        self.FC5 = nn.Sequential(nn.Linear(self.m, self.m * args.in_mult), nn.ReLU()).to(self.device)
        self.FC6 = nn.Sequential(nn.Linear(self.m, self.m * args.in_mult), nn.ReLU()).to(self.device)
        self.FC7 = nn.Sequential(nn.Linear(2 * self.n, 2 * self.n * args.in_mult), nn.ReLU()).to(self.device)

        # for optional gain smoothness regularization
        self.KGain = None

    # -------------------
    # Hidden init
    # -------------------
    def init_hidden(self, batch_size: int):
        self.batch_size = batch_size
        # use zeros as priors in nonlinear case (simple + stable)
        hQ = torch.zeros((1, batch_size, self.m ** 2), device=self.device)
        hS = torch.zeros((1, batch_size, self.n ** 2), device=self.device)
        hSi = torch.zeros((1, batch_size, self.m ** 2), device=self.device)

        if self._is_lstm():
            self.h_Q = (hQ, torch.zeros_like(hQ))
            self.h_S = (hS, torch.zeros_like(hS))
            self.h_Sigma = (hSi, torch.zeros_like(hSi))
        else:
            self.h_Q = hQ
            self.h_S = hS
            self.h_Sigma = hSi

    def InitSequence(self, x0_hat: torch.Tensor, T: int):
        self.T = T
        self.m1x_post = x0_hat.to(self.device)
        self.m1x_post_prev = self.m1x_post
        self.m1x_prior_prev = self.m1x_post
        self.y_prev = self.h(self.m1x_post)

        self.m1x_prior = self.f(self.m1x_post)
        self.m1y = self.h(self.m1x_prior)

    # -------------------
    # Steps
    # -------------------
    def _expand_dim(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((1, self.batch_size, x.shape[-1]), device=self.device)
        out[0, :, :] = x
        return out

    def _rnn_step(self, rnn, inp, h):
        if self._is_lstm():
            out, (h_new, c_new) = rnn(inp, h)
            return out, (h_new, c_new)
        else:
            out, h_new = rnn(inp, h)
            return out, h_new

    def _kgain(self, y: torch.Tensor) -> torch.Tensor:
        # features (same as your KalmanNet)
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_prev, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_post, 2) - torch.squeeze(self.m1x_post_prev, 2)
        fw_update_diff = torch.squeeze(self.m1x_post, 2) - torch.squeeze(self.m1x_prior_prev, 2)

        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        obs_diff = self._expand_dim(obs_diff)
        obs_innov_diff = self._expand_dim(obs_innov_diff)
        fw_evol_diff = self._expand_dim(fw_evol_diff)
        fw_update_diff = self._expand_dim(fw_update_diff)

        out_FC5 = self.FC5(fw_update_diff)
        out_Q, self.h_Q = self._rnn_step(self.RNN_Q, out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), dim=2)
        out_Sigma, self.h_Sigma = self._rnn_step(self.RNN_Sigma, in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), dim=2))
        in_S = torch.cat((out_FC1, out_FC7), dim=2)
        out_S, self.h_S = self._rnn_step(self.RNN_S, in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), dim=2)
        K_vec = self.FC2(in_FC2)[0]  # [B, m*n]

        K = K_vec.view(self.batch_size, self.m, self.n)
        return K

    def step(self, y: torch.Tensor) -> torch.Tensor:
        self.m1x_prior = self.f(self.m1x_post)
        self.m1y = self.h(self.m1x_prior)

        K = self._kgain(y)
        self.KGain = K

        dy = y - self.m1y
        inov = torch.bmm(K, dy)

        self.m1x_post_prev = self.m1x_post
        self.m1x_post = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y

        return self.m1x_post

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)
        return self.step(y)
