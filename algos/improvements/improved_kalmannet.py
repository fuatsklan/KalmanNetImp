# algos/improvements/improved_kalmannet.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImprovedKNetArgs:
    use_cuda: bool = False
    n_batch: int = 32
    in_mult: int = 5
    out_mult: int = 2

    # Improvements
    rnn_type: str = "gru"     # {"gru","lstm"}
    alpha_hybrid: float = 0.5 # fixed mixing weight for K = (1-a)K_kf + a*K_nn


class ImprovedKalmanNet(nn.Module):
    """
    Hybrid KalmanNet:
      - NN predicts K_nn
      - Analytic KF computes K_kf
      - Mixed gain: K = (1-alpha)*K_kf + alpha*K_nn
      - Optionally use GRU or LSTM in the gain network
    """
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args: ImprovedKNetArgs):
        self.device = torch.device("cuda") if (args.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.args = args

        # IMPORTANT: attach F/H matrices HERE (so eval works too)
        # LinearSysModelTorch has these tensors.
        self.F_mat = SysModel.F.to(self.device)  # [m,m]
        self.H_mat = SysModel.H.to(self.device)  # [n,m]

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self._init_priors(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)
        self._init_gain_net(args)

        self.to(self.device)

    # -------------------
    # System + priors
    # -------------------
    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.h = h
        self.m = m
        self.n = n

    def _init_priors(self, prior_Q, prior_Sigma, prior_S):
        self.prior_Q = prior_Q.to(self.device)           # (m,m)
        self.prior_Sigma = prior_Sigma.to(self.device)   # (m,m)
        self.prior_S = prior_S.to(self.device)           # (n,n)

        # Also keep explicit Q and R for internal KF covariance recursion
        self.Q = self.prior_Q
        self.R = self.prior_S

    # -------------------
    # Gain network
    # -------------------
    def _make_rnn(self, input_dim: int, hidden_dim: int):
        rnn_type = self.args.rnn_type.lower()
        if rnn_type == "gru":
            return nn.GRU(input_dim, hidden_dim)
        if rnn_type == "lstm":
            return nn.LSTM(input_dim, hidden_dim)
        raise ValueError(f"Unknown rnn_type={self.args.rnn_type}. Use 'gru' or 'lstm'.")

    def _init_gain_net(self, args: ImprovedKNetArgs):
        self.seq_len_input = 1
        self.batch_size = args.n_batch
        self.alpha = float(args.alpha_hybrid)

        # RNN blocks
        self.d_input_Q = self.m * args.in_mult
        self.d_hidden_Q = self.m ** 2
        self.RNN_Q = self._make_rnn(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.RNN_Sigma = self._make_rnn(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult
        self.d_hidden_S = self.n ** 2
        self.RNN_S = self._make_rnn(self.d_input_S, self.d_hidden_S).to(self.device)

        # FC heads (same style as your current KalmanNet)
        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, self.n ** 2), nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.n * self.m),
        ).to(self.device)

        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + (self.n * self.m), self.m ** 2),
            nn.ReLU()
        ).to(self.device)

        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + (self.m ** 2), self.d_hidden_Sigma),
            nn.ReLU()
        ).to(self.device)

        self.FC5 = nn.Sequential(nn.Linear(self.m, self.m * args.in_mult), nn.ReLU()).to(self.device)
        self.FC6 = nn.Sequential(nn.Linear(self.m, self.m * args.in_mult), nn.ReLU()).to(self.device)
        self.FC7 = nn.Sequential(nn.Linear(2 * self.n, 2 * self.n * args.in_mult), nn.ReLU()).to(self.device)

    def _is_lstm(self):
        return self.args.rnn_type.lower() == "lstm"

    # -------------------
    # Hidden state init
    # -------------------
    def init_hidden(self, batch_size: int):
        self.batch_size = batch_size

        # NN gain-net hidden states
        hQ = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()
        hS = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()
        hSi = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()

        if self._is_lstm():
            # LSTM needs (h,c)
            self.h_Q = (hQ, torch.zeros_like(hQ))
            self.h_S = (hS, torch.zeros_like(hS))
            self.h_Sigma = (hSi, torch.zeros_like(hSi))
        else:
            self.h_Q = hQ
            self.h_S = hS
            self.h_Sigma = hSi

        # KF covariance recursion state (per batch)
        self.P = self.prior_Sigma.unsqueeze(0).repeat(self.batch_size, 1, 1).contiguous()  # [B,m,m]
        self.I_m = torch.eye(self.m, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1).contiguous()

        # will be set each step
        self.KGain = None

    def InitSequence(self, x0_hat: torch.Tensor, T: int):
        """
        x0_hat: [B,m,1]
        """
        self.T = T
        self.m1x_post = x0_hat.to(self.device)
        self.m1x_post_prev = self.m1x_post
        self.m1x_prior_prev = self.m1x_post
        self.y_prev = self.h(self.m1x_post)

        # initialize predicted measurement for first step
        self.m1x_prior = self.f(self.m1x_post)
        self.m1y = self.h(self.m1x_prior)

    # -------------------
    # Analytic KF gain
    # -------------------
    def _kf_gain(self) -> torch.Tensor:
        Fm = self.F_mat  # [m,m]
        Hm = self.H_mat  # [n,m]

        P_pred = Fm.unsqueeze(0) @ self.P @ Fm.t().unsqueeze(0) + self.Q.unsqueeze(0)
        S = Hm.unsqueeze(0) @ P_pred @ Hm.t().unsqueeze(0) + self.R.unsqueeze(0)

        PHt = P_pred @ Hm.t().unsqueeze(0)  # [B,m,n]
        # solve S * X = (PHt^T) -> X = S^{-1} PHt^T
        K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)  # [B,m,n]

        self.P_pred = P_pred
        return K

    def _kf_cov_update(self, K: torch.Tensor):
        """
        Joseph form:
          P = (I-KH)P_pred(I-KH)^T + K R K^T
        """
        Hm = self.H_mat
        I = self.I_m
        KH = K @ Hm.unsqueeze(0)
        IKH = I - KH
        P = IKH @ self.P_pred @ IKH.transpose(1, 2) + K @ self.R.unsqueeze(0) @ K.transpose(1, 2)
        self.P = P

    # -------------------
    # NN gain step
    # -------------------
    def _expand_dim(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1], device=self.device)
        out[0, :, :] = x
        return out

    def _rnn_step(self, rnn, inp, h):
        if self._is_lstm():
            out, (h_new, c_new) = rnn(inp, h)
            return out, (h_new, c_new)
        else:
            out, h_new = rnn(inp, h)
            return out, h_new

    def _kgain_nn(self, y: torch.Tensor) -> torch.Tensor:
        # features
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_prev, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_post, 2) - torch.squeeze(self.m1x_post_prev, 2)
        fw_update_diff = torch.squeeze(self.m1x_post, 2) - torch.squeeze(self.m1x_prior_prev, 2)

        # normalize
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        # expand to [seq=1, B, dim]
        obs_diff = self._expand_dim(obs_diff)
        obs_innov_diff = self._expand_dim(obs_innov_diff)
        fw_evol_diff = self._expand_dim(fw_evol_diff)
        fw_update_diff = self._expand_dim(fw_update_diff)

        # forward
        out_FC5 = self.FC5(fw_update_diff)
        out_Q, self.h_Q = self._rnn_step(self.RNN_Q, out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), dim=2)
        out_Sigma, self.h_Sigma = self._rnn_step(self.RNN_Sigma, in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), dim=2))
        in_S = torch.cat((out_FC1, out_FC7), dim=2)
        out_S, self.h_S = self._rnn_step(self.RNN_S, in_S, self.h_S)

        # K output
        in_FC2 = torch.cat((out_Sigma, out_S), dim=2)
        out_FC2 = self.FC2(in_FC2)  # [1,B,m*n]

        # backward flow (ONLY for GRU; LSTM hidden state stays what LSTM produced)
        out_FC3 = self.FC3(torch.cat((out_S, out_FC2), dim=2))
        out_FC4 = self.FC4(torch.cat((out_Sigma, out_FC3), dim=2))
        if not self._is_lstm():
            self.h_Sigma = out_FC4

        K_vec = out_FC2[0]  # [B, m*n]
        K_nn = K_vec.view(self.batch_size, self.m, self.n)
        return K_nn

    # -------------------
    # One step
    # -------------------
    def step(self, y: torch.Tensor) -> torch.Tensor:
        # prior
        self.m1x_prior = self.f(self.m1x_post)
        self.m1y = self.h(self.m1x_prior)

        # gains
        K_nn = self._kgain_nn(y)
        K_kf = self._kf_gain()

        a = self.alpha
        K = (1.0 - a) * K_kf + a * K_nn
        self.KGain = K

        # state update
        dy = y - self.m1y
        inov = torch.bmm(K, dy)

        self.m1x_post_prev = self.m1x_post
        self.m1x_post = self.m1x_prior + inov
        self.m1x_prior_prev = self.m1x_prior
        self.y_prev = y

        # covariance update (Joseph form, using mixed K)
        self._kf_cov_update(K)

        return self.m1x_post

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)
        return self.step(y)
