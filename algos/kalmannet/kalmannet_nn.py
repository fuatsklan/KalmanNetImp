# algos/kalmannet/kalmannet_nn.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KNetArgs:
    use_cuda: bool = False
    n_batch: int = 32
    in_mult_KNet: int = 5
    out_mult_KNet: int = 2


class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args: KNetArgs):
        # IMPORTANT: respect SysModel.device if provided
        # If SysModel was created on CPU, don't force CUDA.
        self.device = getattr(SysModel, "device", None)
        if self.device is None:
            self.device = torch.device("cuda") if (args.use_cuda and torch.cuda.is_available()) else torch.device("cpu")

        # also reflect final device back into args
        args.use_cuda = (self.device.type == "cuda")
        self.args = args

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

        self.to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.h = h
        self.m = m
        self.n = n

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args: KNetArgs):
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # FC 1: Sigma -> n^2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, self.n ** 2),
            nn.ReLU()
        ).to(self.device)

        # FC 2: [Sigma, S] -> K (m*n)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.n * self.m),
        ).to(self.device)

        # FC 3: [S, K] -> m^2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + (self.n * self.m), self.m ** 2),
            nn.ReLU()
        ).to(self.device)

        # FC 4: [Sigma, m^2] -> m^2 (updated Sigma hidden)
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + (self.m ** 2), self.d_hidden_Sigma),
            nn.ReLU()
        ).to(self.device)

        # FC 5/6/7 are feature encoders
        self.FC5 = nn.Sequential(
            nn.Linear(self.m, self.m * args.in_mult_KNet),
            nn.ReLU()
        ).to(self.device)

        self.FC6 = nn.Sequential(
            nn.Linear(self.m, self.m * args.in_mult_KNet),
            nn.ReLU()
        ).to(self.device)

        self.FC7 = nn.Sequential(
            nn.Linear(2 * self.n, 2 * self.n * args.in_mult_KNet),
            nn.ReLU()
        ).to(self.device)

    def init_hidden_KNet(self, batch_size: int):
        self.batch_size = batch_size  # allow dynamic batch sizes
        weight = next(self.parameters()).data

        # h_S
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()

        # h_Sigma
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()

        # h_Q
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).contiguous()

    def InitSequence(self, M1_0: torch.Tensor, T: int):
        """
        M1_0: [B, m, 1]
        """
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    def step_prior(self):
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            out = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1], device=self.device)
            out[0, :, :] = x
            return out

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # Forward flow
        out_FC5 = self.FC5(fw_update_diff)                   # -> Q input
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)      # -> track Q

        out_FC6 = self.FC6(fw_evol_diff)                     # -> Sigma input
        in_Sigma = torch.cat((out_Q, out_FC6), dim=2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)                        # Sigma -> n^2

        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), dim=2))  # obs features
        in_S = torch.cat((out_FC1, out_FC7), dim=2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)         # track S

        # K output
        in_FC2 = torch.cat((out_Sigma, out_S), dim=2)
        out_FC2 = self.FC2(in_FC2)                           # -> K vector (m*n)

        # Backward flow (update Sigma hidden)
        out_FC3 = self.FC3(torch.cat((out_S, out_FC2), dim=2))
        out_FC4 = self.FC4(torch.cat((out_Sigma, out_FC3), dim=2))
        self.h_Sigma = out_FC4

        return out_FC2

    def step_KGain_est(self, y):
        # y: [B, n, 1]
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        # normalize (as in their repo)
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        KG_vec = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG_vec, (self.batch_size, self.m, self.n))

    def KNet_step(self, y):
        self.step_prior()
        self.step_KGain_est(y)

        dy = y - self.m1y  # innovation
        inov = torch.bmm(self.KGain, dy)

        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + inov
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        return self.m1x_posterior

    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)
