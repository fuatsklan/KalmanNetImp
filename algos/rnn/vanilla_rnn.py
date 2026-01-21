# algos/rnn/vanilla_rnn.py
from __future__ import annotations
import torch
import torch.nn as nn


class VanillaRNNPos(nn.Module):
    """
    Baseline: map velocity observations y_t=[vx,vy] to position estimate [px,py]
    """
    def __init__(self, in_dim=2, hidden=64, out_dim=2, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, Y):  # Y: [B,T,2]
        out, _ = self.gru(Y)
        pos = self.fc(out)  # [B,T,2]
        return pos
