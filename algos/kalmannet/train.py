# algos/kalmannet/train.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """
    Y: [N, T, n]
    X: [N, T+1, m] (includes x0)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # NEW: how we initialize x0_hat during training
    # "zeros" => no leakage, consistent with unknown initial state
    # "true"  => known initial state experiment
    x0_mode: str = "zeros"   # {"zeros","true"}


def train_kalmannet_linear(
    model,
    sysmodel,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    batch_size: int,
    cfg: TrainConfig,
) -> None:
    device = model.device
    ds = TrajectoryDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    for ep in range(cfg.epochs):
        losses = []
        for Xb, Yb in dl:
            Xb = Xb.to(device)  # [B, T+1, m]
            Yb = Yb.to(device)  # [B, T, n]

            B, T, _ = Yb.shape
            m = Xb.shape[-1]

            model.init_hidden_KNet(batch_size=B)

            if cfg.x0_mode == "true":
                x0_hat = Xb[:, 0, :].unsqueeze(-1)  # [B,m,1]
            elif cfg.x0_mode == "zeros":
                x0_hat = torch.zeros((B, m, 1), dtype=Xb.dtype, device=device)
            else:
                raise ValueError(f"Unknown x0_mode={cfg.x0_mode}")

            model.InitSequence(x0_hat, T=T)

            optim.zero_grad()

            xhats = []
            for t in range(T):
                yt = Yb[:, t, :].unsqueeze(-1)   # [B,n,1]
                x_post = model(yt)               # [B,m,1]
                xhats.append(x_post.squeeze(-1)) # [B,m]

            X_hat = torch.stack(xhats, dim=1)     # [B,T,m]
            X_true = Xb[:, 1:, :]                 # [B,T,m]

            loss = torch.mean((X_hat - X_true) ** 2)

            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            losses.append(loss.item())

        print(f"[epoch {ep+1:03d}] train MSE: {np.mean(losses):.6e}")
