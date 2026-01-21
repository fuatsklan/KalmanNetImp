from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@dataclass
class ImprovedNLTrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_K_smooth: float = 0.0


def train_improved_kalmannet_nonlinear(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    batch_size: int,
    cfg: ImprovedNLTrainConfig,
) -> None:
    device = model.device
    ds = TrajectoryDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    for ep in range(cfg.epochs):
        losses = []
        for Xb, Yb in dl:
            Xb = Xb.to(device)  # [B,T+1,m]
            Yb = Yb.to(device)  # [B,T,n]
            B, T, _ = Yb.shape
            m = Xb.shape[-1]

            model.init_hidden(batch_size=B)

            x0_hat = Xb[:, 0, :].unsqueeze(-1)  # in nonlinear script you already use true x0
            model.InitSequence(x0_hat, T=T)

            optim.zero_grad()

            xhats = []
            K_smooth = 0.0
            K_prev = None

            for t in range(T):
                yt = Yb[:, t, :].unsqueeze(-1)
                x_post = model(yt)
                xhats.append(x_post.squeeze(-1))

                if cfg.lambda_K_smooth > 0.0:
                    Kt = model.KGain
                    if K_prev is not None:
                        K_smooth = K_smooth + torch.mean((Kt - K_prev) ** 2)
                    K_prev = Kt

            X_hat = torch.stack(xhats, dim=1)  # [B,T,m]
            X_true = Xb[:, 1:, :]

            loss_state = torch.mean((X_hat - X_true) ** 2)
            if cfg.lambda_K_smooth > 0.0 and T > 1:
                loss = loss_state + cfg.lambda_K_smooth * (K_smooth / (T - 1))
            else:
                loss = loss_state

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            losses.append(loss.item())

        print(f"[epoch {ep+1:03d}] train loss: {np.mean(losses):.6e}")
