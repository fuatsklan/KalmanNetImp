# algos/improvements/train_truncated_improved.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ImpTruncBPTTCfg:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    chunk_len: int = 100
    stride: int = 100
    lambda_K_smooth: float = 0.0   # <-- improvement


class ChunkedTrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, chunk_len: int, stride: int):
        self.X = X
        self.Y = Y
        self.chunk_len = chunk_len
        self.stride = stride

        N, Tlong, _ = Y.shape
        self.index = []
        for i in range(N):
            for t0 in range(0, Tlong - chunk_len + 1, stride):
                self.index.append((i, t0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t0 = self.index[k]
        Xc = self.X[i, t0 : t0 + self.chunk_len + 1, :]
        Yc = self.Y[i, t0 : t0 + self.chunk_len, :]
        return torch.tensor(Xc, dtype=torch.float32), torch.tensor(Yc, dtype=torch.float32)


def train_improved_kalmannet_truncated_bptt(
    model,
    X_train_long: np.ndarray,
    Y_train_long: np.ndarray,
    batch_size: int,
    cfg: ImpTruncBPTTCfg,
):
    device = model.device
    ds = ChunkedTrajectoryDataset(X_train_long, Y_train_long, chunk_len=cfg.chunk_len, stride=cfg.stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    skipped = 0
    total = 0

    for ep in range(cfg.epochs):
        losses = []
        skipped_ep = 0
        total_ep = 0

        for Xb, Yb in dl:
            total += 1
            total_ep += 1

            Xb = Xb.to(device)  # [B, L+1, m]
            Yb = Yb.to(device)  # [B, L, n]

            B = Xb.shape[0]
            T = Yb.shape[1]

            model.init_hidden(batch_size=B)
            x0_hat = Xb[:, 0, :].unsqueeze(-1)
            model.InitSequence(x0_hat, T=T)

            opt.zero_grad()

            xhats = []
            K_smooth = 0.0
            K_prev = None

            for t in range(T):
                yt = Yb[:, t, :].unsqueeze(-1)
                x_post = model(yt)
                xhats.append(x_post.squeeze(-1))

                if cfg.lambda_K_smooth > 0:
                    Kt = model.KGain
                    if K_prev is not None:
                        K_smooth = K_smooth + torch.mean((Kt - K_prev) ** 2)
                    K_prev = Kt

            X_hat = torch.stack(xhats, dim=1)   # [B,T,m]
            X_true = Xb[:, 1:, :]               # [B,T,m]

            loss_state = torch.mean((X_hat - X_true) ** 2)

            if cfg.lambda_K_smooth > 0 and T > 1:
                loss = loss_state + cfg.lambda_K_smooth * (K_smooth / (T - 1))
            else:
                loss = loss_state

            if not torch.isfinite(loss):
                skipped += 1
                skipped_ep += 1
                continue

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            losses.append(loss.item())

        ep_loss = float(np.mean(losses)) if len(losses) else float("nan")
        print(f"[ImpTruncBPTT epoch {ep+1:03d}] train loss: {ep_loss:.6e} | skipped {skipped_ep}/{total_ep}")

    if total > 0:
        print(f"[ImpTruncBPTT] total skipped batches: {skipped}/{total}")
