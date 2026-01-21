# algos/kalmannet/train_truncated.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class TruncBPTTCfg:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    chunk_len: int = 100
    stride: int = 100   # non-overlap like paper-style segmentation


class ChunkedTrajectoryDataset(Dataset):
    """
    Takes long trajectories and yields shuffled chunks:
      X: [N, T_long+1, m]
      Y: [N, T_long, n]
    Yields:
      X_chunk: [chunk_len+1, m] (includes x0 for chunk)
      Y_chunk: [chunk_len, n]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, chunk_len: int, stride: int):
        self.X = X
        self.Y = Y
        self.chunk_len = chunk_len
        self.stride = stride

        N, Tlong, n = Y.shape
        self.index = []
        # chunk start indexes over time for each trajectory
        for i in range(N):
            for t0 in range(0, Tlong - chunk_len + 1, stride):
                self.index.append((i, t0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t0 = self.index[k]
        # X has Tlong+1, chunk needs chunk_len+1 states: [t0 .. t0+chunk_len]
        Xc = self.X[i, t0 : t0 + self.chunk_len + 1, :]
        Yc = self.Y[i, t0 : t0 + self.chunk_len, :]
        return torch.tensor(Xc, dtype=torch.float32), torch.tensor(Yc, dtype=torch.float32)


def train_kalmannet_truncated_bptt(
    model,
    X_train_long: np.ndarray,
    Y_train_long: np.ndarray,
    batch_size: int,
    cfg: TruncBPTTCfg,
):
    device = model.device
    ds = ChunkedTrajectoryDataset(X_train_long, Y_train_long, chunk_len=cfg.chunk_len, stride=cfg.stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    for ep in range(cfg.epochs):
        losses = []
        for Xb, Yb in dl:
            # Xb: [B, chunk_len+1, m], Yb: [B, chunk_len, n]
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            B = Xb.shape[0]
            T = Yb.shape[1]

            model.init_hidden(batch_size=B)  # for Arch1 (GRU hidden)
            x0_hat = Xb[:, 0, :].unsqueeze(-1)  # [B,m,1]
            model.InitSequence(x0_hat, T=T)

            opt.zero_grad()

            xhats = []
            for t in range(T):
                yt = Yb[:, t, :].unsqueeze(-1)  # [B,n,1]
                x_post = model(yt)              # [B,m,1]
                xhats.append(x_post.squeeze(-1))

            X_hat = torch.stack(xhats, dim=1)      # [B,T,m]
            X_true = Xb[:, 1:, :]                  # [B,T,m]
            loss = torch.mean((X_hat - X_true) ** 2)

            # If something goes numerically unstable, skip this batch instead of poisoning training with NaNs.
            if not torch.isfinite(loss):
                continue

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            losses.append(loss.item())

        ep_mse = float(np.mean(losses)) if len(losses) else float("nan")
        print(f"[TruncBPTT epoch {ep+1:03d}] train MSE: {ep_mse:.6e}")
