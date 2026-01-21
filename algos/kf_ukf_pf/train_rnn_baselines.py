# algos/kf_ukf_pf/train_rnn_baselines.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajDataset(Dataset):
    """
    X: [N,T+1,m]
    Y: [N,T,n]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@dataclass
class RNNTrainCfg:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0


def _x0_from_mode(Xb: torch.Tensor, x0_mode: str) -> torch.Tensor | None:
    """
    Xb: [B, T+1, m]
    Returns x0_hat: [B, m] or None
    """
    B, _, m = Xb.shape
    device = Xb.device
    if x0_mode == "true":
        return Xb[:, 0, :]
    if x0_mode == "zeros":
        return torch.zeros((B, m), device=device, dtype=Xb.dtype)
    if x0_mode == "none":
        return None
    raise ValueError(f"Unknown x0_mode={x0_mode}. Use one of: 'true','zeros','none'.")


def train_state_estimator(
    model,
    X_train,
    Y_train,
    batch_size: int,
    cfg: RNNTrainCfg,
    device: torch.device,
    x0_mode: str = "zeros",
):
    ds = TrajDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.to(device)
    model.train()

    for ep in range(cfg.epochs):
        losses = []
        for Xb, Yb in dl:
            Xb = Xb.to(device)  # [B,T+1,m]
            Yb = Yb.to(device)  # [B,T,n]
            X_true = Xb[:, 1:, :]  # [B,T,m]

            x0_hat = _x0_from_mode(Xb, x0_mode=x0_mode)  # [B,m] or None

            opt.zero_grad()
            X_hat = model(Yb, x0_hat=x0_hat)  # [B,T,m]
            loss = torch.mean((X_hat - X_true) ** 2)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            losses.append(loss.item())

        print(f"[epoch {ep+1:03d}] train MSE: {np.mean(losses):.6e}")


@torch.no_grad()
def eval_state_estimator(
    model,
    X_test,
    Y_test,
    device: torch.device,
    x0_mode: str = "zeros",
) -> float:
    model.eval()
    model.to(device)

    mses_db = []
    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,m]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,n]

        # build x0_hat consistent with training
        if x0_mode == "true":
            x0_hat = X[0:1, :]                 # [1,m]
        elif x0_mode == "zeros":
            x0_hat = torch.zeros_like(X[0:1, :])  # [1,m]
        elif x0_mode == "none":
            x0_hat = None
        else:
            raise ValueError(f"Unknown x0_mode={x0_mode}. Use one of: 'true','zeros','none'.")

        X_true = X[1:, :].cpu().numpy()  # [T,m]
        X_hat = model(Y.unsqueeze(0), x0_hat=x0_hat).squeeze(0).cpu().numpy()  # [T,m]

        err = X_true - X_hat
        mse = np.mean(np.sum(err**2, axis=-1))
        mses_db.append(10.0 * np.log10(mse + 1e-12))

    return float(np.mean(mses_db))
