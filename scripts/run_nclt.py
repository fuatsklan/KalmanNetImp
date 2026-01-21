# scripts/run_nclt.py
from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.nclt_dataset import load_nclt_node_odometry, split_nclt_like_paper
from algos.kf_ukf_pf.nclt_kf import make_F, make_H, make_Q, make_stacked, KFLinear
from algos.rnn.vanilla_rnn import VanillaRNNPos
from algos.kalmannet.kalmannet_arch1 import KalmanNetArch1, KNetArch1Args
from algos.kalmannet.nclt_sysmodel import NCLTSysModelTorch


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_out_dir():
    out = Path(__file__).parent / "nclt_results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def mse_db_pos(gt_xy: np.ndarray, hat_xy: np.ndarray) -> float:
    err = gt_xy - hat_xy
    mse = np.mean(np.sum(err**2, axis=-1))
    return float(10 * np.log10(mse + 1e-12))


def integrate_velocity(v: np.ndarray, p0: np.ndarray, dt: float = 1.0):
    """Integrate velocity to position using fixed dt (paper uses 1Hz)."""
    p = np.zeros((v.shape[0], 2), dtype=float)
    cur = p0.astype(float).copy()
    for t in range(v.shape[0]):
        cur = cur + v[t] * dt
        p[t] = cur
    return p


def center_sequence_xy(X_seq: np.ndarray):
    Xc = X_seq.copy()
    p0 = Xc[0, [0, 2]].copy()
    Xc[:, 0] -= p0[0]
    Xc[:, 2] -= p0[1]
    return Xc, p0


def center_batch(Xb: np.ndarray):
    Xc = Xb.copy()
    p0 = Xc[:, 0, :][:, [0, 2]].copy()
    Xc[:, :, 0] -= p0[:, 0:1]
    Xc[:, :, 2] -= p0[:, 1:2]
    return Xc, p0


@torch.no_grad()
def eval_kalmannet_arch1(model: KalmanNetArch1, X_seq: np.ndarray, Y_seq: np.ndarray) -> np.ndarray:
    model.eval()
    device = model.device

    Xc, p0 = center_sequence_xy(X_seq)
    X = torch.tensor(Xc, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_seq, dtype=torch.float32, device=device)
    T = Y.shape[0]

    model.init_hidden(batch_size=1)
    x0_hat = X[0].view(1, 4, 1)
    model.InitSequence(x0_hat, T=T)

    xh = []
    for t in range(T):
        yt = Y[t].view(1, 2, 1)
        x_post = model(yt).squeeze(0).squeeze(-1)  # [4]
        xh.append(x_post)

    X_hat = torch.stack(xh, dim=0).cpu().numpy()  # [T,4]
    pos_hat_centered = np.stack([X_hat[:, 0], X_hat[:, 2]], axis=1)

    return pos_hat_centered + p0[None, :]


def train_vanilla_rnn(model: VanillaRNNPos, Ytr, Xtr, Yva, Xva, device, epochs=200, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr_c, _ = center_batch(Xtr)
    Xva_c, _ = center_batch(Xva)

    Xtr_pos = Xtr_c[:, 1:, :][:, :, [0, 2]]  # [B,T,2]
    Xva_pos = Xva_c[:, 1:, :][:, :, [0, 2]]

    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=device)
    Yva_t = torch.tensor(Yva, dtype=torch.float32, device=device)
    Xtr_t = torch.tensor(Xtr_pos, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_pos, dtype=torch.float32, device=device)

    best = 1e99
    best_state = None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Ytr_t)
        loss = torch.mean((out - Xtr_t) ** 2)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            outv = model(Yva_t)
            vloss = torch.mean((outv - Xva_t) ** 2).item()

        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 20 == 0:
            print(f"[VanillaRNN ep {ep+1:03d}] train={loss.item():.6e} val={vloss:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_knet_val_mse_db(model: KalmanNetArch1, Xva: np.ndarray, Yva: np.ndarray) -> float:
    mses = []
    for i in range(Yva.shape[0]):
        Xv_c, p0 = center_sequence_xy(Xva[i])
        # run model
        pos_hat = eval_kalmannet_arch1(model, Xva[i], Yva[i])  # absolute
        pos_true = Xva[i][1:, :][:, [0, 2]]
        mses.append(mse_db_pos(pos_true, pos_hat))
    return float(np.mean(mses))


def train_kalmannet_c1(
    model: KalmanNetArch1,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xva: np.ndarray,
    Yva: np.ndarray,
    device,
    epochs=300,
    lr=1e-5,
    weight_decay=1e-6,
    batch_size=8,
):
    """
    C1: Arch1 + {F2,F4} + fixed-length segments.
    Train on centered POSITION-only loss. Validate and keep best weights.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    N = Xtr.shape[0]
    T = Ytr.shape[1]

    best_val = 1e99
    best_state = None

    for ep in range(epochs):
        idx = np.random.permutation(N)
        losses = []
        model.train()

        for s in range(0, N, batch_size):
            bidx = idx[s:s + batch_size]
            Xb_np = Xtr[bidx]  # [B,T+1,4]
            Yb_np = Ytr[bidx]  # [B,T,2]

            Xb_c, _ = center_batch(Xb_np)

            Xb = torch.tensor(Xb_c, dtype=torch.float32, device=device)
            Yb = torch.tensor(Yb_np, dtype=torch.float32, device=device)
            B = Xb.shape[0]

            model.init_hidden(batch_size=B)
            x0 = Xb[:, 0, :].unsqueeze(-1)  # [B,4,1]
            model.InitSequence(x0, T=T)

            opt.zero_grad()
            xh = []
            for t in range(T):
                yt = Yb[:, t, :].unsqueeze(-1)         # [B,2,1]
                x_post = model(yt).squeeze(-1)         # [B,4]
                xh.append(x_post)

            X_hat = torch.stack(xh, dim=1)            # [B,T,4]
            X_true = Xb[:, 1:, :]                     # [B,T,4]

            pos_hat = X_hat[:, :, [0, 2]]             # [B,T,2]
            pos_true = X_true[:, :, [0, 2]]           # [B,T,2]
            loss = torch.mean((pos_hat - pos_true) ** 2)

            if not torch.isfinite(loss):
                # skip poisoned batch
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            losses.append(loss.item())

        # validation + checkpoint
        model.eval()
        val_mse_db = eval_knet_val_mse_db(model, Xva, Yva)
        if val_mse_db < best_val:
            best_val = val_mse_db
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0:
            tr = float(np.mean(losses)) if len(losses) else float("nan")
            print(f"[KalmanNet C1 ep {ep+1:03d}] train_mse={tr:.6e} val_mse_db={best_val:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def grid_search_kf(Xtr, Ytr, Xva, Yva, dt=1.0):
    """
    Grid search q2 and r2 on validation (position MSE).
    """
    F1 = make_F(dt)
    H1 = make_H()
    q2_grid = np.logspace(-6, 0, 13)
    r2_grid = np.logspace(-6, 0, 13)

    best = (1e99, None)
    for q2 in q2_grid:
        Q1 = make_Q(dt, q2=q2)
        for r2 in r2_grid:
            F, H, Q, R = make_stacked(F1, H1, Q1, r2=r2)
            kf = KFLinear(F=F, H=H, Q=Q, R=R)

            mses = []
            for i in range(Yva.shape[0]):
                Xv_c, _ = center_sequence_xy(Xva[i])
                X_hat = kf.run(Yva[i], x0=Xv_c[0])  # [T,4] centered
                pos_hat_c = np.stack([X_hat[:, 0], X_hat[:, 2]], axis=1)
                pos_true_c = Xv_c[1:, :][:, [0, 2]]
                mses.append(mse_db_pos(pos_true_c, pos_hat_c))

            score = float(np.mean(mses))
            if score < best[0]:
                best = (score, (q2, r2))

    return best


def main():
    out_dir = get_out_dir()
    tag = now_tag()
    root = project_root / "nclt_data"

    # ---- Load paper-like odom-only velocity measurements (1Hz) ----
    X_true, Y_meas, meta = load_nclt_node_odometry(
        root=root,
        session_dirname="2012-01-22",
        groundtruth_name="groundtruth.csv",
        odom_node_name="odometry_mu.csv",
        odom_dxdy_cols=(1, 2),
        target_dt_s=1.0,
        max_speed_mps=10.0,
        assume_body_frame=True,   # IMPORTANT
    )
    splits = split_nclt_like_paper(X_true, Y_meas)

    print("[NCLT meta]", json.dumps(meta, indent=2))
    print("[Split meta]", splits.meta)

    use_cuda = True
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    # test segment
    Xte = splits.X_test[0]
    Yte = splits.Y_test[0]
    pos_true_test = Xte[1:, :][:, [0, 2]]
    p0_abs = Xte[0, [0, 2]]

    # Integrated velocity baseline (correct units now)
    pos_int = integrate_velocity(Yte, p0_abs, dt=1.0)

    # KF grid search
    best_val_mse, (best_q2, best_r2) = grid_search_kf(splits.X_train, splits.Y_train, splits.X_val, splits.Y_val, dt=1.0)
    F1 = make_F(1.0)
    H1 = make_H()
    Q1 = make_Q(1.0, best_q2)
    F, H, Q, R = make_stacked(F1, H1, Q1, best_r2)
    kf = KFLinear(F=F, H=H, Q=Q, R=R)

    # KF evaluation (center then add back for plot)
    Xte_c, _ = center_sequence_xy(Xte)
    X_hat_kf_c = kf.run(Yte, x0=Xte_c[0])
    pos_kf = np.stack([X_hat_kf_c[:, 0], X_hat_kf_c[:, 2]], axis=1) + p0_abs[None, :]

    # Vanilla RNN (predict centered positions from velocities)
    rnn = VanillaRNNPos(in_dim=2, hidden=64, out_dim=2, n_layers=1)
    rnn = train_vanilla_rnn(
        rnn, splits.Y_train, splits.X_train, splits.Y_val, splits.X_val,
        device=device, epochs=200, lr=1e-3
    )
    rnn.eval()
    with torch.no_grad():
        pos_rnn_c = rnn(torch.tensor(Yte[None, :, :], dtype=torch.float32, device=device)).squeeze(0).cpu().numpy()
    pos_rnn = pos_rnn_c + p0_abs[None, :]

    # KalmanNet C1
    sys_torch = NCLTSysModelTorch.build(prior_q2=1e-3, prior_r2=1e-3, dt=1.0, device=device)
    kn = KalmanNetArch1(feature_set=["F2", "F4"])
    kn.NNBuild(
        sys_torch,
        KNetArch1Args(
            use_cuda=use_cuda,
            n_batch=8,
            hidden_mult=10,
            n_gru_layers=1,
            feat_norm_eps=1e-1,        # IMPORTANT for 1Hz NCLT
            kgain_tanh_scale=1.0,      # IMPORTANT for stability
            enable_nan_guards=False,
        ),
    )
    kn = train_kalmannet_c1(
        kn, splits.X_train, splits.Y_train, splits.X_val, splits.Y_val,
        device=device, epochs=300, lr=1e-5, weight_decay=1e-6, batch_size=8
    )
    pos_kn = eval_kalmannet_arch1(kn, Xte, Yte)

    # Metrics
    mse_int = mse_db_pos(pos_true_test, pos_int)
    mse_kf = mse_db_pos(pos_true_test, pos_kf)
    mse_rnn = mse_db_pos(pos_true_test, pos_rnn)
    mse_kn = mse_db_pos(pos_true_test, pos_kn)

    results = {
        "tag": tag,
        "meta": meta,
        "split": splits.meta,
        "kf_grid_best_val_mse_db": float(best_val_mse),
        "kf_best_q2": float(best_q2),
        "kf_best_r2": float(best_r2),
        "test_mse_db": {
            "integrated_velocity": float(mse_int),
            "KF": float(mse_kf),
            "vanilla_RNN": float(mse_rnn),
            "KalmanNet_C1": float(mse_kn),
        }
    }

    (out_dir / f"nclt_results_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["test_mse_db"], indent=2))

    # Plot (two plots: global + zoom)
    plt.figure()
    plt.plot(pos_true_test[:, 0], pos_true_test[:, 1], label="Ground Truth")
    plt.plot(pos_int[:, 0], pos_int[:, 1], label=f"Integrated Velocity ({mse_int:.2f} dB)")
    plt.plot(pos_kf[:, 0], pos_kf[:, 1], label=f"KF ({mse_kf:.2f} dB)")
    plt.plot(pos_rnn[:, 0], pos_rnn[:, 1], label=f"Vanilla RNN ({mse_rnn:.2f} dB)")
    plt.plot(pos_kn[:, 0], pos_kn[:, 1], label=f"KalmanNet C1 ({mse_kn:.2f} dB)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("NCLT 2012-01-22 (paper-like split)")
    plt.tight_layout()
    plt.savefig(out_dir / f"nclt_traj_{tag}.png", dpi=200)

    # Zoom around GT region to avoid one diverging method flattening the view
    xmin, xmax = np.percentile(pos_true_test[:, 0], [1, 99])
    ymin, ymax = np.percentile(pos_true_test[:, 1], [1, 99])
    pad_x = 0.2 * (xmax - xmin + 1e-6)
    pad_y = 0.2 * (ymax - ymin + 1e-6)

    plt.figure()
    plt.plot(pos_true_test[:, 0], pos_true_test[:, 1], label="Ground Truth")
    plt.plot(pos_int[:, 0], pos_int[:, 1], label="Integrated Velocity")
    plt.plot(pos_kf[:, 0], pos_kf[:, 1], label="KF")
    plt.plot(pos_rnn[:, 0], pos_rnn[:, 1], label="Vanilla RNN")
    plt.plot(pos_kn[:, 0], pos_kn[:, 1], label="KalmanNet C1")
    plt.xlim(xmin - pad_x, xmax + pad_x)
    plt.ylim(ymin - pad_y, ymax + pad_y)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("NCLT (zoomed to GT extent)")
    plt.tight_layout()
    plt.savefig(out_dir / f"nclt_traj_zoom_{tag}.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
