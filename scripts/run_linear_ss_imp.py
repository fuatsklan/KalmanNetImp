# scripts/run_linear_ss_imp.py
from __future__ import annotations

import sys
from pathlib import Path
import json
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch

from models.linear_ssm import make_F, make_H, LinearSSM, make_noise_from_nu
from algos.kf_ukf_pf.kalman_filter import KalmanFilter
from algos.kalmannet.sysmodel import LinearSysModelTorch
from algos.kalmannet.kalmannet_nn import KalmanNetNN, KNetArgs
from algos.kalmannet.train import train_kalmannet_linear, TrainConfig

# Improved KalmanNet
from algos.improvements.improved_kalmannet import ImprovedKalmanNet, ImprovedKNetArgs
from algos.improvements.train_improved import train_improved_kalmannet_linear, ImprovedTrainConfig


# -------------------------
# Results folder utilities
# -------------------------
def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "linear_ss_imp_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------
# Reproducibility
# -------------------------
def seed_all(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Core helpers
# -------------------------
def mse_db(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    err = X_true - X_hat
    mse = np.mean(np.sum(err**2, axis=-1))
    return 10.0 * np.log10(mse + 1e-12)


def _initial_x0_from_mode(X0: np.ndarray, m: int, x0_mode: str) -> np.ndarray:
    if x0_mode == "true":
        return np.asarray(X0, dtype=float).reshape(m,)
    if x0_mode == "zeros":
        return np.zeros((m,), dtype=float)
    raise ValueError(f"Unknown x0_mode={x0_mode}. Use 'true' or 'zeros'.")


def eval_kf(F, H, Q, R, X_test, Y_test, x0_mode: str = "zeros") -> float:
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    mses = []
    for i in range(Y_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        x_true = X[1:]
        x0_hat = _initial_x0_from_mode(X[0], m=F.shape[0], x0_mode=x0_mode)
        x_hat, _ = kf.run(Y, x0_hat=x0_hat)
        mses.append(mse_db(x_true, x_hat))
    return float(np.mean(mses))


@torch.no_grad()
def eval_kalmannet_baseline(model: KalmanNetNN, X_test, Y_test, x0_mode: str = "zeros") -> float:
    model.eval()
    device = model.device
    m = model.m

    mses = []
    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,m]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,n]
        T = Y.shape[0]

        model.init_hidden_KNet(batch_size=1)

        if x0_mode == "true":
            x0_hat = X[0].view(1, -1, 1)
        elif x0_mode == "zeros":
            x0_hat = torch.zeros((1, m, 1), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown x0_mode={x0_mode}")

        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, -1, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)  # [m]
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()
        X_true = X[1:].cpu().numpy()
        mses.append(mse_db(X_true, X_hat))

    return float(np.mean(mses))


@torch.no_grad()
def eval_kalmannet_improved(model: ImprovedKalmanNet, X_test, Y_test, x0_mode: str = "zeros") -> float:
    model.eval()
    device = model.device
    m = model.m

    mses = []
    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,m]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,n]
        T = Y.shape[0]

        model.init_hidden(batch_size=1)

        if x0_mode == "true":
            x0_hat = X[0].view(1, -1, 1)
        elif x0_mode == "zeros":
            x0_hat = torch.zeros((1, m, 1), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown x0_mode={x0_mode}")

        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, -1, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()
        X_true = X[1:].cpu().numpy()
        mses.append(mse_db(X_true, X_hat))

    return float(np.mean(mses))


def generate_dataset(ssm: LinearSSM, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _ in range(N):
        X, Y = ssm.sample(T=T)
        Xs.append(X)
        Ys.append(Y)
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)


def pick_device(force_cuda: bool = True) -> torch.device:
    if force_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rot2d(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s,  c]], dtype=float)


# -------------------------
# Builders
# -------------------------
def build_baseline_knet(sys_torch: LinearSysModelTorch, batch_size: int, use_cuda: bool) -> KalmanNetNN:
    knet = KalmanNetNN()
    knet.NNBuild(sys_torch, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
    return knet


def build_improved_knet(sys_torch: LinearSysModelTorch, batch_size: int, use_cuda: bool,
                        *, rnn_type: str = "lstm", alpha: float = 0.5) -> ImprovedKalmanNet:
    iknet = ImprovedKalmanNet()
    iknet.NNBuild(sys_torch, ImprovedKNetArgs(
        use_cuda=use_cuda,
        n_batch=batch_size,
        in_mult=5,
        out_mult=2,
        rnn_type=rnn_type,        # "gru" or "lstm"
        alpha_hybrid=alpha,
    ))
    return iknet


# -------------------------
# Option 1 (baseline vs improved)
# -------------------------
def run_option1_compare(out_dir: Path, *, nu_db: float = 0.0, r2: float = 0.5,
                        improved_rnn: str = "lstm", alpha: float = 0.5, lamK: float = 1e-3) -> dict:
    """
    Compare:
      KF vs Baseline KalmanNet vs ImprovedKalmanNet
    Train on T=20, test on T=20 and T=200.
    """
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    q2, r2 = make_noise_from_nu(r2=r2, nu_db=nu_db)

    T_train, T_same, T_long = 20, 20, 200
    N_train, N_test = 2000, 500
    batch_size, epochs = 32, 30

    F0 = make_F(m, form="controllable_canonical", seed=0).copy()
    H0 = make_H(n, m, form="inverse_canonical").copy()

    ssm = LinearSSM(F=F0, H=H0, q2=q2, r2=r2, seed=123)
    Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_train)
    Xte_same, Yte_same = generate_dataset(ssm, N=N_test, T=T_same)
    Xte_long, Yte_long = generate_dataset(ssm, N=N_test, T=T_long)

    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    # KF
    kf_same = eval_kf(F0, H0, ssm.Q, ssm.R, Xte_same, Yte_same, x0_mode=x0_mode_eval)
    kf_long = eval_kf(F0, H0, ssm.Q, ssm.R, Xte_long, Yte_long, x0_mode=x0_mode_eval)

    sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2, prior_r2=r2, device=device)

    # Baseline KNet
    knet = build_baseline_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda)
    train_kalmannet_linear(
        knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
        cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
    )
    kn_same = eval_kalmannet_baseline(knet, Xte_same, Yte_same, x0_mode=x0_mode_eval)
    kn_long = eval_kalmannet_baseline(knet, Xte_long, Yte_long, x0_mode=x0_mode_eval)

    # Improved KNet
    iknet = build_improved_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda, rnn_type=improved_rnn, alpha=alpha)
    train_improved_kalmannet_linear(
        iknet, sys_torch, Xtr, Ytr, batch_size=batch_size,
        cfg=ImprovedTrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0,
                                x0_mode=x0_mode_train, lambda_K_smooth=lamK)
    )
    ik_same = eval_kalmannet_improved(iknet, Xte_same, Yte_same, x0_mode=x0_mode_eval)
    ik_long = eval_kalmannet_improved(iknet, Xte_long, Yte_long, x0_mode=x0_mode_eval)

    results = {
        "tag": tag,
        "mode": "option1_compare",
        "config": {
            "m": m, "n": n, "nu_db": nu_db, "q2": float(q2), "r2": float(r2),
            "T_train": T_train, "T_test_same": T_same, "T_test_long": T_long,
            "N_train": N_train, "N_test": N_test,
            "epochs": epochs, "batch_size": batch_size,
            "x0_mode_train": x0_mode_train, "x0_mode_eval": x0_mode_eval,
            "device": str(device),
            "improved": {"rnn_type": improved_rnn, "alpha": float(alpha), "lambda_K_smooth": float(lamK)},
        },
        "mse_db": {
            "KF_T20": float(kf_same),
            "KF_T200": float(kf_long),
            "KNet_T20": float(kn_same),
            "KNet_T200": float(kn_long),
            "ImpKNet_T20": float(ik_same),
            "ImpKNet_T200": float(ik_long),
        }
    }

    print("\n=== Option1 Compare ===")
    print(f"KF        | T20 {kf_same:8.2f} | T200 {kf_long:8.2f}")
    print(f"KNet      | T20 {kn_same:8.2f} | T200 {kn_long:8.2f}")
    print(f"ImpKNet   | T20 {ik_same:8.2f} | T200 {ik_long:8.2f} | rnn={improved_rnn} alpha={alpha} lamK={lamK}")

    # Plot
    labels = ["KF", "KNet", "ImpKNet"]
    t20 = [kf_same, kn_same, ik_same]
    t200 = [kf_long, kn_long, ik_long]
    x = np.arange(len(labels))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, t20, w, label="Test T=20")
    plt.bar(x + w/2, t200, w, label="Test T=200")
    plt.xticks(x, labels)
    plt.ylabel("MSE [dB]")
    plt.title("Option1: KF vs KNet vs ImprovedKNet")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / f"imp_option1_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    save_json(results, out_dir / f"imp_option1_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 2A Grid (baseline vs improved)
# -------------------------
def run_option2_grid_compare(out_dir: Path, *, improved_rnn: str = "lstm", alpha: float = 0.5, lamK: float = 1e-3) -> dict:
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    nu_db = 0.0
    r2 = 1e-2
    q2, r2 = make_noise_from_nu(r2=r2, nu_db=nu_db)

    dims = [(2, 2), (5, 5)]
    T_list = [50, 100, 200]

    N_train = 1000
    N_test = 200
    batch_size = 32
    epochs = 20  # shorten

    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    rows = []
    start = time.time()

    for (m, n) in dims:
        F0 = make_F(m, form="controllable_canonical", seed=0).copy()
        H0 = make_H(n, m, form="inverse_canonical").copy()

        for T in T_list:
            ssm = LinearSSM(F=F0, H=H0, q2=q2, r2=r2, seed=123)
            Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T)
            Xte, Yte = generate_dataset(ssm, N=N_test, T=T)

            kf_mse = eval_kf(F0, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)

            sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2, prior_r2=r2, device=device)

            # baseline
            knet = build_baseline_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda)
            train_kalmannet_linear(
                knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
                cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
            )
            kn_mse = eval_kalmannet_baseline(knet, Xte, Yte, x0_mode=x0_mode_eval)

            # improved
            iknet = build_improved_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda, rnn_type=improved_rnn, alpha=alpha)
            train_improved_kalmannet_linear(
                iknet, sys_torch, Xtr, Ytr, batch_size=batch_size,
                cfg=ImprovedTrainConfig(epochs=epochs, lr=1e-3, grad_clip=1.0,
                                        x0_mode=x0_mode_train, lambda_K_smooth=lamK)
            )
            ik_mse = eval_kalmannet_improved(iknet, Xte, Yte, x0_mode=x0_mode_eval)

            print(f"[grid] {m}x{n} T={T} | KF={kf_mse:.2f} | KNet={kn_mse:.2f} | ImpKNet={ik_mse:.2f}")
            rows.append({
                "m": m, "n": n, "T": T,
                "KF_mse_db": float(kf_mse),
                "KNet_mse_db": float(kn_mse),
                "ImpKNet_mse_db": float(ik_mse),
            })

    elapsed = time.time() - start

    # Plot
    plt.figure()
    for (m, n) in dims:
        Ts = [r["T"] for r in rows if r["m"] == m and r["n"] == n]
        kf = [r["KF_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        kn = [r["KNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        ik = [r["ImpKNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        plt.plot(Ts, kf, marker="o", label=f"KF {m}x{n}")
        plt.plot(Ts, kn, marker="o", linestyle="--", label=f"KNet {m}x{n}")
        plt.plot(Ts, ik, marker="o", linestyle=":", label=f"ImpKNet {m}x{n}")

    plt.xlabel("T")
    plt.ylabel("MSE [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.title("Option2A Grid: KF vs KNet vs ImprovedKNet")
    plt.tight_layout()

    fig_path = out_dir / f"imp_option2_grid_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option2_grid_compare",
        "config": {
            "nu_db": nu_db, "q2": float(q2), "r2": float(r2),
            "dims": dims, "T_list": T_list,
            "N_train": N_train, "N_test": N_test,
            "epochs": epochs, "batch_size": batch_size,
            "device": str(device),
            "improved": {"rnn_type": improved_rnn, "alpha": float(alpha), "lambda_K_smooth": float(lamK)},
            "x0_mode_train": x0_mode_train, "x0_mode_eval": x0_mode_eval,
            "elapsed_sec": float(elapsed),
        },
        "rows": rows,
    }
    save_json(results, out_dir / f"imp_option2_grid_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 2B Noise sweep (baseline vs improved)
# -------------------------
def run_option2_noise_compare(out_dir: Path, *, improved_rnn: str = "lstm", alpha: float = 0.5, lamK: float = 1e-3) -> dict:
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    T_train = 20
    T_test = 200
    nu_db = 0.0

    N_train = 1000
    N_test = 200
    batch_size = 32
    epochs = 20  # shorten

    F0 = make_F(m, form="controllable_canonical", seed=0).copy()
    H0 = make_H(n, m, form="inverse_canonical").copy()

    r2_list = np.logspace(-4, -1, 7)
    inv_r2_db = 10.0 * np.log10(1.0 / r2_list)

    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    kf_curve, kn_curve, ik_curve = [], [], []

    start = time.time()
    for r2 in r2_list:
        q2, rr2 = make_noise_from_nu(r2=float(r2), nu_db=nu_db)
        ssm = LinearSSM(F=F0, H=H0, q2=q2, r2=rr2, seed=123)

        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_train)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_test)

        kf_mse = eval_kf(F0, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)

        sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2, prior_r2=rr2, device=device)

        # baseline
        knet = build_baseline_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda)
        train_kalmannet_linear(
            knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, grad_clip=1.0, x0_mode=x0_mode_train)
        )
        kn_mse = eval_kalmannet_baseline(knet, Xte, Yte, x0_mode=x0_mode_eval)

        # improved
        iknet = build_improved_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda, rnn_type=improved_rnn, alpha=alpha)
        train_improved_kalmannet_linear(
            iknet, sys_torch, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedTrainConfig(epochs=epochs, lr=1e-3, grad_clip=1.0, x0_mode=x0_mode_train, lambda_K_smooth=lamK)
        )
        ik_mse = eval_kalmannet_improved(iknet, Xte, Yte, x0_mode=x0_mode_eval)

        print(f"[noise] r2={rr2:.2e} | KF={kf_mse:.2f} | KNet={kn_mse:.2f} | ImpKNet={ik_mse:.2f}")

        kf_curve.append(kf_mse)
        kn_curve.append(kn_mse)
        ik_curve.append(ik_mse)

    elapsed = time.time() - start

    plt.figure()
    plt.plot(inv_r2_db, kf_curve, marker="o", label="KF")
    plt.plot(inv_r2_db, kn_curve, marker="o", label="KNet")
    plt.plot(inv_r2_db, ik_curve, marker="o", label="ImpKNet")
    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Option2B Noise Sweep Compare (nu={nu_db} dB)")
    plt.tight_layout()

    fig_path = out_dir / f"imp_option2_noise_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option2_noise_compare",
        "config": {
            "m": m, "n": n, "nu_db": nu_db,
            "T_train": T_train, "T_test": T_test,
            "N_train": N_train, "N_test": N_test,
            "epochs": epochs, "batch_size": batch_size,
            "r2_list": [float(x) for x in r2_list],
            "device": str(device),
            "improved": {"rnn_type": improved_rnn, "alpha": float(alpha), "lambda_K_smooth": float(lamK)},
            "x0_mode_train": x0_mode_train, "x0_mode_eval": x0_mode_eval,
            "elapsed_sec": float(elapsed),
        },
        "inv_r2_db": [float(x) for x in inv_r2_db],
        "kf_curve_db": [float(x) for x in kf_curve],
        "knet_curve_db": [float(x) for x in kn_curve],
        "imp_curve_db": [float(x) for x in ik_curve],
    }
    save_json(results, out_dir / f"imp_option2_noise_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 3 Partial info (baseline vs improved)
# -------------------------
def run_option3_partial_compare(out_dir: Path, *, improved_rnn: str = "lstm", alpha: float = 0.5, lamK: float = 1e-3) -> dict:
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    # A) F mismatch
    nu_db_A = 0.0
    r2_A = 1e-2
    q2_A, r2_A = make_noise_from_nu(r2=r2_A, nu_db=nu_db_A)

    T_A = 20
    N_train = 2000
    N_test = 500
    batch_size = 32
    epochs = 30

    F0 = make_F(m, form="controllable_canonical", seed=0).copy()
    H0 = make_H(n, m, form="inverse_canonical").copy()

    alphas = [10.0, 20.0]
    rows_F = []

    for a_deg in alphas:
        R = rot2d(a_deg)
        F_alpha = R @ F0

        ssm = LinearSSM(F=F_alpha, H=H0, q2=q2_A, r2=r2_A, seed=123)
        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_A)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_A)

        kf_m = eval_kf(F0, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)
        kf_or = eval_kf(F_alpha, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)

        sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2_A, prior_r2=r2_A, device=device)

        # baseline KNet
        knet = build_baseline_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda)
        train_kalmannet_linear(
            knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, grad_clip=1.0, x0_mode=x0_mode_train)
        )
        kn_m = eval_kalmannet_baseline(knet, Xte, Yte, x0_mode=x0_mode_eval)

        # improved KNet
        iknet = build_improved_knet(sys_torch, batch_size=batch_size, use_cuda=use_cuda, rnn_type=improved_rnn, alpha=alpha)
        train_improved_kalmannet_linear(
            iknet, sys_torch, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedTrainConfig(epochs=epochs, lr=1e-3, grad_clip=1.0, x0_mode=x0_mode_train, lambda_K_smooth=lamK)
        )
        ik_m = eval_kalmannet_improved(iknet, Xte, Yte, x0_mode=x0_mode_eval)

        print(f"[partial F] rot={a_deg:.1f} | KFmis={kf_m:.2f} | KNet={kn_m:.2f} | ImpKNet={ik_m:.2f} | KFor={kf_or:.2f}")

        rows_F.append({
            "alpha_deg": float(a_deg),
            "KF_mismatched_db": float(kf_m),
            "KF_oracle_db": float(kf_or),
            "KNet_mismatched_db": float(kn_m),
            "ImpKNet_mismatched_db": float(ik_m),
        })

    # plot F mismatch compare
    plt.figure()
    xs = [r["alpha_deg"] for r in rows_F]
    plt.plot(xs, [r["KF_mismatched_db"] for r in rows_F], marker="o", label="KF (mismatch F0)")
    plt.plot(xs, [r["KNet_mismatched_db"] for r in rows_F], marker="o", label="KNet (mismatch F0)")
    plt.plot(xs, [r["ImpKNet_mismatched_db"] for r in rows_F], marker="o", label="ImpKNet (mismatch F0)")
    plt.plot(xs, [r["KF_oracle_db"] for r in rows_F], marker="o", label="KF oracle (true F)")
    plt.xlabel("Rotation alpha [deg]")
    plt.ylabel("MSE [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.title("Option3A F-mismatch: baseline vs improved")
    plt.tight_layout()

    fig_F = out_dir / f"imp_option3_F_{tag}.png"
    plt.savefig(fig_F, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option3_partial_compare",
        "config": {
            "x0_mode_train": x0_mode_train, "x0_mode_eval": x0_mode_eval,
            "device": str(device),
            "improved": {"rnn_type": improved_rnn, "alpha": float(alpha), "lambda_K_smooth": float(lamK)},
        },
        "F_mismatch": {"rows": rows_F, "plot_path": str(fig_F)},
    }
    save_json(results, out_dir / f"imp_option3_{tag}.json")
    results["plot_path"] = str(fig_F)
    return results


# -------------------------
# Combined “4-in-1” compare figure
# -------------------------
def make_combined_compare(out_dir: Path, tag: str, o1: dict, o2g: dict, o2n: dict, o3: dict) -> Path:
    fig = plt.figure(figsize=(14, 10))

    # (1) option1 bars
    ax1 = fig.add_subplot(2, 2, 1)
    mse = o1["mse_db"]
    labels = ["KF", "KNet", "ImpKNet"]
    t20 = [mse["KF_T20"], mse["KNet_T20"], mse["ImpKNet_T20"]]
    t200 = [mse["KF_T200"], mse["KNet_T200"], mse["ImpKNet_T200"]]
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, t20, w, label="T=20")
    ax1.bar(x + w/2, t200, w, label="T=200")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("MSE [dB]")
    ax1.set_title("Option1 Compare")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    # (2) grid
    ax2 = fig.add_subplot(2, 2, 2)
    rows = o2g["rows"]
    dims = o2g["config"]["dims"]
    for (m, n) in dims:
        Ts = [r["T"] for r in rows if r["m"] == m and r["n"] == n]
        kf = [r["KF_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        kn = [r["KNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        ik = [r["ImpKNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        ax2.plot(Ts, kf, marker="o", label=f"KF {m}x{n}")
        ax2.plot(Ts, kn, marker="o", linestyle="--", label=f"KNet {m}x{n}")
        ax2.plot(Ts, ik, marker="o", linestyle=":", label=f"ImpKNet {m}x{n}")
    ax2.set_xlabel("T")
    ax2.set_ylabel("MSE [dB]")
    ax2.set_title("Option2A Grid Compare")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # (3) noise sweep
    ax3 = fig.add_subplot(2, 2, 3)
    inv_r2_db = o2n["inv_r2_db"]
    ax3.plot(inv_r2_db, o2n["kf_curve_db"], marker="o", label="KF")
    ax3.plot(inv_r2_db, o2n["knet_curve_db"], marker="o", label="KNet")
    ax3.plot(inv_r2_db, o2n["imp_curve_db"], marker="o", label="ImpKNet")
    ax3.set_xlabel("10log10(1/r^2) [dB]")
    ax3.set_ylabel("MSE [dB]")
    ax3.set_title("Option2B Noise Compare")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # (4) partial F mismatch
    ax4 = fig.add_subplot(2, 2, 4)
    rowsF = o3["F_mismatch"]["rows"]
    xs = [r["alpha_deg"] for r in rowsF]
    ax4.plot(xs, [r["KF_mismatched_db"] for r in rowsF], marker="o", label="KF mismatch")
    ax4.plot(xs, [r["KNet_mismatched_db"] for r in rowsF], marker="o", label="KNet")
    ax4.plot(xs, [r["ImpKNet_mismatched_db"] for r in rowsF], marker="o", label="ImpKNet")
    ax4.plot(xs, [r["KF_oracle_db"] for r in rowsF], marker="o", label="KF oracle")
    ax4.set_xlabel("Rotation alpha [deg]")
    ax4.set_ylabel("MSE [dB]")
    ax4.set_title("Option3 F-mismatch Compare")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"imp_combined_{tag}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    seed_all(0)
    out_dir = get_results_dir()

    master_tag = now_tag()
    print(f"\n=== Running IMPROVEMENTS suite | tag={master_tag} ===\n")

    # Improvement config
    improved_rnn = "lstm"   # "gru" or "lstm"
    alpha = 0.5             # fixed hybrid mix
    lamK = 1e-3             # gain smoothness penalty (0.0 disables)

    o1 = run_option1_compare(out_dir, nu_db=0.0, r2=0.5, improved_rnn=improved_rnn, alpha=alpha, lamK=lamK)
    o2g = run_option2_grid_compare(out_dir, improved_rnn=improved_rnn, alpha=alpha, lamK=lamK)
    o2n = run_option2_noise_compare(out_dir, improved_rnn=improved_rnn, alpha=alpha, lamK=lamK)
    o3 = run_option3_partial_compare(out_dir, improved_rnn=improved_rnn, alpha=alpha, lamK=lamK)

    combined_path = make_combined_compare(out_dir, master_tag, o1, o2g, o2n, o3)

    summary = {
        "tag": master_tag,
        "mode": "improvements_all",
        "improved": {"rnn_type": improved_rnn, "alpha": float(alpha), "lambda_K_smooth": float(lamK)},
        "plots": {
            "combined": str(combined_path),
            "option1": o1.get("plot_path"),
            "option2_grid": o2g.get("plot_path"),
            "option2_noise": o2n.get("plot_path"),
            "option3_F": o3.get("plot_path"),
        },
        "results_files": {
            "o1_json": str(out_dir / f"imp_option1_{o1['tag']}.json"),
            "o2g_json": str(out_dir / f"imp_option2_grid_{o2g['tag']}.json"),
            "o2n_json": str(out_dir / f"imp_option2_noise_{o2n['tag']}.json"),
            "o3_json": str(out_dir / f"imp_option3_{o3['tag']}.json"),
        }
    }
    save_json(summary, out_dir / f"imp_all_summary_{master_tag}.json")

    print(f"\nSaved combined plot: {combined_path}")
    print(f"Saved summary JSON:  {out_dir / f'imp_all_summary_{master_tag}.json'}")


if __name__ == "__main__":
    main()
