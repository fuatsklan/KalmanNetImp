# scripts/run_linear_ss.py
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


# -------------------------
# Results folder utilities
# -------------------------
def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "linear_ss_results"
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
        x_true = X[1:]  # [T,m]
        x0_hat = _initial_x0_from_mode(X[0], m=F.shape[0], x0_mode=x0_mode)
        x_hat, _ = kf.run(Y, x0_hat=x0_hat)
        mses.append(mse_db(x_true, x_hat))
    return float(np.mean(mses))


@torch.no_grad()
def eval_kalmannet(model: KalmanNetNN, X_test, Y_test, x0_mode: str = "zeros") -> float:
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
            raise ValueError(f"Unknown x0_mode={x0_mode}. Use 'true' or 'zeros'.")

        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, -1, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)  # [m]
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()  # [T,m]
        X_true = X[1:].cpu().numpy()                     # [T,m]
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
# Option 1: Neural model selection (paper B.2)
# -------------------------
def run_neural_model_selection(out_dir: Path, *, nu_db: float = 0.0, r2: float = 1e-2) -> dict:
    """
    Vanilla RNN vs MB-RNN vs KalmanNet vs KF.
    Train on T=20, test on T=20 and T=200.
    Saves option1 plot + json.
    """
    from algos.kf_ukf_pf.rnn_baselines import VanillaRNNStateEstimator, MBRNNIncrementEstimator
    from algos.kf_ukf_pf.train_rnn_baselines import train_state_estimator, eval_state_estimator, RNNTrainCfg

    tag = now_tag()

    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    q2, r2 = make_noise_from_nu(r2=r2, nu_db=nu_db)

    T_train = 20
    T_test_same = 20
    T_test_long = 200

    N_train = 2000
    N_test = 500
    batch_size = 32
    epochs = 30

    F0 = make_F(m, form="controllable_canonical", seed=0).copy()
    H0 = make_H(n, m, form="inverse_canonical").copy()

    ssm = LinearSSM(F=F0, H=H0, q2=q2, r2=r2, seed=123)
    Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_train)
    Xte_same, Yte_same = generate_dataset(ssm, N=N_test, T=T_test_same)
    Xte_long, Yte_long = generate_dataset(ssm, N=N_test, T=T_test_long)

    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    # KF
    kf_same = eval_kf(F0, H0, ssm.Q, ssm.R, Xte_same, Yte_same, x0_mode=x0_mode_eval)
    kf_long = eval_kf(F0, H0, ssm.Q, ssm.R, Xte_long, Yte_long, x0_mode=x0_mode_eval)

    # Vanilla
    vanilla = VanillaRNNStateEstimator(n=n, m=m, hidden=128, n_layers=1)
    train_state_estimator(
        vanilla, Xtr, Ytr, batch_size=batch_size,
        cfg=RNNTrainCfg(epochs=epochs, lr=1e-3),
        device=device,
        x0_mode=x0_mode_train,
    )
    van_same = eval_state_estimator(vanilla, Xte_same, Yte_same, device=device, x0_mode=x0_mode_eval)
    van_long = eval_state_estimator(vanilla, Xte_long, Yte_long, device=device, x0_mode=x0_mode_eval)

    # MB-RNN
    F_torch = torch.tensor(np.ascontiguousarray(F0), dtype=torch.float32, device=device)
    mbrnn = MBRNNIncrementEstimator(F_mat=F_torch, n=n, m=m, hidden=128, n_layers=1)
    train_state_estimator(
        mbrnn, Xtr, Ytr, batch_size=batch_size,
        cfg=RNNTrainCfg(epochs=epochs, lr=1e-3),
        device=device,
        x0_mode=x0_mode_train,
    )
    mbr_same = eval_state_estimator(mbrnn, Xte_same, Yte_same, device=device, x0_mode=x0_mode_eval)
    mbr_long = eval_state_estimator(mbrnn, Xte_long, Yte_long, device=device, x0_mode=x0_mode_eval)

    # KalmanNet
    sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2, prior_r2=r2, device=device)
    knet = KalmanNetNN()
    knet.NNBuild(sys_torch, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))

    train_kalmannet_linear(
        knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
        cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
    )

    kn_same = eval_kalmannet(knet, Xte_same, Yte_same, x0_mode=x0_mode_eval)
    kn_long = eval_kalmannet(knet, Xte_long, Yte_long, x0_mode=x0_mode_eval)

    results = {
        "tag": tag,
        "mode": "option1_neural_model_selection",
        "x0_mode_train": x0_mode_train,
        "x0_mode_eval": x0_mode_eval,
        "config": {
            "m": m, "n": n,
            "nu_db": nu_db,
            "q2": float(q2), "r2": float(r2),
            "T_train": T_train,
            "T_test_same": T_test_same,
            "T_test_long": T_test_long,
            "N_train": N_train,
            "N_test": N_test,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
        },
        "mse_db": {
            "KF_T20": float(kf_same),
            "KF_T200": float(kf_long),
            "VanillaRNN_T20": float(van_same),
            "VanillaRNN_T200": float(van_long),
            "MBRNN_T20": float(mbr_same),
            "MBRNN_T200": float(mbr_long),
            "KalmanNet_T20": float(kn_same),
            "KalmanNet_T200": float(kn_long),
        }
    }

    print("\n=== Option1: Neural Model Selection (train T=20) ===")
    print(f"init: train={x0_mode_train} eval={x0_mode_eval}")
    print(f"KF (MMSE)     | test T=20:  {kf_same:8.2f} dB | test T=200: {kf_long:8.2f} dB")
    print(f"Vanilla RNN   | test T=20:  {van_same:8.2f} dB | test T=200: {van_long:8.2f} dB")
    print(f"MB-RNN        | test T=20:  {mbr_same:8.2f} dB | test T=200: {mbr_long:8.2f} dB")
    print(f"KalmanNet     | test T=20:  {kn_same:8.2f} dB | test T=200: {kn_long:8.2f} dB")

    labels = ["KF", "VanillaRNN", "MBRNN", "KalmanNet"]
    t20 = [kf_same, van_same, mbr_same, kn_same]
    t200 = [kf_long, van_long, mbr_long, kn_long]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, t20, width, label="Test T=20")
    plt.bar(x + width/2, t200, width, label="Test T=200")
    plt.xticks(x, labels)
    plt.ylabel("MSE [dB]")
    plt.title("Option1: Neural Model Selection (train T=20)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / f"option1_model_selection_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    save_json(results, out_dir / f"option1_results_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 2A: Paper-like Full Information grid (B.1)
# -------------------------
def run_full_information_grid(out_dir: Path) -> dict:
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
    epochs = 30

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
            knet = KalmanNetNN()
            knet.NNBuild(sys_torch, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))

            train_kalmannet_linear(
                knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
                cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
            )
            kn_mse = eval_kalmannet(knet, Xte, Yte, x0_mode=x0_mode_eval)

            print(f"[full-info-grid] m={m} n={n} T={T} | KF={kf_mse:.2f} dB | KNet={kn_mse:.2f} dB")
            rows.append({"m": m, "n": n, "T": T, "KF_mse_db": float(kf_mse), "KNet_mse_db": float(kn_mse)})

    elapsed = time.time() - start

    plt.figure()
    for (m, n) in dims:
        Ts = [r["T"] for r in rows if r["m"] == m and r["n"] == n]
        kf = [r["KF_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        kn = [r["KNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        plt.plot(Ts, kf, marker="o", label=f"KF {m}x{n}")
        plt.plot(Ts, kn, marker="o", linestyle="--", label=f"KNet {m}x{n}")

    plt.xlabel("T")
    plt.ylabel("MSE [dB]")
    plt.grid(True)
    plt.legend()
    plt.title("Option2A: Full Information Grid (nu=0 dB)")
    plt.tight_layout()

    fig_path = out_dir / f"option2_fullinfo_grid_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option2_fullinfo_grid",
        "x0_mode_train": x0_mode_train,
        "x0_mode_eval": x0_mode_eval,
        "config": {
            "nu_db": nu_db,
            "q2": float(q2), "r2": float(r2),
            "dims": dims,
            "T_list": T_list,
            "N_train": N_train, "N_test": N_test,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
            "elapsed_sec": float(elapsed),
        },
        "rows": rows,
    }
    save_json(results, out_dir / f"option2_grid_results_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 2B: Noise sweep
# -------------------------
def run_full_information_sweep(out_dir: Path) -> dict:
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    T_train = 20
    T_test = 200
    nu_db = 0.0

    N_train = 2000
    N_test = 500

    F0 = make_F(m, form="controllable_canonical", seed=0).copy()
    H0 = make_H(n, m, form="inverse_canonical").copy()

    r2_list = np.logspace(-4, -1, 7)
    inv_r2_db = 10.0 * np.log10(1.0 / r2_list)

    kf_curve, kn_curve = [], []

    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    start = time.time()
    for r2 in r2_list:
        q2, r2 = make_noise_from_nu(r2=float(r2), nu_db=nu_db)
        ssm = LinearSSM(F=F0, H=H0, q2=q2, r2=r2, seed=123)

        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_train)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_test)

        sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2, prior_r2=r2, device=device)
        args = KNetArgs(use_cuda=use_cuda, n_batch=32, in_mult_KNet=5, out_mult_KNet=2)
        knet = KalmanNetNN()
        knet.NNBuild(sys_torch, args)

        train_cfg = TrainConfig(epochs=30, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
        train_kalmannet_linear(knet, sys_torch, Xtr, Ytr, batch_size=args.n_batch, cfg=train_cfg)

        kf_mse = eval_kf(F0, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)
        kn_mse = eval_kalmannet(knet, Xte, Yte, x0_mode=x0_mode_eval)

        print(f"[noise-sweep] r2={r2:.2e} q2={q2:.2e} | KF={kf_mse:.2f} dB | KNet={kn_mse:.2f} dB")
        kf_curve.append(kf_mse)
        kn_curve.append(kn_mse)

    elapsed = time.time() - start

    plt.figure()
    plt.plot(inv_r2_db, kf_curve, marker="o", label="KF (full info)")
    plt.plot(inv_r2_db, kn_curve, marker="o", label="KalmanNet (learned gain)")
    plt.xlabel("10log10(1/r^2)  [dB]")
    plt.ylabel("MSE [dB]")
    plt.grid(True)
    plt.legend()
    plt.title(f"Option2B: Full-info Noise Sweep (m={m}, n={n}, nu={nu_db} dB)")
    plt.tight_layout()

    fig_path = out_dir / f"option2_noise_sweep_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option2_noise_sweep",
        "x0_mode_train": x0_mode_train,
        "x0_mode_eval": x0_mode_eval,
        "config": {
            "m": m, "n": n,
            "nu_db": nu_db,
            "T_train": T_train, "T_test": T_test,
            "N_train": N_train, "N_test": N_test,
            "r2_list": [float(x) for x in r2_list],
            "device": str(device),
            "elapsed_sec": float(elapsed),
        },
        "inv_r2_db": [float(x) for x in inv_r2_db],
        "kf_curve_db": [float(x) for x in kf_curve],
        "knet_curve_db": [float(x) for x in kn_curve],
    }
    save_json(results, out_dir / f"option2_noise_results_{tag}.json")
    results["plot_path"] = str(fig_path)
    return results


# -------------------------
# Option 3: Partial Information (paper B.3)
# -------------------------
def run_partial_information(out_dir: Path) -> dict:
    tag = now_tag()
    x0_mode_train = "zeros"
    x0_mode_eval = "zeros"

    m, n = 2, 2
    device = pick_device(force_cuda=True)
    use_cuda = (device.type == "cuda")

    # --- A) F mismatch: T=20, nu=0 dB, alpha in {10,20}
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

    for alpha in alphas:
        R = rot2d(alpha)
        F_alpha = R @ F0

        ssm = LinearSSM(F=F_alpha, H=H0, q2=q2_A, r2=r2_A, seed=123)
        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T_A)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_A)

        kf_mse = eval_kf(F0, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)        # mismatched
        kf_true = eval_kf(F_alpha, H0, ssm.Q, ssm.R, Xte, Yte, x0_mode=x0_mode_eval)   # oracle

        sys_torch = LinearSysModelTorch.from_numpy(F=F0, H=H0, m=m, n=n, prior_q2=q2_A, prior_r2=r2_A, device=device)
        knet = KalmanNetNN()
        knet.NNBuild(sys_torch, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            knet, sys_torch, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
        )
        kn_mse = eval_kalmannet(knet, Xte, Yte, x0_mode=x0_mode_eval)

        print(f"[partial F] alpha={alpha:>5.1f} | KF(F0)={kf_mse:.2f} | KNet(F0)={kn_mse:.2f} | KF(trueF)={kf_true:.2f}")

        rows_F.append({
            "alpha_deg": float(alpha),
            "KF_mismatched_db": float(kf_mse),
            "KNet_mismatched_db": float(kn_mse),
            "KF_true_db": float(kf_true),
        })

    # Plot F mismatch
    plt.figure()
    xs = [r["alpha_deg"] for r in rows_F]
    kf_m = [r["KF_mismatched_db"] for r in rows_F]
    kn_m = [r["KNet_mismatched_db"] for r in rows_F]
    kf_t = [r["KF_true_db"] for r in rows_F]
    plt.plot(xs, kf_m, marker="o", label="KF (mismatch F0)")
    plt.plot(xs, kn_m, marker="o", label="KalmanNet (mismatch F0)")
    plt.plot(xs, kf_t, marker="o", label="KF (oracle true F)")
    plt.xlabel("Rotation alpha [deg]")
    plt.ylabel("MSE [dB]")
    plt.grid(True)
    plt.legend()
    plt.title("Option3A: State-evolution mismatch (F)")
    plt.tight_layout()
    fig_F = out_dir / f"option3_F_mismatch_{tag}.png"
    plt.savefig(fig_F, dpi=200)
    plt.close()

    # --- B) H mismatch: T=100, nu=-20 dB, alpha=10, filters use H=I
    nu_db_B = -20.0
    r2_B = 1e-2
    q2_B, r2_B = make_noise_from_nu(r2=r2_B, nu_db=nu_db_B)

    T_B = 100
    alpha_H = 10.0
    R = rot2d(alpha_H)

    H_alpha = R @ H0
    H_used = np.eye(2, dtype=float)

    ssmB = LinearSSM(F=F0, H=H_alpha, q2=q2_B, r2=r2_B, seed=321)
    XtrB, YtrB = generate_dataset(ssmB, N=N_train, T=T_B)
    XteB, YteB = generate_dataset(ssmB, N=N_test, T=T_B)

    kfB = eval_kf(F0, H_used, ssmB.Q, ssmB.R, XteB, YteB, x0_mode=x0_mode_eval)       # mismatched
    kfB_true = eval_kf(F0, H_alpha, ssmB.Q, ssmB.R, XteB, YteB, x0_mode=x0_mode_eval)  # oracle

    sysB = LinearSysModelTorch.from_numpy(F=F0, H=H_used, m=m, n=n, prior_q2=q2_B, prior_r2=r2_B, device=device)
    knB = KalmanNetNN()
    knB.NNBuild(sysB, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
    train_kalmannet_linear(
        knB, sysB, XtrB, YtrB, batch_size=batch_size,
        cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, x0_mode=x0_mode_train)
    )
    knB_mse = eval_kalmannet(knB, XteB, YteB, x0_mode=x0_mode_eval)

    print(f"[partial H] alpha={alpha_H:.1f} | KF(H=I)={kfB:.2f} | KNet(H=I)={knB_mse:.2f} | KF(trueH)={kfB_true:.2f}")

    # Plot H mismatch (single alpha bar)
    plt.figure()
    labels = ["KF (H=I)", "KalmanNet (H=I)", "KF oracle (true H)"]
    vals = [kfB, knB_mse, kfB_true]
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("MSE [dB]")
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Option3B: Observation mismatch (H)")
    plt.tight_layout()
    fig_H = out_dir / f"option3_H_mismatch_{tag}.png"
    plt.savefig(fig_H, dpi=200)
    plt.close()

    results = {
        "tag": tag,
        "mode": "option3_partial_information",
        "x0_mode_train": x0_mode_train,
        "x0_mode_eval": x0_mode_eval,
        "F_mismatch": {
            "nu_db": nu_db_A,
            "T": T_A,
            "alphas_deg": alphas,
            "rows": rows_F,
            "plot_path": str(fig_F),
        },
        "H_mismatch": {
            "nu_db": nu_db_B,
            "T": T_B,
            "alpha_deg": alpha_H,
            "KF_mismatched_db": float(kfB),
            "KNet_mismatched_db": float(knB_mse),
            "KF_true_db": float(kfB_true),
            "plot_path": str(fig_H),
        }
    }

    save_json(results, out_dir / f"option3_partial_results_{tag}.json")
    # for combined plot selection (use F plot as representative)
    results["plot_path"] = str(fig_F)
    results["plot_path_H"] = str(fig_H)
    return results


# -------------------------
# Combined "4-in-1" figure
# -------------------------
def make_combined_figure(out_dir: Path, tag: str, opt1: dict, opt2_grid: dict, opt2_sweep: dict, opt3: dict) -> Path:
    """
    Creates a single figure with 4 subplots:
      (1) option1 bars
      (2) option2 grid curves
      (3) option2 noise sweep curves
      (4) option3 partial info (F mismatch curves + H mismatch bar in same axes)
    """
    fig = plt.figure(figsize=(14, 10))

    # ---- (1) option1 bars
    ax1 = fig.add_subplot(2, 2, 1)
    mse = opt1["mse_db"]
    labels = ["KF", "VanillaRNN", "MBRNN", "KalmanNet"]
    t20 = [mse["KF_T20"], mse["VanillaRNN_T20"], mse["MBRNN_T20"], mse["KalmanNet_T20"]]
    t200 = [mse["KF_T200"], mse["VanillaRNN_T200"], mse["MBRNN_T200"], mse["KalmanNet_T200"]]
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, t20, width, label="Test T=20")
    ax1.bar(x + width/2, t200, width, label="Test T=200")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("MSE [dB]")
    ax1.set_title("Option1: Neural Model Selection")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    # ---- (2) option2 grid
    ax2 = fig.add_subplot(2, 2, 2)
    rows = opt2_grid["rows"]
    dims = opt2_grid["config"]["dims"]
    for (m, n) in dims:
        Ts = [r["T"] for r in rows if r["m"] == m and r["n"] == n]
        kf = [r["KF_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        kn = [r["KNet_mse_db"] for r in rows if r["m"] == m and r["n"] == n]
        ax2.plot(Ts, kf, marker="o", label=f"KF {m}x{n}")
        ax2.plot(Ts, kn, marker="o", linestyle="--", label=f"KNet {m}x{n}")
    ax2.set_xlabel("T")
    ax2.set_ylabel("MSE [dB]")
    ax2.set_title("Option2A: Full-info Grid")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # ---- (3) option2 noise sweep
    ax3 = fig.add_subplot(2, 2, 3)
    inv_r2_db = opt2_sweep["inv_r2_db"]
    kf_curve = opt2_sweep["kf_curve_db"]
    kn_curve = opt2_sweep["knet_curve_db"]
    ax3.plot(inv_r2_db, kf_curve, marker="o", label="KF")
    ax3.plot(inv_r2_db, kn_curve, marker="o", label="KalmanNet")
    ax3.set_xlabel("10log10(1/r^2) [dB]")
    ax3.set_ylabel("MSE [dB]")
    ax3.set_title("Option2B: Noise Sweep")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # ---- (4) option3 partial info
    ax4 = fig.add_subplot(2, 2, 4)
    rows_F = opt3["F_mismatch"]["rows"]
    xs = [r["alpha_deg"] for r in rows_F]
    ax4.plot(xs, [r["KF_mismatched_db"] for r in rows_F], marker="o", label="KF (mismatch F0)")
    ax4.plot(xs, [r["KNet_mismatched_db"] for r in rows_F], marker="o", label="KNet (mismatch F0)")
    ax4.plot(xs, [r["KF_true_db"] for r in rows_F], marker="o", label="KF oracle (true F)")

    # add H-mismatch single point as a separate bar-like annotation
    kfH = opt3["H_mismatch"]["KF_mismatched_db"]
    knH = opt3["H_mismatch"]["KNet_mismatched_db"]
    kfH_or = opt3["H_mismatch"]["KF_true_db"]
    # place as small grouped bars at x = max(alpha)+10 for visualization
    xbar = max(xs) + 10.0
    bw = 1.8
    ax4.bar([xbar - bw, xbar, xbar + bw], [kfH, knH, kfH_or], width=1.2, alpha=0.6)
    ax4.set_xticks(xs + [xbar])
    ax4.set_xticklabels([str(int(a)) for a in xs] + ["H"])
    ax4.set_xlabel("alpha [deg] (F mismatch), H=obs mismatch summary")
    ax4.set_ylabel("MSE [dB]")
    ax4.set_title("Option3: Partial Information")
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"combined_all_{tag}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    seed_all(0)
    out_dir = get_results_dir()

    master_tag = now_tag()
    print(f"\n=== Running ALL experiments | tag={master_tag} ===\n")

    # Run everything (each function still saves its own plots+json with its own tag)
    # Option 1: Full-information neural model selection.
    # Increase noise to make long-horizon (T=200) evaluation more challenging.
    # - r2 controls measurement noise (R = r2 I)
    # - nu_db controls process/measurement ratio via q2 = (10^(nu_db/10)) * r2
    opt1 = run_neural_model_selection(out_dir, nu_db=0.0, r2=1e-1) #1e-1
    opt2_grid = run_full_information_grid(out_dir)
    opt2_sweep = run_full_information_sweep(out_dir)
    opt3 = run_partial_information(out_dir)

    # Combined 4-in-1 plot
    combined_path = make_combined_figure(out_dir, master_tag, opt1, opt2_grid, opt2_sweep, opt3)

    summary = {
        "tag": master_tag,
        "mode": "all",
        "plots": {
            "combined": str(combined_path),
            "option1": opt1.get("plot_path"),
            "option2_grid": opt2_grid.get("plot_path"),
            "option2_noise": opt2_sweep.get("plot_path"),
            "option3_F": opt3["F_mismatch"].get("plot_path"),
            "option3_H": opt3["H_mismatch"].get("plot_path"),
        },
        "results_files": {
            "option1_json": str(out_dir / f"option1_results_{opt1['tag']}.json"),
            "option2_grid_json": str(out_dir / f"option2_grid_results_{opt2_grid['tag']}.json"),
            "option2_noise_json": str(out_dir / f"option2_noise_results_{opt2_sweep['tag']}.json"),
            "option3_json": str(out_dir / f"option3_partial_results_{opt3['tag']}.json"),
        }
    }
    save_json(summary, out_dir / f"all_summary_{master_tag}.json")
    print(f"\nSaved combined plot: {combined_path}")
    print(f"Saved summary JSON:  {out_dir / f'all_summary_{master_tag}.json'}")


if __name__ == "__main__":
    main()
