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

from models.nonlinear_toy import NonlinearToySSM, make_noise_from_nu
from algos.kf_ukf_pf.ekf_nonlinear_toy import EKFNonlinearToy
from algos.kf_ukf_pf.ukf_nonlinear_toy import UKFNonlinearToy
from algos.kf_ukf_pf.particle_filter_nonlinear_toy import ParticleFilterNonlinearToy
from algos.kalmannet.kalmannet_nn import KalmanNetNN, KNetArgs
from algos.kalmannet.train import train_kalmannet_linear, TrainConfig
from algos.kalmannet.nonlinear_sysmodel import NonlinearToySysModelTorch

from algos.improvements.improved_kalmannet_nonlinear import ImprovedNonlinearKalmanNet, ImprovedNLKNetArgs
from algos.improvements.train_improved_nonlinear import train_improved_kalmannet_nonlinear, ImprovedNLTrainConfig


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "nonlinear_imp_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def mse_db(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    err = X_true - X_hat
    mse = np.mean(np.sum(err**2, axis=-1))
    return 10.0 * np.log10(mse + 1e-12)


def generate_dataset(ssm: NonlinearToySSM, N: int, T: int):
    Xs, Ys = [], []
    for _ in range(N):
        X, Y = ssm.sample(T=T)
        Xs.append(X)
        Ys.append(Y)
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)


def eval_mb_filter(filter_obj, X_test, Y_test) -> float:
    mses = []
    for i in range(Y_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        x_true = X[1:]
        out = filter_obj.run(Y, x0_hat=X[0])
        x_hat = out[0] if isinstance(out, (tuple, list)) else out
        mses.append(mse_db(x_true, np.asarray(x_hat, dtype=float)))
    return float(np.mean(mses))


@torch.no_grad()
def eval_kalmannet(model, X_test, Y_test) -> float:
    model.eval()
    device = model.device
    mses = []
    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,m]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,n]
        T = Y.shape[0]

        # both KNet and ImprovedNonlinearKalmanNet implement init_hidden + InitSequence
        if hasattr(model, "init_hidden_KNet"):
            model.init_hidden_KNet(batch_size=1)
        else:
            model.init_hidden(batch_size=1)

        x0_hat = X[0].view(1, -1, 1)
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


def main():
    out_dir = get_results_dir()
    tag = now_tag()

    # -------------------------------
    # Nonlinear toy experiment
    # -------------------------------
    T = 100
    nu_db = -20.0

    params_true = dict(alpha=0.9, beta=1.1, phi=0.1 * np.pi, delta=0.01, a=1.0, b=1.0, c=0.0)
    params_partial = dict(alpha=1.0, beta=1.0, phi=0.0, delta=0.0, a=1.0, b=1.0, c=0.0)

    r2_list = np.logspace(-4, -1, 7)
    inv_r2_db = 10.0 * np.log10(1.0 / r2_list)

    N_train = 3000
    N_test = 500
    batch_size = 32
    epochs = 30

    use_cuda = True
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    curves = {
        "full":    {"EKF": [], "UKF": [], "PF": [], "KalmanNet": [], "ImpKalmanNet": []},
        "partial": {"EKF": [], "UKF": [], "PF": [], "KalmanNet": [], "ImpKalmanNet": []},
    }

    for r2 in r2_list:
        q2, r2 = make_noise_from_nu(r2=float(r2), nu_db=nu_db)

        ssm = NonlinearToySSM(params_true=params_true, q2=q2, r2=r2, seed=123)
        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T)

        # ---- FULL ----
        ekf_full = EKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        ukf_full = UKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        pf_full = ParticleFilterNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_full_mse = eval_mb_filter(ekf_full, Xte, Yte)
        ukf_full_mse = eval_mb_filter(ukf_full, Xte, Yte)
        pf_full_mse = eval_mb_filter(pf_full, Xte, Yte)

        sys_full = NonlinearToySysModelTorch.from_numpy_params(
            params_design=params_true, prior_q2=q2, prior_r2=r2, device=device
        )

        kn_full = KalmanNetNN()
        kn_full.NNBuild(sys_full, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_full, sys_full, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_full_mse = eval_kalmannet(kn_full, Xte, Yte)

        imp_full = ImprovedNonlinearKalmanNet()
        imp_full.NNBuild(sys_full, ImprovedNLKNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult=5, out_mult=2, rnn_type="lstm"))
        train_improved_kalmannet_nonlinear(
            imp_full, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedNLTrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, lambda_K_smooth=1e-3)
        )
        imp_full_mse = eval_kalmannet(imp_full, Xte, Yte)

        # ---- PARTIAL ----
        ekf_part = EKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        ukf_part = UKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        pf_part = ParticleFilterNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_part_mse = eval_mb_filter(ekf_part, Xte, Yte)
        ukf_part_mse = eval_mb_filter(ukf_part, Xte, Yte)
        pf_part_mse = eval_mb_filter(pf_part, Xte, Yte)

        sys_part = NonlinearToySysModelTorch.from_numpy_params(
            params_design=params_partial, prior_q2=q2, prior_r2=r2, device=device
        )

        kn_part = KalmanNetNN()
        kn_part.NNBuild(sys_part, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_part, sys_part, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_part_mse = eval_kalmannet(kn_part, Xte, Yte)

        imp_part = ImprovedNonlinearKalmanNet()
        imp_part.NNBuild(sys_part, ImprovedNLKNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult=5, out_mult=2, rnn_type="lstm"))
        train_improved_kalmannet_nonlinear(
            imp_part, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedNLTrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0, lambda_K_smooth=1e-3)
        )
        imp_part_mse = eval_kalmannet(imp_part, Xte, Yte)

        # store
        curves["full"]["EKF"].append(ekf_full_mse)
        curves["full"]["UKF"].append(ukf_full_mse)
        curves["full"]["PF"].append(pf_full_mse)
        curves["full"]["KalmanNet"].append(kn_full_mse)
        curves["full"]["ImpKalmanNet"].append(imp_full_mse)

        curves["partial"]["EKF"].append(ekf_part_mse)
        curves["partial"]["UKF"].append(ukf_part_mse)
        curves["partial"]["PF"].append(pf_part_mse)
        curves["partial"]["KalmanNet"].append(kn_part_mse)
        curves["partial"]["ImpKalmanNet"].append(imp_part_mse)

        print(
            f"r2={r2:.2e} | FULL: EKF={ekf_full_mse:.2f} UKF={ukf_full_mse:.2f} PF={pf_full_mse:.2f} "
            f"KNet={kn_full_mse:.2f} ImpKNet={imp_full_mse:.2f} | "
            f"PART: EKF={ekf_part_mse:.2f} UKF={ukf_part_mse:.2f} PF={pf_part_mse:.2f} "
            f"KNet={kn_part_mse:.2f} ImpKNet={imp_part_mse:.2f}"
        )

    # plot
    plt.figure()
    plt.plot(inv_r2_db, curves["full"]["EKF"], marker="o", label="EKF (full)")
    plt.plot(inv_r2_db, curves["full"]["UKF"], marker="o", label="UKF (full)")
    plt.plot(inv_r2_db, curves["full"]["PF"], marker="o", label="PF (full)")
    plt.plot(inv_r2_db, curves["full"]["KalmanNet"], marker="o", label="KalmanNet (full)")
    plt.plot(inv_r2_db, curves["full"]["ImpKalmanNet"], marker="o", label="ImpKalmanNet (full)")

    plt.plot(inv_r2_db, curves["partial"]["EKF"], marker="o", linestyle="--", label="EKF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["UKF"], marker="o", linestyle="--", label="UKF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["PF"], marker="o", linestyle="--", label="PF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["KalmanNet"], marker="o", linestyle="--", label="KalmanNet (partial)")
    plt.plot(inv_r2_db, curves["partial"]["ImpKalmanNet"], marker="o", linestyle="--", label="ImpKalmanNet (partial)")

    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB]")
    plt.grid(True)
    plt.legend()
    plt.title("Nonlinear Toy SSM: EKF/UKF/PF vs KalmanNet vs Improved (full & partial)")
    plt.tight_layout()

    fig_path = out_dir / f"nonlinear_toy_imp_curves_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()

    results = {
        "tag": tag,
        "T": T,
        "nu_db": nu_db,
        "params_true": params_true,
        "params_partial": params_partial,
        "r2_list": [float(x) for x in r2_list],
        "inv_r2_db": [float(x) for x in inv_r2_db],
        "curves": curves,
        "N_train": N_train,
        "N_test": N_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": str(device),
        "improved": {
            "rnn_type": "lstm",
            "lambda_K_smooth": 1e-3
        }
    }
    save_json(results, out_dir / f"nonlinear_toy_imp_results_{tag}.json")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

"""
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

from models.nonlinear_toy import NonlinearToySSM, make_noise_from_nu
from algos.kf_ukf_pf.ekf_nonlinear_toy import EKFNonlinearToy
from algos.kf_ukf_pf.ukf_nonlinear_toy import UKFNonlinearToy
from algos.kf_ukf_pf.particle_filter_nonlinear_toy import ParticleFilterNonlinearToy
from algos.kalmannet.kalmannet_nn import KalmanNetNN, KNetArgs
from algos.kalmannet.train import train_kalmannet_linear, TrainConfig
from algos.kalmannet.nonlinear_sysmodel import NonlinearToySysModelTorch

from algos.improvements.improved_kalmannet_nonlinear import ImprovedNonlinearKalmanNet, ImprovedNLKNetArgs
from algos.improvements.train_improved_nonlinear import train_improved_kalmannet_nonlinear, ImprovedNLTrainConfig


# -------------------------
# IO utils
# -------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "nonlinear_imp_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# -------------------------
# Metrics
# -------------------------
def mse_db(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    err = X_true - X_hat
    mse = np.mean(np.sum(err**2, axis=-1))
    return 10.0 * np.log10(mse + 1e-12)


def summarize_mu_sigma(samples_db: list[float]) -> tuple[float, float]:
    """Return μ̂ and σ̂ in dB."""
    arr = np.asarray(samples_db, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))
    return mu, sigma


# -------------------------
# Data generation
# -------------------------
def generate_dataset(ssm: NonlinearToySSM, N: int, T: int):
    Xs, Ys = [], []
    for _ in range(N):
        X, Y = ssm.sample(T=T)
        Xs.append(X)
        Ys.append(Y)
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)


# -------------------------
# Evaluation: return per-trajectory samples
# -------------------------
def eval_mb_filter_samples(filter_obj, X_test, Y_test) -> list[float]:
    samples = []
    for i in range(Y_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        x_true = X[1:]  # [T,m]
        out = filter_obj.run(Y, x0_hat=X[0])
        x_hat = out[0] if isinstance(out, (tuple, list)) else out
        samples.append(mse_db(x_true, np.asarray(x_hat, dtype=float)))
    return samples


@torch.no_grad()
def eval_kalmannet_samples(model, X_test, Y_test) -> list[float]:
    model.eval()
    device = model.device
    samples = []

    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,m]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,n]
        T = Y.shape[0]

        # KNet has init_hidden_KNet, improved uses init_hidden
        if hasattr(model, "init_hidden_KNet"):
            model.init_hidden_KNet(batch_size=1)
        else:
            model.init_hidden(batch_size=1)

        x0_hat = X[0].view(1, -1, 1)
        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, -1, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()  # [T,m]
        X_true = X[1:].cpu().numpy()
        samples.append(mse_db(X_true, X_hat))

    return samples


# -------------------------
# Paper-style table extraction + plotting
# -------------------------
def nearest_indices(x_list: list[float], targets: list[float]) -> list[int]:
    x = np.asarray(x_list, dtype=float)
    idxs = []
    for t in targets:
        idxs.append(int(np.argmin(np.abs(x - t))))
    # keep order of targets, allow duplicates if sweep is coarse
    return idxs


def build_table_dict(inv_r2_db: list[float], curves_mu: dict, curves_sigma: dict,
                     targets_db: list[float], methods: list[str]) -> dict:
    """
    Returns a dict with selected points:
      columns: targets_db (requested) + actual_selected_db (from sweep)
      values: mu_hat, sigma_hat for each method
    """
    idxs = nearest_indices(inv_r2_db, targets_db)
    selected_actual = [float(inv_r2_db[i]) for i in idxs]

    table = {
        "targets_inv_r2_db": [float(t) for t in targets_db],
        "selected_inv_r2_db": selected_actual,
        "indices": idxs,
        "methods": {},
    }
    for m in methods:
        table["methods"][m] = {
            "mu_hat_db": [float(curves_mu[m][i]) for i in idxs],
            "sigma_hat_db": [float(curves_sigma[m][i]) for i in idxs],
        }
    return table


def save_table_png(path: Path, title: str, table: dict, methods: list[str]) -> None:
    """
    Save a paper-like table image (μ̂ row + σ̂ row for each method).
    """
    targets = table["targets_inv_r2_db"]
    selected = table["selected_inv_r2_db"]

    # Header like the paper: show selected inv_r2_db (since exact targets might not exist in sweep)
    header = ["1/r^2 [dB]"] + [f"{v:.2f}" for v in selected]

    # Build cell text: two rows per method (mu, sigma)
    cell_text = []
    row_labels = []
    for m in methods:
        mu = table["methods"][m]["mu_hat_db"]
        sg = table["methods"][m]["sigma_hat_db"]
        cell_text.append([f"{v:.2f}" for v in mu])
        row_labels.append(f"{m}  μ̂")
        cell_text.append([f"±{v:.2f}" for v in sg])
        row_labels.append(f"{m}  σ̂")

    fig, ax = plt.subplots(figsize=(12, 0.7 + 0.35 * len(row_labels)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=header,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    out_dir = get_results_dir()
    tag = now_tag()

    # -------------------------------
    # Nonlinear toy experiment
    # -------------------------------
    T = 100
    nu_db = -20.0

    params_true = dict(alpha=0.9, beta=1.1, phi=0.1 * np.pi, delta=0.01, a=1.0, b=1.0, c=0.0)
    params_partial = dict(alpha=1.0, beta=1.0, phi=0.0, delta=0.0, a=1.0, b=1.0, c=0.0)

    r2_list = np.logspace(-4, -1, 7)
    inv_r2_db = (10.0 * np.log10(1.0 / r2_list)).tolist()

    # Paper table uses points like [-12.04, -6.02, 0, 20, 40]
    # Your sweep might not hit them exactly; we’ll select the nearest.
    table_targets_db = [-12.04, -6.02, 0.0, 20.0, 40.0]

    N_train = 3000
    N_test = 500
    batch_size = 32
    epochs = 30

    use_cuda = True
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    methods = ["EKF", "UKF", "PF", "KalmanNet", "ImpKalmanNet"]

    # We store mu/sigma curves (paper table uses these)
    curves_mu = {
        "full": {m: [] for m in methods},
        "partial": {m: [] for m in methods},
    }
    curves_sigma = {
        "full": {m: [] for m in methods},
        "partial": {m: [] for m in methods},
    }

    # Optional: store raw per-trajectory MSE[dB] samples (big JSON but very useful)
    raw_samples = {
        "full": {m: [] for m in methods},     # list over r2 points, each entry is list length N_test
        "partial": {m: [] for m in methods},
    }

    for r2 in r2_list:
        q2, r2 = make_noise_from_nu(r2=float(r2), nu_db=nu_db)

        ssm = NonlinearToySSM(params_true=params_true, q2=q2, r2=r2, seed=123)
        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T)

        # ---------------- FULL ----------------
        ekf_full = EKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        ukf_full = UKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        pf_full = ParticleFilterNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_full_s = eval_mb_filter_samples(ekf_full, Xte, Yte)
        ukf_full_s = eval_mb_filter_samples(ukf_full, Xte, Yte)
        pf_full_s  = eval_mb_filter_samples(pf_full,  Xte, Yte)

        sys_full = NonlinearToySysModelTorch.from_numpy_params(
            params_design=params_true, prior_q2=q2, prior_r2=r2, device=device
        )

        kn_full = KalmanNetNN()
        kn_full.NNBuild(sys_full, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_full, sys_full, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_full_s = eval_kalmannet_samples(kn_full, Xte, Yte)

        imp_full = ImprovedNonlinearKalmanNet()
        imp_full.NNBuild(sys_full, ImprovedNLKNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult=5, out_mult=2, rnn_type="lstm"))
        train_improved_kalmannet_nonlinear(
            imp_full, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedNLTrainConfig(
                epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0,
                lambda_K_smooth=1e-3
            )
        )
        imp_full_s = eval_kalmannet_samples(imp_full, Xte, Yte)

        full_pack = {
            "EKF": ekf_full_s, "UKF": ukf_full_s, "PF": pf_full_s,
            "KalmanNet": kn_full_s, "ImpKalmanNet": imp_full_s
        }
        for name, samples in full_pack.items():
            mu, sg = summarize_mu_sigma(samples)
            curves_mu["full"][name].append(mu)
            curves_sigma["full"][name].append(sg)
            raw_samples["full"][name].append([float(x) for x in samples])

        # ---------------- PARTIAL ----------------
        ekf_part = EKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        ukf_part = UKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        pf_part  = ParticleFilterNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_part_s = eval_mb_filter_samples(ekf_part, Xte, Yte)
        ukf_part_s = eval_mb_filter_samples(ukf_part, Xte, Yte)
        pf_part_s  = eval_mb_filter_samples(pf_part,  Xte, Yte)

        sys_part = NonlinearToySysModelTorch.from_numpy_params(
            params_design=params_partial, prior_q2=q2, prior_r2=r2, device=device
        )

        kn_part = KalmanNetNN()
        kn_part.NNBuild(sys_part, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_part, sys_part, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_part_s = eval_kalmannet_samples(kn_part, Xte, Yte)

        imp_part = ImprovedNonlinearKalmanNet()
        imp_part.NNBuild(sys_part, ImprovedNLKNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult=5, out_mult=2, rnn_type="lstm"))
        train_improved_kalmannet_nonlinear(
            imp_part, Xtr, Ytr, batch_size=batch_size,
            cfg=ImprovedNLTrainConfig(
                epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0,
                lambda_K_smooth=1e-3
            )
        )
        imp_part_s = eval_kalmannet_samples(imp_part, Xte, Yte)

        part_pack = {
            "EKF": ekf_part_s, "UKF": ukf_part_s, "PF": pf_part_s,
            "KalmanNet": kn_part_s, "ImpKalmanNet": imp_part_s
        }
        for name, samples in part_pack.items():
            mu, sg = summarize_mu_sigma(samples)
            curves_mu["partial"][name].append(mu)
            curves_sigma["partial"][name].append(sg)
            raw_samples["partial"][name].append([float(x) for x in samples])

        # console (μ only)
        print(
            f"r2={r2:.2e} | FULL μ̂: EKF={curves_mu['full']['EKF'][-1]:.2f} UKF={curves_mu['full']['UKF'][-1]:.2f} "
            f"PF={curves_mu['full']['PF'][-1]:.2f} KNet={curves_mu['full']['KalmanNet'][-1]:.2f} Imp={curves_mu['full']['ImpKalmanNet'][-1]:.2f} "
            f"| PART μ̂: EKF={curves_mu['partial']['EKF'][-1]:.2f} UKF={curves_mu['partial']['UKF'][-1]:.2f} "
            f"PF={curves_mu['partial']['PF'][-1]:.2f} KNet={curves_mu['partial']['KalmanNet'][-1]:.2f} Imp={curves_mu['partial']['ImpKalmanNet'][-1]:.2f}"
        )

    # -------------------------
    # Plot (same style as your current code), but now plotting μ̂ (mean MSE[dB])
    # -------------------------
    plt.figure()
    for name in methods:
        plt.plot(inv_r2_db, curves_mu["full"][name], marker="o", label=f"{name} (full)")
    for name in methods:
        plt.plot(inv_r2_db, curves_mu["partial"][name], marker="o", linestyle="--", label=f"{name} (partial)")

    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB] (μ̂ over trajectories)")
    plt.grid(True)
    plt.legend()
    plt.title("Nonlinear Toy SSM: EKF/UKF/PF vs KalmanNet vs Improved (full & partial)")
    plt.tight_layout()

    fig_path = out_dir / f"nonlinear_toy_imp_mu_curves_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()

    # -------------------------
    # Build paper-style tables (full & partial) at target inv_r2_db points
    # -------------------------
    table_full = build_table_dict(inv_r2_db, curves_mu["full"], curves_sigma["full"], table_targets_db, methods)
    table_partial = build_table_dict(inv_r2_db, curves_mu["partial"], curves_sigma["partial"], table_targets_db, methods)

    # Save table images (optional but nice)
    table_full_png = out_dir / f"table_full_{tag}.png"
    table_part_png = out_dir / f"table_partial_{tag}.png"
    save_table_png(table_full_png, "TABLE: Synthetic Nonlinear SS Model (FULL) — μ̂ / σ̂", table_full, methods)
    save_table_png(table_part_png, "TABLE: Synthetic Nonlinear SS Model (PARTIAL) — μ̂ / σ̂", table_partial, methods)

    # -------------------------
    # Save JSON
    # -------------------------
    results = {
        "tag": tag,
        "T": T,
        "nu_db": nu_db,
        "params_true": params_true,
        "params_partial": params_partial,
        "r2_list": [float(x) for x in r2_list],
        "inv_r2_db": [float(x) for x in inv_r2_db],
        "curves_mu": curves_mu,            # μ̂ curves
        "curves_sigma": curves_sigma,      # σ̂ curves
        "paper_tables": {
            "targets_inv_r2_db": table_targets_db,
            "full": table_full,
            "partial": table_partial,
            "full_table_png": str(table_full_png),
            "partial_table_png": str(table_part_png),
        },
        "raw_mse_db_samples": raw_samples,  # remove if JSON too large
        "N_train": N_train,
        "N_test": N_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": str(device),
        "improved": {"rnn_type": "lstm", "lambda_K_smooth": 1e-3},
        "plots": {"mu_curve_png": str(fig_path)},
    }
    out_json = out_dir / f"nonlinear_toy_imp_mu_sigma_{tag}.json"
    save_json(results, out_json)

    print(f"\nSaved curve plot:   {fig_path}")
    print(f"Saved FULL table:   {table_full_png}")
    print(f"Saved PART table:   {table_part_png}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()

"""