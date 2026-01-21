# scripts/run_nonlinear_ss.py
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


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "nonlinear_results"
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
        x_true = X[1:]  # [T,2]

        out = filter_obj.run(Y, x0_hat=X[0])

        if isinstance(out, tuple) or isinstance(out, list):
            x_hat = out[0]
        else:
            x_hat = out

        x_hat = np.asarray(x_hat, dtype=float)  # ensure numeric array
        mses.append(mse_db(x_true, x_hat))

    return float(np.mean(mses))



@torch.no_grad()
def eval_kalmannet(model: KalmanNetNN, X_test, Y_test) -> float:
    model.eval()
    device = model.device
    mses = []

    for i in range(Y_test.shape[0]):
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,2]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,2]
        T = Y.shape[0]

        model.init_hidden_KNet(batch_size=1)
        x0_hat = X[0].view(1, 2, 1)
        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, 2, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)  # [2]
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()  # [T,2]
        X_true = X[1:].cpu().numpy()
        mses.append(mse_db(X_true, X_hat))

    return float(np.mean(mses))


def main():
    out_dir = get_results_dir()
    tag = now_tag()

    # -------------------------------
    # Paper nonlinear toy experiment
    # -------------------------------
    T = 100
    nu_db = -20.0

    params_true = dict(alpha=0.9, beta=1.1, phi=0.1 * np.pi, delta=0.01, a=1.0, b=1.0, c=0.0)
    params_partial = dict(alpha=1.0, beta=1.0, phi=0.0, delta=0.0, a=1.0, b=1.0, c=0.0)

    # Sweep observation noise
    r2_list = np.logspace(-4, -1, 7)
    inv_r2_db = 10.0 * np.log10(1.0 / r2_list)

    # Data sizes (tune if slow)
    N_train = 3000
    N_test = 500
    batch_size = 32
    epochs = 30

    # Device
    use_cuda = True
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    # Store curves for plotting like Fig.7
    curves = {
        "full": {"EKF": [], "UKF": [], "PF": [], "KalmanNet": []},
        "partial": {"EKF": [], "UKF": [], "PF": [], "KalmanNet": []},
    }

    for r2 in r2_list:
        q2, r2 = make_noise_from_nu(r2=float(r2), nu_db=nu_db)

        # True data generated from FULL params
        ssm = NonlinearToySSM(params_true=params_true, q2=q2, r2=r2, seed=123)
        Xtr, Ytr = generate_dataset(ssm, N=N_train, T=T)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T)

        # -------- FULL information evaluation (filters know true params) --------
        ekf_full = EKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        ukf_full = UKFNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R)
        pf_full = ParticleFilterNonlinearToy(params_design=params_true, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_full_mse = eval_mb_filter(ekf_full, Xte, Yte)
        ukf_full_mse = eval_mb_filter(ukf_full, Xte, Yte)
        pf_full_mse = eval_mb_filter(pf_full, Xte, Yte)

        sys_full = NonlinearToySysModelTorch.from_numpy_params(params_design=params_true, prior_q2=q2, prior_r2=r2, device=device)
        kn_full = KalmanNetNN()
        kn_full.NNBuild(sys_full, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_full, sys_full, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_full_mse = eval_kalmannet(kn_full, Xte, Yte)

        # -------- PARTIAL information evaluation (filters use mismatched params) --------
        ekf_part = EKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        ukf_part = UKFNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R)
        pf_part = ParticleFilterNonlinearToy(params_design=params_partial, Q=ssm.Q, R=ssm.R, n_particles=100)

        ekf_part_mse = eval_mb_filter(ekf_part, Xte, Yte)
        ukf_part_mse = eval_mb_filter(ukf_part, Xte, Yte)
        pf_part_mse = eval_mb_filter(pf_part, Xte, Yte)

        sys_part = NonlinearToySysModelTorch.from_numpy_params(params_design=params_partial, prior_q2=q2, prior_r2=r2, device=device)
        kn_part = KalmanNetNN()
        kn_part.NNBuild(sys_part, KNetArgs(use_cuda=use_cuda, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=2))
        train_kalmannet_linear(
            kn_part, sys_part, Xtr, Ytr, batch_size=batch_size,
            cfg=TrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.0, grad_clip=1.0)
        )
        kn_part_mse = eval_kalmannet(kn_part, Xte, Yte)

        # Save points
        curves["full"]["EKF"].append(ekf_full_mse)
        curves["full"]["UKF"].append(ukf_full_mse)
        curves["full"]["PF"].append(pf_full_mse)
        curves["full"]["KalmanNet"].append(kn_full_mse)

        curves["partial"]["EKF"].append(ekf_part_mse)
        curves["partial"]["UKF"].append(ukf_part_mse)
        curves["partial"]["PF"].append(pf_part_mse)
        curves["partial"]["KalmanNet"].append(kn_part_mse)

        print(
            f"r2={r2:.2e} | FULL: EKF={ekf_full_mse:.2f} UKF={ukf_full_mse:.2f} PF={pf_full_mse:.2f} KNet={kn_full_mse:.2f} "
            f"| PART: EKF={ekf_part_mse:.2f} UKF={ukf_part_mse:.2f} PF={pf_part_mse:.2f} KNet={kn_part_mse:.2f}"
        )

    # -------- Plot curves (Fig.7 style) --------
    plt.figure()
    plt.plot(inv_r2_db, curves["full"]["EKF"], marker="o", label="EKF (full)")
    plt.plot(inv_r2_db, curves["full"]["UKF"], marker="o", label="UKF (full)")
    plt.plot(inv_r2_db, curves["full"]["PF"], marker="o", label="PF (full)")
    plt.plot(inv_r2_db, curves["full"]["KalmanNet"], marker="o", label="KalmanNet (full)")

    plt.plot(inv_r2_db, curves["partial"]["EKF"], marker="o", linestyle="--", label="EKF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["UKF"], marker="o", linestyle="--", label="UKF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["PF"], marker="o", linestyle="--", label="PF (partial)")
    plt.plot(inv_r2_db, curves["partial"]["KalmanNet"], marker="o", linestyle="--", label="KalmanNet (partial)")

    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB]")
    plt.grid(True)
    plt.legend()
    plt.title("Nonlinear Toy SSM: EKF/UKF/PF vs KalmanNet (full & partial)")
    plt.tight_layout()

    fig_path = out_dir / f"nonlinear_toy_curves_{tag}.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()

    # Save JSON
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
    }
    save_json(results, out_dir / f"nonlinear_toy_results_{tag}.json")


if __name__ == "__main__":
    main()
