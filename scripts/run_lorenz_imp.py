# scripts/run_lorenz_imp.py
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

from models.lorenz import LorenzSSM, make_noise_from_nu
from algos.kf_ukf_pf.nonlinear_filters_generic import EKFGeneric, UKFGeneric, ParticleFilterGeneric

from algos.kalmannet.lorenz_sysmodel import LorenzSysModelTorch
from algos.kalmannet.kalmannet_arch1 import KalmanNetArch1, KNetArch1Args
from algos.kalmannet.train_truncated import train_kalmannet_truncated_bptt, TruncBPTTCfg

from algos.improvements.lorenz_kalmannet_arch1 import ImprovedKalmanNetArch1, ImpKNetArch1Args
from algos.improvements.train_truncated_improved import (
    train_improved_kalmannet_truncated_bptt,
    ImpTruncBPTTCfg,
)


# -------------------------
# IO utils
# -------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "lorenz_imp_results"
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
    arr = np.asarray(samples_db, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def inv_r2_db_to_r2(inv_db: float) -> float:
    return 10.0 ** (-inv_db / 10.0)


# -------------------------
# Data generation
# -------------------------
def generate_dataset(ssm: LorenzSSM, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _ in range(N):
        X, Y = ssm.sample(T=T)
        Xs.append(X)
        Ys.append(Y)
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)


# -------------------------
# Evaluation (samples)
# -------------------------
def eval_filter_samples(filter_obj, X_test, Y_test) -> list[float]:
    samples = []
    for i in range(Y_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        x_true = X[1:]
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
        X = torch.tensor(X_test[i], dtype=torch.float32, device=device)  # [T+1,3]
        Y = torch.tensor(Y_test[i], dtype=torch.float32, device=device)  # [T,3]
        T = Y.shape[0]

        # Arch1 + improved arch1 both have init_hidden()
        model.init_hidden(batch_size=1)

        x0_hat = X[0].view(1, 3, 1)
        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, 3, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()
        X_true = X[1:].cpu().numpy()
        samples.append(mse_db(X_true, X_hat))

    return samples


# -------------------------
# One sweep run
# -------------------------
def run_lorenz_sweep(
    *,
    out_dir: Path,
    tag: str,
    exp_name: str,
    obs_type: str,
    inv_r2_points_db: list[float],
    dt: float = 0.02,
    J_true: int = 5,
    J_model: int = 5,
    nu_db: float = -20.0,
    T_train_long: int = 2000,
    T_test: int = 2000,
    chunk_len: int = 100,
    stride: int = 100,
    N_train: int = 200,
    N_test: int = 100,
    batch_size: int = 32,
    epochs: int = 10,
    n_particles: int = 100,
    use_cuda: bool = True,
    # improved options
    imp_rnn_type: str = "lstm",
    lambda_K_smooth: float = 1e-3,
) -> dict:
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    curves_mu = {"EKF": [], "UKF": [], "PF": [], "KalmanNet": [], "ImpKalmanNet": []}
    curves_sigma = {"EKF": [], "UKF": [], "PF": [], "KalmanNet": [], "ImpKalmanNet": []}
    rows = []

    for inv_db in inv_r2_points_db:
        r2 = inv_r2_db_to_r2(inv_db)
        q2, r2 = make_noise_from_nu(r2=r2, nu_db=nu_db)

        ssm = LorenzSSM(dt=dt, J=J_true, q2=q2, r2=r2, obs_type=obs_type, seed=123)

        Xtr_long, Ytr_long = generate_dataset(ssm, N=N_train, T=T_train_long)
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_test)

        f = lambda x: ssm.f(x)
        h = lambda x: ssm.h(x)

        ekf = EKFGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R)
        ukf = UKFGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R)
        pf = ParticleFilterGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R, n_particles=n_particles)

        ekf_s = eval_filter_samples(ekf, Xte, Yte)
        ukf_s = eval_filter_samples(ukf, Xte, Yte)
        pf_s = eval_filter_samples(pf, Xte, Yte)

        sys_torch = LorenzSysModelTorch.build(
            dt=dt,
            J=J_model,
            prior_q2=q2,
            prior_r2=r2,
            obs_type=obs_type,
            rot_deg=0.0,
            device=device,
        )

        # ----------------
        # KalmanNet C3 (GRU)
        # ----------------
        kn = KalmanNetArch1(feature_set=["F1", "F3", "F4"])
        kn.NNBuild(
            sys_torch,
            KNetArch1Args(
                use_cuda=use_cuda,
                n_batch=batch_size,
                hidden_mult=10,
                n_gru_layers=1,
                feat_norm_eps=1e-3,
                enable_nan_guards=False,
            ),
        )

        train_kalmannet_truncated_bptt(
            kn,
            X_train_long=Xtr_long,
            Y_train_long=Ytr_long,
            batch_size=batch_size,
            cfg=TruncBPTTCfg(
                epochs=epochs,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                chunk_len=chunk_len,
                stride=stride,
            ),
        )

        kn_s = eval_kalmannet_samples(kn, Xte, Yte)

        # ----------------
        # Improved KalmanNet (LSTM + K-smoothness)
        # ----------------
        imp = ImprovedKalmanNetArch1(feature_set=["F1", "F3", "F4"])
        imp.NNBuild(
            sys_torch,
            ImpKNetArch1Args(
                use_cuda=use_cuda,
                n_batch=batch_size,
                hidden_mult=10,
                n_rnn_layers=1,
                rnn_type=imp_rnn_type,
                feat_norm_eps=1e-3,
                enable_nan_guards=False,
            ),
        )

        train_improved_kalmannet_truncated_bptt(
            imp,
            X_train_long=Xtr_long,
            Y_train_long=Ytr_long,
            batch_size=batch_size,
            cfg=ImpTruncBPTTCfg(
                epochs=epochs,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                chunk_len=chunk_len,
                stride=stride,
                lambda_K_smooth=lambda_K_smooth,
            ),
        )

        imp_s = eval_kalmannet_samples(imp, Xte, Yte)

        # ---- summarize mu/sigma
        ekf_mu, ekf_sig = summarize_mu_sigma(ekf_s)
        ukf_mu, ukf_sig = summarize_mu_sigma(ukf_s)
        pf_mu, pf_sig = summarize_mu_sigma(pf_s)
        kn_mu, kn_sig = summarize_mu_sigma(kn_s)
        imp_mu, imp_sig = summarize_mu_sigma(imp_s)

        for k, mu, sig in [
            ("EKF", ekf_mu, ekf_sig),
            ("UKF", ukf_mu, ukf_sig),
            ("PF", pf_mu, pf_sig),
            ("KalmanNet", kn_mu, kn_sig),
            ("ImpKalmanNet", imp_mu, imp_sig),
        ]:
            curves_mu[k].append(mu)
            curves_sigma[k].append(sig)

        rows.append(
            {
                "inv_r2_db": float(inv_db),
                "r2": float(r2),
                "q2": float(q2),
                "mu_hat_db": {
                    "EKF": ekf_mu,
                    "UKF": ukf_mu,
                    "PF": pf_mu,
                    "KalmanNet": kn_mu,
                    "ImpKalmanNet": imp_mu,
                },
                "sigma_hat_db": {
                    "EKF": ekf_sig,
                    "UKF": ukf_sig,
                    "PF": pf_sig,
                    "KalmanNet": kn_sig,
                    "ImpKalmanNet": imp_sig,
                },
            }
        )

        print(
            f"[{exp_name}] inv={inv_db:>6.2f} dB | "
            f"EKF={ekf_mu:>7.2f} UKF={ukf_mu:>7.2f} PF={pf_mu:>7.2f} "
            f"KNet={kn_mu:>7.2f} Imp={imp_mu:>7.2f}"
        )

    # ---- Plot sweep (mu only)
    plt.figure()
    x = inv_r2_points_db
    plt.plot(x, curves_mu["EKF"], marker="o", label="EKF")
    plt.plot(x, curves_mu["UKF"], marker="o", label="UKF")
    plt.plot(x, curves_mu["PF"], marker="o", label="PF")
    plt.plot(x, curves_mu["KalmanNet"], marker="o", label="KalmanNet")
    plt.plot(x, curves_mu["ImpKalmanNet"], marker="o", label="ImpKalmanNet")

    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB] (mu_hat over trajectories)")
    plt.grid(True)
    plt.legend()
    plt.title(exp_name)
    plt.tight_layout()

    sweep_fig = out_dir / f"{exp_name}_sweep_{tag}.png"
    plt.savefig(sweep_fig, dpi=200)
    plt.close()

    return {
        "exp_name": exp_name,
        "obs_type": obs_type,
        "dt": dt,
        "J_true": J_true,
        "J_model": J_model,
        "nu_db": float(nu_db),
        "T_train_long": int(T_train_long),
        "T_test": int(T_test),
        "chunk_len": int(chunk_len),
        "stride": int(stride),
        "N_train": int(N_train),
        "N_test": int(N_test),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "device": str(device),
        "inv_r2_points_db": [float(v) for v in inv_r2_points_db],
        "curves_mu": curves_mu,
        "curves_sigma": curves_sigma,
        "rows": rows,
        "plots": {"sweep_png": str(sweep_fig)},
        "improved": {"rnn_type": imp_rnn_type, "lambda_K_smooth": float(lambda_K_smooth)},
    }


def main():
    out_dir = get_results_dir()
    tag = now_tag()

    # Table V points (identity obs): 0,10,20,30,40
    inv_points_tableV = [0.0, 10.0, 20.0, 30.0, 40.0]

    # Table VI points (spherical obs): -10,0,10,20,30
    inv_points_tableVI = [-10.0, 0.0, 10.0, 20.0, 30.0]

    common = dict(
        dt=0.02,
        J_true=5,
        J_model=5,
        nu_db=-20.0,
        T_train_long=2000,
        T_test=2000,
        chunk_len=100,
        stride=100,
        N_train=200,
        N_test=100,
        batch_size=32,
        epochs=10,
        n_particles=100,
        use_cuda=True,
        imp_rnn_type="lstm",
        lambda_K_smooth=1e-3,
    )

    results = {"tag": tag, "experiments": {}}

    expV = run_lorenz_sweep(
        out_dir=out_dir,
        tag=tag,
        exp_name="TABLE_V_identity_obs_imp",
        obs_type="identity",
        inv_r2_points_db=inv_points_tableV,
        **common,
    )
    results["experiments"][expV["exp_name"]] = expV

    expVI = run_lorenz_sweep(
        out_dir=out_dir,
        tag=tag,
        exp_name="TABLE_VI_spherical_obs_imp",
        obs_type="spherical",
        inv_r2_points_db=inv_points_tableVI,
        **common,
    )
    results["experiments"][expVI["exp_name"]] = expVI

    out_json = out_dir / f"lorenz_imp_tables_{tag}.json"
    save_json(results, out_json)

    print(f"\nSaved JSON: {out_json}")
    for k, v in results["experiments"].items():
        print(f"{k}: sweep={v['plots']['sweep_png']}")

    # Optional: also save a compact "table-like" text file
    table_txt = out_dir / f"lorenz_imp_tables_{tag}.txt"
    lines = []
    for exp_name, exp in results["experiments"].items():
        lines.append(exp_name)
        lines.append("inv_r2_db | EKF(mu±sig) | UKF(mu±sig) | PF(mu±sig) | KNet(mu±sig) | Imp(mu±sig)")
        for row in exp["rows"]:
            invdb = row["inv_r2_db"]
            mu = row["mu_hat_db"]
            sig = row["sigma_hat_db"]
            def fmt(k): return f"{mu[k]:.2f}±{sig[k]:.2f}"
            lines.append(
                f"{invdb:>7.2f} | {fmt('EKF'):>12} | {fmt('UKF'):>12} | {fmt('PF'):>12} | "
                f"{fmt('KalmanNet'):>12} | {fmt('ImpKalmanNet'):>12}"
            )
        lines.append("")
    table_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved TXT table: {table_txt}")


if __name__ == "__main__":
    main()
