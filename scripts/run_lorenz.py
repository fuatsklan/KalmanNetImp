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


# -------------------------
# IO utils
# -------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_dir() -> Path:
    out_dir = Path(__file__).parent / "lorenz_results"
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
    """
    inv_db = 10 log10(1/r^2)  =>  1/r^2 = 10^(inv_db/10)  =>  r^2 = 10^(-inv_db/10)
    """
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
# Evaluation (samples, not just mean)
# -------------------------
def eval_filter_samples(filter_obj, X_test, Y_test) -> list[float]:
    samples = []
    for i in range(Y_test.shape[0]):
        X, Y = X_test[i], Y_test[i]
        x_true = X[1:]  # [T,3]
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

        if hasattr(model, "init_hidden_KNet"):
            model.init_hidden_KNet(batch_size=1)
        else:
            model.init_hidden(batch_size=1)

        x0_hat = X[0].view(1, 3, 1)
        model.InitSequence(x0_hat, T=T)

        xhats = []
        for t in range(T):
            yt = Y[t].view(1, 3, 1)
            x_post = model(yt).squeeze(0).squeeze(-1)
            xhats.append(x_post)

        X_hat = torch.stack(xhats, dim=0).cpu().numpy()  # [T,3]
        X_true = X[1:].cpu().numpy()
        samples.append(mse_db(X_true, X_hat))

    return samples


# -------------------------
# One sweep run (returns curves + table rows)
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
) -> dict:
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    # store curves (μ̂ only) and also table-style μ̂/σ̂
    curves_mu = {"EKF": [], "UKF": [], "PF": [], "KalmanNet": []}
    curves_sigma = {"EKF": [], "UKF": [], "PF": [], "KalmanNet": []}

    rows = []  # per inv_r2_db point

    for inv_db in inv_r2_points_db:
        r2 = inv_r2_db_to_r2(inv_db)
        q2, r2 = make_noise_from_nu(r2=r2, nu_db=nu_db)

        # true generator
        ssm = LorenzSSM(dt=dt, J=J_true, q2=q2, r2=r2, obs_type=obs_type, seed=123)

        # long trajectories for truncated BPTT training
        Xtr_long, Ytr_long = generate_dataset(ssm, N=N_train, T=T_train_long)

        # test long trajectories
        Xte, Yte = generate_dataset(ssm, N=N_test, T=T_test)

        # MB filters with the SAME discrete model used to generate
        f = lambda x: ssm.f(x)
        h = lambda x: ssm.h(x)

        ekf = EKFGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R)
        ukf = UKFGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R)
        pf = ParticleFilterGeneric(f=f, h=h, Q=ssm.Q, R=ssm.R, n_particles=n_particles)

        ekf_s = eval_filter_samples(ekf, Xte, Yte)
        ukf_s = eval_filter_samples(ukf, Xte, Yte)
        pf_s  = eval_filter_samples(pf,  Xte, Yte)

        # KalmanNet C3 (Arch1 + features)
        sys_torch = LorenzSysModelTorch.build(
            dt=dt,
            J=J_model,
            prior_q2=q2,
            prior_r2=r2,
            obs_type=obs_type,
            rot_deg=0.0,
            device=device,
        )

        kn = KalmanNetArch1(feature_set=["F1", "F3", "F4"])
        kn.NNBuild(sys_torch, KNetArch1Args(use_cuda=use_cuda, n_batch=batch_size, hidden_mult=10, n_gru_layers=1))

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

        # summarize
        ekf_mu, ekf_sig = summarize_mu_sigma(ekf_s)
        ukf_mu, ukf_sig = summarize_mu_sigma(ukf_s)
        pf_mu,  pf_sig  = summarize_mu_sigma(pf_s)
        kn_mu,  kn_sig  = summarize_mu_sigma(kn_s)

        curves_mu["EKF"].append(ekf_mu)
        curves_mu["UKF"].append(ukf_mu)
        curves_mu["PF"].append(pf_mu)
        curves_mu["KalmanNet"].append(kn_mu)

        curves_sigma["EKF"].append(ekf_sig)
        curves_sigma["UKF"].append(ukf_sig)
        curves_sigma["PF"].append(pf_sig)
        curves_sigma["KalmanNet"].append(kn_sig)

        rows.append({
            "inv_r2_db": float(inv_db),
            "r2": float(r2),
            "q2": float(q2),
            "mu_hat_db": {"EKF": ekf_mu, "UKF": ukf_mu, "PF": pf_mu, "KalmanNet": kn_mu},
            "sigma_hat_db": {"EKF": ekf_sig, "UKF": ukf_sig, "PF": pf_sig, "KalmanNet": kn_sig},
        })

        print(
            f"[{exp_name}] inv={inv_db:>6.2f} dB | "
            f"EKF={ekf_mu:>7.2f} UKF={ukf_mu:>7.2f} PF={pf_mu:>7.2f} KNet={kn_mu:>7.2f}"
        )

    # ---- Plot sweep curves (μ̂)
    plt.figure()
    x = inv_r2_points_db
    plt.plot(x, curves_mu["EKF"], marker="o", label="EKF")
    plt.plot(x, curves_mu["UKF"], marker="o", label="UKF")
    plt.plot(x, curves_mu["PF"], marker="o", label="PF")
    plt.plot(x, curves_mu["KalmanNet"], marker="o", label="KalmanNet")

    plt.xlabel("10log10(1/r^2) [dB]")
    plt.ylabel("MSE [dB] (μ̂ over trajectories)")
    plt.grid(True)
    plt.legend()
    plt.title(exp_name)
    plt.tight_layout()

    sweep_fig = out_dir / f"{exp_name}_sweep_{tag}.png"
    plt.savefig(sweep_fig, dpi=200)
    plt.close()

    # ---- Save a “single trajectory” plot with MSE numbers in title
    # Use the middle noise point for the demo plot
    mid_idx = len(inv_r2_points_db) // 2
    inv_mid = inv_r2_points_db[mid_idx]
    r2_mid = inv_r2_db_to_r2(inv_mid)
    q2_mid, r2_mid = make_noise_from_nu(r2=r2_mid, nu_db=nu_db)
    ssm_mid = LorenzSSM(dt=dt, J=J_true, q2=q2_mid, r2=r2_mid, obs_type=obs_type, seed=123)
    Xdemo, Ydemo = generate_dataset(ssm_mid, N=1, T=T_test)
    Xdemo, Ydemo = Xdemo[0], Ydemo[0]

    f = lambda x: ssm_mid.f(x)
    h = lambda x: ssm_mid.h(x)
    ekf_demo = EKFGeneric(f=f, h=h, Q=ssm_mid.Q, R=ssm_mid.R)
    xhat_ekf = ekf_demo.run(Ydemo, x0_hat=Xdemo[0])
    if isinstance(xhat_ekf, (tuple, list)):
        xhat_ekf = xhat_ekf[0]

    # re-train a knet quickly for the mid point (keeps the demo consistent)
    # (you can skip this if you want, but then you'd need to cache models per point)
    Xtr_long, Ytr_long = generate_dataset(ssm_mid, N=N_train, T=T_train_long)
    sys_torch = LorenzSysModelTorch.build(
        dt=dt, J=J_model, prior_q2=q2_mid, prior_r2=r2_mid, obs_type=obs_type, rot_deg=0.0, device=device
    )
    kn_demo = KalmanNetArch1(feature_set=["F1", "F3", "F4"])
    kn_demo.NNBuild(sys_torch, KNetArch1Args(use_cuda=use_cuda, n_batch=batch_size, hidden_mult=10, n_gru_layers=1))
    train_kalmannet_truncated_bptt(
        kn_demo,
        X_train_long=Xtr_long,
        Y_train_long=Ytr_long,
        batch_size=batch_size,
        cfg=TruncBPTTCfg(
            epochs=max(3, epochs // 3),
            lr=1e-3, weight_decay=0.0, grad_clip=1.0,
            chunk_len=chunk_len, stride=stride
        ),
    )

    kn_demo.eval()
    if hasattr(kn_demo, "init_hidden"):
        kn_demo.init_hidden(batch_size=1)
    x0_hat = torch.tensor(Xdemo[0], dtype=torch.float32, device=kn_demo.device).view(1, 3, 1)
    Yt = torch.tensor(Ydemo, dtype=torch.float32, device=kn_demo.device)
    kn_demo.InitSequence(x0_hat, T=Yt.shape[0])
    with torch.no_grad():
        xs = []
        for t in range(Yt.shape[0]):
            yt = Yt[t].view(1, 3, 1)
            xs.append(kn_demo(yt).squeeze(0).squeeze(-1))
        xhat_kn = torch.stack(xs, dim=0).cpu().numpy()

    mse_ekf_demo = mse_db(Xdemo[1:], xhat_ekf)
    mse_kn_demo = mse_db(Xdemo[1:], xhat_kn)

    plt.figure(figsize=(10, 4))
    plt.plot(Xdemo[1:, 0], label="x1 true")
    plt.plot(xhat_ekf[:, 0], label=f"x1 EKF (MSE={mse_ekf_demo:.2f} dB)")
    plt.plot(xhat_kn[:, 0], label=f"x1 KalmanNet (MSE={mse_kn_demo:.2f} dB)")
    plt.grid(True)
    plt.legend()
    plt.title(f"{exp_name} | demo at inv_r2_db={inv_mid:.2f} dB")
    plt.tight_layout()

    demo_fig = out_dir / f"{exp_name}_demo_{tag}.png"
    plt.savefig(demo_fig, dpi=200)
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
        "inv_r2_points_db": [float(x) for x in inv_r2_points_db],
        "curves_mu": curves_mu,
        "curves_sigma": curves_sigma,
        "rows": rows,
        "plots": {"sweep_png": str(sweep_fig), "demo_png": str(demo_fig)},
    }


def main():
    out_dir = get_results_dir()
    tag = now_tag()

    # Table V points (noisy state observations): 0,10,20,30,40
    inv_points_tableV = [0.0, 10.0, 20.0, 30.0, 40.0]

    # Table VI points (nonlinear observations): -10,0,10,20,30
    inv_points_tableVI = [-10.0, 0.0, 10.0, 20.0, 30.0]

    # Common settings (tune if slow)
    common = dict(
        dt=0.02,
        J_true=5,
        J_model=5,
        nu_db=-20.0,
        T_train_long=2000,
        T_test=2000,
        chunk_len=100,
        stride=100,
        N_train=200,     # increase later for better results
        N_test=100,
        batch_size=32,
        epochs=10,       # increase later
        n_particles=100,
        use_cuda=True,
    )

    results = {"tag": tag, "experiments": {}}

    # -------------------------
    # Table V-like: identity observations
    # -------------------------
    expV = run_lorenz_sweep(
        out_dir=out_dir,
        tag=tag,
        exp_name="TABLE_V_identity_obs",
        obs_type="identity",
        inv_r2_points_db=inv_points_tableV,
        **common,
    )
    results["experiments"][expV["exp_name"]] = expV

    # -------------------------
    # Table VI-like: nonlinear observations (your spherical)
    # -------------------------
    expVI = run_lorenz_sweep(
        out_dir=out_dir,
        tag=tag,
        exp_name="TABLE_VI_spherical_obs",
        obs_type="spherical",
        inv_r2_points_db=inv_points_tableVI,
        **common,
    )
    results["experiments"][expVI["exp_name"]] = expVI

    # Save JSON
    out_json = out_dir / f"lorenz_tables_{tag}.json"
    save_json(results, out_json)

    print(f"\nSaved JSON: {out_json}")
    for k, v in results["experiments"].items():
        print(f"{k}: sweep={v['plots']['sweep_png']} demo={v['plots']['demo_png']}")


if __name__ == "__main__":
    main()
