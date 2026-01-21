# models/nclt_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class NCLTSplits:
    X_train: np.ndarray  # [Ntr, T+1, 4]   state [px,vx,py,vy]
    Y_train: np.ndarray  # [Ntr, T, 2]     meas  [vx,vy]
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray   # [1, Ttest+1, 4]
    Y_test: np.ndarray   # [1, Ttest, 2]
    meta: dict


def _load_csv_no_header(path: Path, n_cols: int) -> np.ndarray:
    A = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != n_cols:
        raise ValueError(f"{path} expected {n_cols} cols, got {A.shape}")
    return A


def _interp1(ts_query_us: np.ndarray, ts_us: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation for 1D or 2D values along time."""
    ts_query_us = np.asarray(ts_query_us, dtype=np.int64)
    ts_us = np.asarray(ts_us, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    t0 = ts_us[0]
    tq = (ts_query_us - t0) / 1e6
    tt = (ts_us - t0) / 1e6

    if values.ndim == 1:
        return np.interp(tq, tt, values)
    else:
        out = []
        for k in range(values.shape[1]):
            out.append(np.interp(tq, tt, values[:, k]))
        return np.stack(out, axis=1)


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _interp_yaw(ts_query_us: np.ndarray, gt_ts_us: np.ndarray, gt_yaw: np.ndarray) -> np.ndarray:
    """
    Interpolate yaw robustly by unwrapping first.
    """
    yaw = np.asarray(gt_yaw, dtype=np.float64)
    yaw_unwrap = np.unwrap(yaw)  # remove 2pi jumps
    yaw_q = _interp1(ts_query_us, gt_ts_us, yaw_unwrap)
    return _wrap_to_pi(yaw_q)


def load_nclt_node_odometry(
    root: Path,
    session_dirname: str = "2012-01-22",
    groundtruth_name: str = "groundtruth.csv",
    odom_node_name: str = "odometry_mu.csv",
    odom_dxdy_cols: tuple[int, int] = (1, 2),  # dx_body, dy_body (per odom node)
    target_dt_s: float = 1.0,
    max_speed_mps: float = 10.0,
    min_dt_s: float = 1e-3,
    assume_body_frame: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Paper-like NCLT construction for the Wiener velocity model:

      state x_t = [px, vx, py, vy]
      measurement y_t = noisy velocity [vx, vy] from odometry only

    Key fixes:
    - rotate odometry (dx,dy) into world frame using GT yaw if assume_body_frame=True
    - aggregate to ~1Hz and compute velocity as Δp / Δt
    """
    session_path = root / session_dirname
    gt_path = session_path / groundtruth_name
    od_path = session_path / odom_node_name

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not od_path.exists():
        raise FileNotFoundError(od_path)

    gt = _load_csv_no_header(gt_path, 7)
    od = _load_csv_no_header(od_path, 7)

    # Ground truth cols: [ts, x, y, z, roll, pitch, yaw]
    gt_ts = gt[:, 0].astype(np.int64)
    gt_xy = gt[:, 1:3].astype(np.float64)
    gt_yaw = gt[:, 6].astype(np.float64)

    good_gt = np.isfinite(gt_ts) & np.isfinite(gt_xy).all(axis=1) & np.isfinite(gt_yaw)
    gt_ts = gt_ts[good_gt]
    gt_xy = gt_xy[good_gt]
    gt_yaw = gt_yaw[good_gt]

    # Odometry node stream: 7 cols numeric
    od_ts = od[:, 0].astype(np.int64)
    jx, jy = odom_dxdy_cols
    od_dxy = od[:, [jx, jy]].astype(np.float64)

    good_od = np.isfinite(od_ts) & np.isfinite(od_dxy).all(axis=1)
    od_ts = od_ts[good_od]
    od_dxy = od_dxy[good_od]

    if od_ts.shape[0] < 3:
        raise ValueError("Not enough odometry rows after filtering.")

    # Build per-step increments aligned to intervals [t-1 -> t]
    dt_raw = (od_ts[1:] - od_ts[:-1]) / 1e6
    dt_raw = np.maximum(dt_raw, min_dt_s)

    dxy_body = od_dxy[1:]  # increment associated with dt_raw interval
    ts_used = od_ts[1:]    # timestamp at the end of interval

    # Rotate into world frame using yaw at ts_used (end of interval)
    if assume_body_frame:
        yaw = _interp_yaw(ts_used, gt_ts, gt_yaw)  # [T]
        c = np.cos(yaw)
        s = np.sin(yaw)
        dx = dxy_body[:, 0]
        dy = dxy_body[:, 1]
        # world = R(yaw) * body
        dxy_world = np.stack([c * dx - s * dy, s * dx + c * dy], axis=1)
    else:
        dxy_world = dxy_body

    # Bin to regular target_dt_s (≈1Hz)
    t0 = ts_used[0]
    t_sec = (ts_used - t0) / 1e6  # seconds from start
    bin_id = np.floor(t_sec / target_dt_s).astype(int)

    # Accumulate displacement within each bin
    nbins = int(bin_id.max()) + 1
    dp = np.zeros((nbins, 2), dtype=np.float64)
    dt_bin = np.zeros((nbins,), dtype=np.float64)

    for k in range(len(bin_id)):
        b = bin_id[k]
        dp[b] += dxy_world[k]
        dt_bin[b] += dt_raw[k]

    # Remove empty/degenerate bins
    keep = (dt_bin > 0.5 * target_dt_s) & np.isfinite(dp).all(axis=1)
    dp = dp[keep]
    dt_bin = dt_bin[keep]

    # Velocity measurements (paper: noisy velocity readings)
    v_meas = dp / dt_bin[:, None]  # [T,2]

    # Drop unstable spikes
    speed = np.linalg.norm(v_meas, axis=1)
    keep2 = np.isfinite(speed) & (speed < max_speed_mps)
    v_meas = v_meas[keep2]
    dt_bin = dt_bin[keep2]

    # Construct timestamps for each 1Hz sample (center/end of bin)
    # We’ll just use synthetic equally spaced times for GT interpolation:
    # t_k = t0 + k*target_dt_s
    T = v_meas.shape[0]
    ts_bins = t0 + (np.arange(T) * target_dt_s * 1e6).astype(np.int64)

    # Interpolate GT positions at bin times
    gt_xy_used = _interp1(ts_bins, gt_ts, gt_xy)  # [T,2]

    # Build GT velocity from GT positions (for state only)
    gt_v = np.zeros_like(gt_xy_used)
    dtg = np.maximum(dt_bin, min_dt_s)
    gt_v[1:] = (gt_xy_used[1:] - gt_xy_used[:-1]) / dtg[1:, None]
    gt_v[0] = gt_v[1]

    px, py = gt_xy_used[:, 0], gt_xy_used[:, 1]
    vx, vy = gt_v[:, 0], gt_v[:, 1]

    X = np.stack([px, vx, py, vy], axis=1)        # [T,4]
    X_true = np.concatenate([X[:1], X], axis=0)   # [T+1,4]
    Y_meas = v_meas                               # [T,2]

    meta = {
        "gt_path": str(gt_path),
        "odom_path": str(od_path),
        "odom_dxdy_cols": odom_dxdy_cols,
        "target_dt_s": float(target_dt_s),
        "assume_body_frame": bool(assume_body_frame),
        "max_speed_mps": float(max_speed_mps),
        "T_total": int(Y_meas.shape[0]),
        "dt_used_mean": float(np.mean(dt_bin)) if dt_bin.size else None,
        "dt_used_min": float(np.min(dt_bin)) if dt_bin.size else None,
        "dt_used_max": float(np.max(dt_bin)) if dt_bin.size else None,
        "speed_median": float(np.median(speed[np.isfinite(speed)])) if np.any(np.isfinite(speed)) else None,
        "speed_max": float(np.max(speed[np.isfinite(speed)])) if np.any(np.isfinite(speed)) else None,
    }
    return X_true, Y_meas, meta


def split_nclt_like_paper(X_true: np.ndarray, Y: np.ndarray) -> NCLTSplits:
    """
    Paper split: 23×200 train, 2×200 val, 1×277 test (or shorter if needed).
    """
    T_train = 200
    n_train = 23
    n_val = 2
    T_test = 277

    need = n_train * T_train + n_val * T_train + T_test
    if Y.shape[0] < need:
        T_test = max(50, Y.shape[0] - (n_train + n_val) * T_train)
        need = (n_train + n_val) * T_train + T_test

    X_true = X_true[:need + 1]
    Y = Y[:need]

    def chunk(t0: int, T: int):
        return X_true[t0:t0 + T + 1], Y[t0:t0 + T]

    Xtr, Ytr, Xva, Yva = [], [], [], []
    t = 0
    for _ in range(n_train):
        Xc, Yc = chunk(t, T_train)
        Xtr.append(Xc); Ytr.append(Yc)
        t += T_train

    for _ in range(n_val):
        Xc, Yc = chunk(t, T_train)
        Xva.append(Xc); Yva.append(Yc)
        t += T_train

    Xte, Yte = chunk(t, T_test)

    return NCLTSplits(
        X_train=np.stack(Xtr, axis=0),
        Y_train=np.stack(Ytr, axis=0),
        X_val=np.stack(Xva, axis=0),
        Y_val=np.stack(Yva, axis=0),
        X_test=Xte[None, ...],
        Y_test=Yte[None, ...],
        meta={"T_train": 200, "n_train": 23, "n_val": 2, "T_test": int(T_test), "need": int(need)},
    )
