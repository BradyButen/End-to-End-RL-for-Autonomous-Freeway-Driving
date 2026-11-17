"""Analysis plots (spec §13 — first cut): learning curves, metric comparison, episode time series."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from handoff.rl.records.recorder import load_episode  # noqa: E402

X_AXES = {"control_steps": "train_control_steps", "decisions": "train_decisions", "sim_time": "train_sim_time"}


def load_eval(run_dirs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        f = Path(rd) / "episodes.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["is_eval"] == 1].copy()
        df["run_dir"] = str(rd)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def learning_curves(run_dirs, metric: str = "success", x: str = "control_steps", out: str | Path = "curves.png"):
    """Mean ± std over seeds of an eval metric vs. training progress, one line per (method, reward variant)."""
    df = load_eval(run_dirs)
    if df.empty:
        raise ValueError("no evaluation rows found")
    xcol = X_AXES[x]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, variant), g in df.groupby(["method", "reward_variant"]):
        per_run = g.groupby(["run_dir", xcol])[metric].mean().reset_index()
        piv = per_run.pivot(index=xcol, columns="run_dir", values=metric).sort_index()
        mean, std = piv.mean(axis=1), piv.std(axis=1).fillna(0)
        ax.plot(piv.index, mean, label=f"{method}/{variant} (n={piv.shape[1]})")
        ax.fill_between(piv.index, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def comparison_table(csv_files, metrics=None) -> pd.DataFrame:
    """Mean ± std of final-evaluation metrics per (method, reward variant). Accepts evaluate.py outputs."""
    metrics = metrics or ["success", "collision", "off_road", "timeout", "time_to_complete", "mean_speed",
                          "min_gap", "min_ttc", "rms_long_jerk", "rms_lat_accel", "rms_steer_rate",
                          "tracking_error_rms", "solve_time_mean", "fallback_count", "lane_changes"]
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(["method", "reward_variant"])[metrics]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


def episode_timeseries(npz_path, out: str | Path = "episode.png"):
    ep = load_episode(npz_path)
    d = ep["data"]
    t = d["step_t"]
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, d["step_v"] * 3.6, label="v")
    if np.isfinite(d["step_ref_v"]).any():
        axes[0].plot(t, d["step_ref_v"] * 3.6, "--", label="v_ref")
    axes[0].set_ylabel("speed (km/h)")
    axes[0].legend()
    axes[1].plot(t, d["step_d"], label="d")
    if np.isfinite(d["step_ref_d"]).any():
        axes[1].plot(t, d["step_ref_d"], "--", label="d_ref")
    axes[1].set_ylabel("lateral d (m)")
    axes[1].legend()
    axes[2].plot(t, d["step_applied_steering"], label="steering")
    axes[2].plot(t, d["step_applied_throttle"], label="throttle_brake")
    axes[2].set_ylabel("command")
    axes[2].legend()
    axes[3].plot(t, np.minimum(d["step_min_gap"], 50), label="min gap (m)")
    axes[3].set_ylabel("gap (m)")
    axes[3].set_xlabel("t (s)")
    fig.suptitle(f"{ep['meta'].get('method')} seed {ep['meta'].get('seed')} — {ep['meta'].get('termination')}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
