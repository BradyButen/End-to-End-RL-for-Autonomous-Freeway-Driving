"""What gets written to disk: run folders, the per-episode recorder, and episode metrics.

    runs/<method>/<reward>/seed_<n>/<timestamp>/
        config.yaml  git_commit.txt  device.txt  train_progress.csv  episodes.csv  checkpoints/  episodes/<id>.npz

* Every episode -> one row in ``episodes.csv`` (termination, return and per-term returns, speed, jerk, gaps,
  solver stats, ...), returned by the env in ``info["episode_summary"]``.
* Every eval episode and every Nth training episode (``logging.train_full_log_every``) -> ``episodes/<id>.npz``:
  the ego state, controller scalars and raw reward terms at every control step, plus the episode summary. Enough to
  re-simulate the episode (``view.py replay``) or turn a human drive into training data (``build_demos.py``).
  Load with ``load_episode``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import subprocess
from pathlib import Path

import numpy as np

from handoff.config import REPO_ROOT, save_config
from handoff.reward import TERMS, crashed
from handoff.sim import EGO_FIELDS, EgoState

TERMINATIONS = ("success", "collision", "off_road", "timeout", "other")
STEP_SCALARS = ("reward", "tracking_error", "solve_time", "fallback", "min_gap", "min_ttc")


def termination_reason(info: dict, truncated: bool) -> str:
    """Priority: success > collision > off_road > timeout."""
    if info.get("arrive_dest"):
        return "success"
    if crashed(info):
        return "collision"
    if info.get("out_of_road"):
        return "off_road"
    if truncated or info.get("max_step"):
        return "timeout"
    return "other"


# ======================================================================================= recorder
class EpisodeRecorder:
    def __init__(self, out_dir: str | Path | None, dt: float, max_steering_rad: float):
        self.out_dir = Path(out_dir) if out_dir else None
        self.dt = dt
        self.max_steering_rad = max_steering_rad

    def start(self, episode_id: str, seed: int, full: bool, meta: dict) -> None:
        self.episode_id, self.seed, self.meta = episode_id, seed, dict(meta)
        self.full = full and self.out_dir is not None
        self.steps: dict[str, list] = {k: [] for k in (*EGO_FIELDS, *STEP_SCALARS)}
        self.terms: dict[str, list] = {n: [] for n in TERMS}
        self.weighted_sums = {n: 0.0 for n in TERMS}

    def record_step(self, ego: EgoState, raw: dict, weighted: dict, scalars: dict) -> None:
        for f in EGO_FIELDS:
            self.steps[f].append(getattr(ego, f))
        for k in STEP_SCALARS:
            val = scalars.get(k)
            self.steps[k].append(np.nan if val is None else float(val))
        for n in TERMS:
            self.terms[n].append(raw[n])
            self.weighted_sums[n] += weighted[n]

    def finish(self, reason: str, decisions: int, counters: dict) -> dict:
        steps = {k: np.asarray(v) for k, v in self.steps.items()}
        summary = {"episode_id": self.episode_id, "seed": self.seed, **self.meta,
                   **episode_metrics(steps, reason, self.dt, self.max_steering_rad)}
        for n in TERMS:
            summary[f"return_{n}"] = float(self.weighted_sums[n])
            summary[f"raw_{n}"] = float(np.sum(self.terms[n]))
        summary["lane_changes"] = int(summary["raw_lane_change"])
        summary["return"] = float(np.sum(steps["reward"]))
        summary.update(counters)
        summary["control_steps"] = len(steps["t"])
        summary["decisions"] = decisions
        summary["log_file"] = self._write_npz(steps, summary) if self.full else ""
        return summary

    def _write_npz(self, steps, summary) -> str:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.episode_id}.npz"
        arrays = {f"step_{k}": np.asarray(v, dtype=np.float64) for k, v in steps.items()}
        arrays.update({f"term_{n}": np.asarray(v, np.float64) for n, v in self.terms.items()})
        arrays["meta_keys"] = np.array(list(summary.keys()), dtype=str)
        arrays["meta_values"] = np.array([str(v) for v in summary.values()], dtype=str)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        return str(path)


def load_episode(path: str | Path) -> dict:
    """An episode .npz -> {"data": arrays, "meta": episode summary (strings)}."""
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    meta = dict(zip(data.pop("meta_keys").tolist(), data.pop("meta_values").tolist()))
    return {"data": data, "meta": meta}


# ======================================================================================= metrics
def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def _stat(fn, x) -> float:
    return float(fn(x)) if len(x) else float("nan")


def episode_metrics(steps: dict, reason: str, dt: float, max_steering_rad: float) -> dict:
    v, t = np.asarray(steps["v"], float), np.asarray(steps["t"], float)
    out: dict = {r: float(reason == r) for r in TERMINATIONS}
    out["termination"] = reason
    out["sim_time"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out["time_to_complete"] = out["sim_time"] if reason == "success" else float("nan")
    out["mean_speed"] = _stat(np.mean, v)
    out["distance"] = float(steps["s"][-1] - steps["s"][0]) if len(v) else 0.0
    out["min_gap"] = float(np.min(steps["min_gap"])) if len(v) else float("inf")
    out["min_ttc"] = float(np.min(steps["min_ttc"])) if len(v) else float("inf")

    acc = np.diff(v) / dt
    out["rms_long_jerk"] = _rms(np.diff(acc) / dt)
    yaw_rate = np.diff(np.unwrap(np.asarray(steps["psi"], float))) / dt
    out["rms_lat_accel"] = _rms(v[1:] * yaw_rate) if len(yaw_rate) else float("nan")
    steer_rate = np.diff(np.asarray(steps["steering"], float)) / dt
    out["rms_steer_rate"] = _rms(steer_rate)                        # normalised command / s
    out["rms_steer_rate_rad"] = _rms(steer_rate * max_steering_rad)  # wheel angle rad / s

    te = np.asarray(steps["tracking_error"], float)
    te = te[np.isfinite(te)]
    out["tracking_error_rms"] = _rms(te)
    out["tracking_error_max"] = _stat(np.max, te)
    st = np.asarray(steps["solve_time"], float)
    st = st[st > 0]
    out["solve_time_mean"] = _stat(np.mean, st)
    out["solve_time_p95"] = float(np.percentile(st, 95)) if len(st) else float("nan")
    out["solve_time_max"] = _stat(np.max, st)
    out["fallback_count"] = int(np.nansum(np.asarray(steps["fallback"], float) > 0))
    return out


# ======================================================================================= run folders
def git_description() -> str:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except Exception:  # noqa: BLE001 - no git / no commits
        return "unknown"


def make_run_dir(cfg: dict, run_dir: str | Path | None = None) -> Path:
    """Create ``run_dir`` (default runs/<method>/<reward>/seed_<n>/<timestamp>) with a config + code snapshot."""
    if run_dir is None:
        exp = cfg["experiment"]
        root = Path(exp.get("runs_root", "runs"))
        root = root if root.is_absolute() else REPO_ROOT / root
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / exp["method"] / exp["reward_variant"] / f"seed_{exp['seed']}" / stamp
    run = Path(run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "episodes").mkdir(exist_ok=True)
    save_config(cfg, run / "config.yaml")
    (run / "git_commit.txt").write_text(git_description() + "\n")
    return run


class CsvAppender:
    """Append dict rows to a CSV; the header is fixed by the first row (later unknown keys are dropped)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fields: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path) as f:
                self.fields = next(csv.reader(f))

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.fields is None:
            self.fields = list(rows[0].keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, self.fields, restval="", extrasaction="ignore")
            w.writerows(rows)
