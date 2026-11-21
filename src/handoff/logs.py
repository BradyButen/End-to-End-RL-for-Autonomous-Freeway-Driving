"""What gets written to disk: run folders, the per-episode recorder, and episode metrics.

    runs/<method>/<reward>/seed_<n>/<timestamp>/
        config.yaml  git_commit.txt  device.txt  train_progress.csv  episodes.csv  checkpoints/  episodes/<id>.npz

* Every episode -> one row in ``episodes.csv`` (termination, return and per-term returns, speed, jerk, gaps,
  solver stats, ...), returned by the env in ``info["episode_summary"]``.
* Every eval episode and every Nth training episode (``logging.train_full_log_every``) -> ``episodes/<id>.npz`` with
  every control step and decision (ego, cars, reward terms, controller diagnostics, references, MPC predictions;
  M3 corridor every ``logging.tube_every`` steps). Load with ``load_episode``.
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
from handoff.sim import EGO_FIELDS, OBJECT_FIELDS, EgoState, pack_objects

TERMINATIONS = ("success", "collision", "emergency_brake", "off_road", "timeout", "other")
STEP_SCALARS = ("reward", "tracking_error", "solve_time", "cpu_time", "fallback", "gated_brake", "min_gap",
                "min_ttc", "ref_d", "ref_v", "ref_psi", "cmd_steering", "cmd_throttle", "decision_index")


def termination_reason(info: dict, truncated: bool) -> str:
    """Priority: success > collision > emergency_brake (M3) > off_road > timeout."""
    if info.get("arrive_dest"):
        return "success"
    if crashed(info):
        return "collision"
    if info.get("emergency_brake"):
        return "emergency_brake"
    if info.get("out_of_road"):
        return "off_road"
    if truncated or info.get("max_step"):
        return "timeout"
    return "other"


# ======================================================================================= recorder
class EpisodeRecorder:
    def __init__(self, out_dir: str | Path | None, max_objects: int, dt: float, max_steering_rad: float):
        self.out_dir = Path(out_dir) if out_dir else None
        self.max_objects = max_objects
        self.dt = dt
        self.max_steering_rad = max_steering_rad

    def start(self, episode_id: str, seed: int, full: bool, meta: dict) -> None:
        self.episode_id, self.seed, self.meta = episode_id, seed, dict(meta)
        self.full = full and self.out_dir is not None
        self.steps: dict[str, list] = {k: [] for k in (*EGO_FIELDS, *STEP_SCALARS)}
        self.terms: dict[str, list] = {n: [] for n in TERMS}
        self.weighted_sums = {n: 0.0 for n in TERMS}
        self.status: list[str] = []
        self.objects, self.object_ids, self.pred_paths = [], [], []
        self.tube_steps, self.tubes, self.tube_trajs = [], [], []
        self.dec: dict[str, list] = {"k": [], "control_steps": [], "reward": [], "start_step": []}
        self.dec_full: dict[str, list] = {"raw_action": [], "decoded": [], "waypoints": []}
        self.references: list[np.ndarray] = []

    def record_step(self, ego: EgoState, objects, raw: dict, weighted: dict, scalars: dict, status: str,
                    pred_path=None, tube=None) -> None:
        for f in EGO_FIELDS:
            self.steps[f].append(getattr(ego, f))
        for k in STEP_SCALARS:
            val = scalars.get(k)
            self.steps[k].append(np.nan if val is None else float(val))
        for n in TERMS:
            self.terms[n].append(raw[n])
            self.weighted_sums[n] += weighted[n]
        self.status.append(status)
        if not self.full:
            return
        self.objects.append(pack_objects(objects, self.max_objects))
        self.object_ids.append([o.id for o in objects[: self.max_objects]])
        self.pred_paths.append(np.zeros((0, 2), np.float32) if pred_path is None else np.asarray(pred_path, np.float32))
        if tube is not None:
            self.tube_steps.append(len(self.steps["t"]) - 1)
            self.tubes.append(np.asarray(tube.corners, np.float32))
            self.tube_trajs.append(np.asarray(tube.traj, np.float32))

    def record_decision(self, raw_action, decision: dict, reward: float, n_steps: int, start_step: int,
                        reference=None) -> None:
        self.dec["k"].append(int(decision.get("k", 0)))
        self.dec["control_steps"].append(n_steps)
        self.dec["reward"].append(reward)
        self.dec["start_step"].append(start_step)
        if self.full:
            self.dec_full["raw_action"].append(np.asarray(raw_action, np.float32).ravel())
            self.dec_full["decoded"].append(np.asarray(decision.get("vector", []), np.float32).ravel())
            self.dec_full["waypoints"].append(_pad_rows(decision.get("waypoints"), 5))
            self.references.append(np.zeros((0, 7), np.float32) if reference is None else reference.as_array())

    def finish(self, reason: str, counters: dict) -> dict:
        steps = {k: np.asarray(v) for k, v in self.steps.items()}
        decs = {k: np.asarray(v) for k, v in self.dec.items()}
        summary = {"episode_id": self.episode_id, "seed": self.seed, **self.meta,
                   **episode_metrics(steps, decs, reason, self.dt, self.max_steering_rad)}
        for n in TERMS:
            summary[f"return_{n}"] = float(self.weighted_sums[n])
            summary[f"raw_{n}"] = float(np.sum(self.terms[n]))
        summary["lane_changes"] = int(summary["raw_lane_change"])
        summary["return"] = float(np.sum(steps["reward"]))
        summary.update(counters)
        summary["control_steps"] = len(steps["t"])
        summary["decisions"] = len(decs["k"])
        summary["log_file"] = self._write_npz(steps, decs, summary) if self.full else ""
        return summary

    def _write_npz(self, steps, decs, summary) -> str:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.episode_id}.npz"
        arrays = {f"step_{k}": np.asarray(v, dtype=np.float64) for k, v in steps.items()}
        arrays.update({f"term_{n}": np.asarray(v, np.float64) for n, v in self.terms.items()})
        arrays.update({f"dec_{k}": np.asarray(v) for k, v in decs.items()})
        arrays.update({f"dec_{k}": _stack_ragged(v) for k, v in self.dec_full.items()})
        n = self.max_objects
        arrays["objects"] = np.stack(self.objects) if self.objects else np.zeros((0, n, len(OBJECT_FIELDS)))
        arrays["object_fields"] = np.array(OBJECT_FIELDS)
        arrays["object_ids"] = np.array([ids + [""] * (n - len(ids)) for ids in self.object_ids]
                                        or np.zeros((0, n)), dtype=object).astype(str)
        arrays["pred_paths"] = _stack_ragged(self.pred_paths)
        arrays["status"] = np.array(self.status, dtype=str)
        arrays["references"] = _stack_ragged(self.references)
        arrays["tube_steps"] = np.asarray(self.tube_steps, np.int64)
        arrays["tubes"] = _stack_ragged(self.tubes)              # (n, N+1, 4, 2) corridor box corners
        arrays["tube_trajs"] = _stack_ragged(self.tube_trajs)    # (n, N+1, 6) [s, d, v, x, y, psi]
        arrays["term_names"] = np.array(TERMS)
        arrays["meta_keys"] = np.array(list(summary.keys()), dtype=str)
        arrays["meta_values"] = np.array([str(v) for v in summary.values()], dtype=str)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        return str(path)


def _pad_rows(arr, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan, np.float32)
    if arr is not None:
        a = np.asarray(arr, np.float32).reshape(-1, 2)[:n]
        out[: len(a)] = a
    return out


def _stack_ragged(items: list) -> np.ndarray:
    """Stack arrays of varying first dimension into one NaN-padded array."""
    if not items:
        return np.zeros((0,), np.float32)
    items = [np.asarray(i, np.float32) for i in items]
    max_len = max(i.shape[0] for i in items)
    out = np.full((len(items), max_len, *items[0].shape[1:]), np.nan, np.float32)
    for j, it in enumerate(items):
        if it.size:
            out[j, : it.shape[0]] = it
    return out


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


def episode_metrics(steps: dict, decisions: dict, reason: str, dt: float, max_steering_rad: float) -> dict:
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
    out["gated_brake_count"] = int(np.nansum(np.asarray(steps["gated_brake"], float) > 0))
    ks = np.asarray(decisions["k"], int)
    for k in range(1, 6):
        out[f"k{k}_count"] = int(np.sum(ks == k))
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
