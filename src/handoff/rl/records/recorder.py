"""Custom three-level logging (spec §12): control step (10 Hz), decision step, episode.

* Episode-level summaries are produced for every episode (returned in ``info["episode_summary"]`` and
  written to ``episodes.csv`` by the trainer / evaluator).
* Full control- and decision-level logs (``episodes/<id>.npz``) only for episodes flagged ``full``
  (all evaluation episodes, every Nth training episode).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from handoff.rl.env.rewards import TERM_NAMES
from handoff.rl.eval.metrics import episode_metrics
from handoff.simulator.state import EGO_FIELDS, OBJECT_FIELDS, EgoState, pack_objects

STEP_SCALARS = ("tracking_error", "solve_time", "fallback", "min_gap", "min_ttc", "ref_d", "ref_v", "ref_psi",
                "applied_steering", "applied_throttle", "cmd_steering", "cmd_throttle", "decision_index",
                "tube_rebuilt", "tube_blocked", "gated_brake", "reward")


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path | None, max_objects: int, dt: float, lane_centers_fn,
                 lane_width: float, max_steering_rad: float):
        self.out_dir = Path(out_dir) if out_dir else None
        self.max_objects = max_objects
        self.dt = dt
        self._lane_centers_fn = lane_centers_fn
        self.lane_width = lane_width
        self.max_steering_rad = max_steering_rad
        self.active = False

    # ------------------------------------------------------------------ lifecycle
    def start(self, episode_id: str, seed: int, full: bool, meta: dict) -> None:
        self.episode_id = episode_id
        self.seed = seed
        self.full = full and self.out_dir is not None
        self.meta = dict(meta)
        self.steps: dict[str, list] = {f: [] for f in EGO_FIELDS}
        for k in STEP_SCALARS:
            self.steps[k] = []
        self.step_terms: dict[str, list] = {n: [] for n in TERM_NAMES}
        self.weighted_sums: dict[str, float] = {n: 0.0 for n in TERM_NAMES}
        self.objects: list[np.ndarray] = []
        self.object_ids: list[list[str]] = []
        self.pred_paths: list[np.ndarray] = []
        self.status: list[str] = []
        self.dec: dict[str, list] = {"k": [], "projection_distance_sum": [], "control_steps": [], "reward": [],
                                     "start_step": []}
        self.dec_arrays: dict[str, list] = {"raw_action": [], "decoded": [], "waypoints_raw": [],
                                            "waypoints_projected": [], "projection_distances": []}
        self.references: list[np.ndarray] = []
        self.tube_steps: list[int] = []
        self.tubes: list[np.ndarray] = []
        self.tube_trajs: list[np.ndarray] = []
        self.active = True

    def record_control(self, ego: EgoState, objects, terms_raw: dict, terms_weighted: dict, reward: float,
                       extra: dict, pred_path=None, status: str = "", tube=None) -> None:
        for f in EGO_FIELDS:
            self.steps[f].append(getattr(ego, f))
        for k in STEP_SCALARS:
            val = extra.get(k, np.nan)
            self.steps[k].append(float(val) if val is not None else np.nan)
        self.steps["reward"][-1] = reward
        for n in TERM_NAMES:
            self.step_terms[n].append(terms_raw.get(n, 0.0))
            self.weighted_sums[n] += terms_weighted.get(n, 0.0)
        self.status.append(status)
        if self.full and tube is not None:  # corridor rebuilt this step: keep corners + initial trajectory
            self.tube_steps.append(len(self.steps["t"]) - 1)
            self.tubes.append(np.asarray(tube.corners, np.float32))
            self.tube_trajs.append(np.asarray(tube.traj, np.float32))
        if self.full:
            self.objects.append(pack_objects(objects, self.max_objects))
            self.object_ids.append([o.id for o in objects[: self.max_objects]])
            self.pred_paths.append(np.asarray(pred_path, np.float32) if pred_path is not None
                                   else np.zeros((0, 2), np.float32))

    def record_decision(self, raw_action, decoded: dict, reward: float, control_steps: int, start_step: int,
                        reference=None) -> None:
        self.dec["k"].append(int(decoded.get("k", 0)))
        pd = decoded.get("projection_distances")
        self.dec["projection_distance_sum"].append(float(np.sum(pd)) if pd is not None else np.nan)
        self.dec["control_steps"].append(control_steps)
        self.dec["reward"].append(reward)
        self.dec["start_step"].append(start_step)
        if self.full:
            self.dec_arrays["raw_action"].append(np.asarray(raw_action, np.float32).ravel())
            self.dec_arrays["decoded"].append(np.asarray(decoded.get("vector", []), np.float32).ravel())
            self.dec_arrays["waypoints_raw"].append(_pad(decoded.get("waypoints_raw"), 5))
            self.dec_arrays["waypoints_projected"].append(_pad(decoded.get("waypoints_projected"), 5))
            self.dec_arrays["projection_distances"].append(_pad1(pd, 5))
            self.references.append(reference.as_array() if reference is not None else np.zeros((0, 7), np.float32))

    def finish(self, reason: str, counters: dict) -> dict:
        self.active = False
        steps = {k: np.asarray(v) for k, v in self.steps.items()}
        decs = {k: np.asarray(v) for k, v in self.dec.items()}
        metrics = episode_metrics(steps, decs, reason, self.dt, self._lane_centers_fn(), self.lane_width,
                                  self.max_steering_rad)
        summary = {"episode_id": self.episode_id, "seed": self.seed, **self.meta, **metrics}
        for n in TERM_NAMES:
            summary[f"return_{n}"] = float(self.weighted_sums[n])                    # weighted
            summary[f"raw_{n}"] = float(np.sum(self.step_terms[n])) if self.step_terms[n] else 0.0
        summary["return"] = float(np.sum(steps["reward"])) if len(steps["reward"]) else 0.0
        summary.update(counters)
        summary["control_steps"] = len(steps["t"])
        summary["decisions"] = len(decs["k"])
        if self.full:
            summary["log_file"] = self._write_npz(steps, decs, summary)
        else:
            summary["log_file"] = ""
        return summary

    def _write_npz(self, steps, decs, summary) -> str:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.episode_id}.npz"
        arrays = {f"step_{k}": np.asarray(v, dtype=np.float64) for k, v in steps.items()}
        arrays.update({f"term_{n}": np.asarray(v, np.float64) for n, v in self.step_terms.items()})
        arrays.update({f"dec_{k}": np.asarray(v) for k, v in decs.items()})
        for k, v in self.dec_arrays.items():
            arrays[f"dec_{k}"] = _stack_ragged(v)
        arrays["objects"] = np.stack(self.objects) if self.objects else np.zeros((0, self.max_objects, 10))
        arrays["object_fields"] = np.array(OBJECT_FIELDS)
        arrays["object_ids"] = np.array([ids + [""] * (self.max_objects - len(ids)) for ids in self.object_ids]
                                        or np.zeros((0, self.max_objects)), dtype=object).astype(str)
        arrays["pred_paths"] = _stack_ragged(self.pred_paths)
        arrays["status"] = np.array(self.status, dtype=str)
        arrays["references"] = _stack_ragged(self.references)
        arrays["tube_steps"] = np.asarray(self.tube_steps, np.int64)
        arrays["tubes"] = _stack_ragged(self.tubes)              # (n_rebuilds, K+1, 4, 2) box corners
        arrays["tube_trajs"] = _stack_ragged(self.tube_trajs)    # (n_rebuilds, K+1, 6) [s, d, v, x, y, psi]
        arrays["term_names"] = np.array(TERM_NAMES)
        arrays["meta_keys"] = np.array(list(summary.keys()), dtype=str)
        arrays["meta_values"] = np.array([str(v) for v in summary.values()], dtype=str)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        return str(path)


def _pad(arr, n):
    out = np.full((n, 2), np.nan, np.float32)
    if arr is not None:
        a = np.asarray(arr, np.float32).reshape(-1, 2)[:n]
        out[: len(a)] = a
    return out


def _pad1(arr, n):
    out = np.full(n, np.nan, np.float32)
    if arr is not None:
        a = np.asarray(arr, np.float32).ravel()[:n]
        out[: len(a)] = a
    return out


def _stack_ragged(items: list[np.ndarray]) -> np.ndarray:
    """Stack arrays with varying first dimension into a NaN-padded array."""
    if not items:
        return np.zeros((0,), np.float32)
    items = [np.asarray(i, np.float32) for i in items]
    if items[0].ndim == 0:
        return np.asarray(items, np.float32)
    max_len = max(i.shape[0] for i in items)
    tail = items[0].shape[1:]
    out = np.full((len(items), max_len, *tail), np.nan, np.float32)
    for j, it in enumerate(items):
        if it.size:
            out[j, : it.shape[0]] = it
    return out


def load_episode(path: str | Path) -> dict:
    """Load an episode npz back into nested dicts (for replay / plots)."""
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    meta = dict(zip(data.pop("meta_keys").tolist(), data.pop("meta_values").tolist()))
    return {"data": data, "meta": meta}
