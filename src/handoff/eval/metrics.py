"""Episode metrics (spec §11.2), computed from per-control-step arrays. Pure numpy."""

from __future__ import annotations

import numpy as np

TERMINATION_REASONS = ("success", "collision", "off_road", "timeout", "other")


def termination_reason(info: dict, truncated: bool) -> str:
    if info.get("arrive_dest", False):
        return "success"
    if info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("crash_human", False):
        return "collision"
    if info.get("out_of_road", False):
        return "off_road"
    if truncated or info.get("max_step", False):
        return "timeout"
    return "other"


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def count_lane_changes(d: np.ndarray, lane_centers: list[float], lane_width: float, settle: float = 0.3) -> int:
    """Lane changes with hysteresis: a change counts once the ego is within settle·lane_width of a new
    lane's centre."""
    if len(d) == 0:
        return 0
    centers = np.asarray(lane_centers)
    stable = int(np.argmin(np.abs(centers - d[0])))
    n = 0
    for di in d[1:]:
        j = int(np.argmin(np.abs(centers - di)))
        if j != stable and abs(di - centers[j]) < settle * lane_width:
            n += abs(j - stable)
            stable = j
    return n


def episode_metrics(steps: dict, decisions: dict, reason: str, dt: float, lane_centers: list[float],
                    lane_width: float, max_steering_rad: float) -> dict:
    v = np.asarray(steps["v"], float)
    t = np.asarray(steps["t"], float)
    out: dict = {r: float(reason == r) for r in TERMINATION_REASONS}
    out["termination"] = reason
    sim_time = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out["sim_time"] = sim_time
    out["time_to_complete"] = sim_time if reason == "success" else float("nan")
    out["mean_speed"] = float(np.mean(v)) if len(v) else float("nan")
    out["distance"] = float(steps["s"][-1] - steps["s"][0]) if len(v) else 0.0

    gaps = np.asarray(steps["min_gap"], float)
    ttcs = np.asarray(steps["min_ttc"], float)
    out["min_gap"] = float(np.min(gaps)) if len(gaps) else float("inf")
    out["min_ttc"] = float(np.min(ttcs)) if len(ttcs) else float("inf")

    acc = np.diff(v) / dt if len(v) > 1 else np.array([])
    jerk = np.diff(acc) / dt if len(acc) > 1 else np.array([])
    out["rms_long_jerk"] = _rms(jerk)
    psi = np.unwrap(np.asarray(steps["psi"], float))
    yaw_rate = np.diff(psi) / dt if len(psi) > 1 else np.array([])
    out["rms_lat_accel"] = _rms(v[1:] * yaw_rate) if len(yaw_rate) else float("nan")
    steer = np.asarray(steps["steering"], float)
    sr = np.diff(steer) / dt if len(steer) > 1 else np.array([])
    out["rms_steer_rate"] = _rms(sr)                       # normalised command / s
    out["rms_steer_rate_rad"] = _rms(sr * max_steering_rad)  # wheel angle rad / s

    te = np.asarray(steps.get("tracking_error", []), float)
    te = te[np.isfinite(te)]
    out["tracking_error_rms"] = _rms(te) if len(te) else float("nan")
    out["tracking_error_max"] = float(np.max(te)) if len(te) else float("nan")

    st = np.asarray(steps.get("solve_time", []), float)
    st = st[st > 0]
    out["solve_time_mean"] = float(np.mean(st)) if len(st) else float("nan")
    out["solve_time_p95"] = float(np.percentile(st, 95)) if len(st) else float("nan")
    out["solve_time_max"] = float(np.max(st)) if len(st) else float("nan")
    out["fallback_count"] = int(np.sum(np.asarray(steps.get("fallback", []), bool)))

    pd = np.asarray(decisions.get("projection_distance_sum", []), float)
    out["projection_distance_mean"] = float(np.mean(pd)) if len(pd) else float("nan")
    out["projection_distance_total"] = float(np.sum(pd)) if len(pd) else float("nan")
    ks = np.asarray(decisions.get("k", []), int)
    for k in range(1, 6):
        out[f"k{k}_count"] = int(np.sum(ks == k))

    out["lane_changes"] = count_lane_changes(np.asarray(steps["d"], float), lane_centers, lane_width)
    return out
