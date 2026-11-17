"""Safe-set projection of M3 waypoints (spec §7.3), once per decision, before the MPC.

Safe set for waypoint i (Frenet (s, d)):
  * drivable: d ∈ [d_min + w/2 + edge_margin, d_max − w/2 − edge_margin] (ego half-width accounted for);
  * collision-free: outside every object's inflated, forward-predicted footprint. An object's footprint
    is an s/d box around its constant-velocity prediction at the ego's estimated arrival time
    t_i = (s_i − s_ego) / max(v_ego, v_min), swept over [t_i − τ, t_i + τ] and inflated by half the ego
    size plus margins. (t_i uses the raw waypoint's s; it is not re-evaluated after projection.)
  * reachable: s_i ≥ s_{i−1} + min_ds; s_i ≤ s_ego + v_cap·reach_time; and laterally
        |d_i − d_{i−1}| ≤ min(max_slope·Δs,  a_lat_max·Δs² / (5.77·v²)),   Δs = s_i − s_{i−1}
    The second term is exact for the spline used (spec §7.3 reference): a quintic segment with zero end
    slope/curvature has peak d'' = 5.77·Δd/Δs², i.e. peak lateral acceleration 5.77·Δd·v²/Δs² at speed v
    (v = max(v_ego, min_speed_for_timing)).

Each waypoint outside the set is replaced by the nearest point in it (Euclidean in (s, d)), searched over
the clipped point and projections onto footprint edges/corners. If no safe point exists (spec §19), that
waypoint and all later ones are dropped and each dropped waypoint is charged
``infeasible_penalty_distance`` as its projection distance (DECISIONS D-06).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handoff.sim.road import RoadFrame
from handoff.sim.state import EgoState, ObjectList, frenet_velocity

_EPS = 1e-3


@dataclass
class ProjectionResult:
    projected: np.ndarray          # (k, 2); rows of dropped waypoints are NaN
    distances: np.ndarray          # (k,) projection distance per waypoint (penalty value when dropped)
    valid: np.ndarray              # (k,) bool
    boxes: list                    # [(s_lo, s_hi, d_lo, d_hi)] footprints used (for viz/logging), per waypoint
    all_infeasible: bool


def _footprints(ego: EgoState, objects: ObjectList, road: RoadFrame, t_arr: float, cfg: dict):
    tau = cfg["time_uncertainty"]
    boxes = []
    for o in objects:
        sdot, ddot = frenet_velocity(o, road)
        half_l = 0.5 * (o.length + ego.length) + cfg["obj_margin_s"]
        half_w = 0.5 * (o.width + ego.width) + cfg["obj_margin_d"]
        s_a = o.s + sdot * max(t_arr - tau, 0.0)
        s_b = o.s + sdot * (t_arr + tau)
        d_c = o.d + ddot * t_arr
        boxes.append((min(s_a, s_b) - half_l, max(s_a, s_b) + half_l, d_c - half_w, d_c + half_w))
    return boxes


def _inside(p, box) -> bool:
    return box[0] < p[0] < box[1] and box[2] < p[1] < box[3]


def project_waypoints_detailed(waypoints: np.ndarray, ego: EgoState, objects: ObjectList, road: RoadFrame,
                               cfg: dict, v_cap: float) -> ProjectionResult:
    wps = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    k = len(wps)
    projected = np.full((k, 2), np.nan)
    dist = np.zeros(k)
    valid = np.zeros(k, dtype=bool)
    all_boxes = []
    d_lo_lane = road.d_min + 0.5 * ego.width + cfg["edge_margin"]
    d_hi_lane = road.d_max - 0.5 * ego.width - cfg["edge_margin"]
    s_max = ego.s + v_cap * cfg["reach_time"]
    slope = cfg["max_slope"]
    v_t = max(ego.v, cfg["min_speed_for_timing"])
    a_lat = cfg.get("max_lat_accel")
    curv = (a_lat / (5.77 * v_t**2)) if a_lat else np.inf

    def reach(ds: float) -> float:
        return min(slope * ds, curv * ds * ds)
    prev = np.array([ego.s, ego.d])
    failed = False
    for i in range(k):
        if failed:
            dist[i] = cfg["infeasible_penalty_distance"]
            all_boxes.append([])
            continue
        raw = wps[i]
        t_arr = max(raw[0] - ego.s, 0.0) / v_t
        boxes = _footprints(ego, objects, road, t_arr, cfg)
        all_boxes.append(boxes)
        s_lo = prev[0] + cfg["min_ds"]

        def d_interval(s, prev=prev):
            r = reach(max(s - prev[0], 0.0))
            lo = max(d_lo_lane, prev[1] - r)
            hi = min(d_hi_lane, prev[1] + r)
            if lo > hi:  # cone and lane disjoint (ego currently outside the lane bounds): lanes win
                return d_lo_lane, d_hi_lane
            return lo, hi

        def clip(p, s_lo=s_lo):
            s = float(np.clip(p[0], s_lo, max(s_max, s_lo)))
            lo, hi = d_interval(s)
            return np.array([s, float(np.clip(p[1], lo, hi))])

        def feasible(p, s_lo=s_lo, boxes=boxes):
            if p[0] < s_lo - 1e-6 or p[0] > max(s_max, s_lo) + 1e-6:
                return False
            lo, hi = d_interval(p[0])
            if p[1] < lo - 1e-6 or p[1] > hi + 1e-6:
                return False
            return not any(_inside(p, b) for b in boxes)

        cands = [clip(raw)]
        frontier = list(cands)
        for _depth in range(3):
            new = []
            for p in frontier:
                for b in boxes:
                    if not _inside(p, b):
                        continue
                    for q in ((b[0] - _EPS, p[1]), (b[1] + _EPS, p[1]), (p[0], b[2] - _EPS), (p[0], b[3] + _EPS),
                              (b[0] - _EPS, b[2] - _EPS), (b[0] - _EPS, b[3] + _EPS),
                              (b[1] + _EPS, b[2] - _EPS), (b[1] + _EPS, b[3] + _EPS)):
                        new.append(clip(np.array(q)))
            if not new:
                break
            cands.extend(new)
            frontier = new
        good = [c for c in cands if feasible(c)]
        if not good:
            failed = True
            dist[i] = cfg["infeasible_penalty_distance"]
            continue
        best = min(good, key=lambda c: float(np.hypot(*(c - raw))))
        projected[i] = best
        dist[i] = float(np.hypot(*(best - raw)))
        valid[i] = True
        prev = best
    return ProjectionResult(projected=projected, distances=dist, valid=valid, boxes=all_boxes,
                            all_infeasible=not valid.any())


def project_waypoints(waypoints: np.ndarray, ego: EgoState, objects: ObjectList, road: RoadFrame,
                      cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Spec §16 signature. ``cfg`` is the method.safe_set dict plus ``v_cap`` (m/s).
    Returns (projected_waypoints, projection_distances)."""
    res = project_waypoints_detailed(waypoints, ego, objects, road, cfg, cfg["v_cap"])
    return res.projected, res.distances
