"""M3 — RL picks up to 5 Frenet waypoints every 10 control steps; the pipeline below turns them into controls.

    action [-1,1]^11 --decode--> k waypoints (s, d)
        1. safe_set.py  project waypoints off cars / road edges, enforce reachability   (once per decision)
        2. spline.py    piecewise-quintic path through the projected waypoints          (once per decision)
        3. corridor.py  initial trajectory + convex tube (4 half-spaces per stage)      (rebuilt every 10 steps)
        4. mpc.py       hard-constrained MPC in the tube + fallback chain               (every control step)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.env.core import CoreEnv
from handoff.env.decision import DecisionWrapper, affine
from handoff.methods.common.actuation import ActuationMap
from handoff.methods.m3.mpc import MPCController
from handoff.methods.m3.safe_set import ProjectionResult, project_waypoints_detailed
from handoff.methods.m3.spline import build_m3_reference, lane_hold_reference


def waypoint_count(c: float, n_slots: int = 5) -> int:
    """Count channel c ∈ [-1, 1] -> k = clip(1 + floor((c + 1)/2 · n), 1, n) (spec §7.3)."""
    c = float(np.clip(c, -1.0, 1.0))
    return int(np.clip(1 + np.floor((c + 1.0) / 2.0 * n_slots), 1, n_slots))


class MPCWrapper(DecisionWrapper):
    """M3: RL chooses up to 5 Frenet waypoints (Δs, d) + count; safe-set projection; spline; MPC."""

    def __init__(self, core: CoreEnv, cfg: dict):
        m = cfg["method"]
        super().__init__(core, m["decision_period"])
        self.m = m
        self.n_slots = int(m["n_slots"])
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2 * self.n_slots + 1,), dtype=np.float32)
        ego = cfg["sim"]["ego"]
        self.actuation = ActuationMap(cfg["actuation"]["table"])
        self.mpc = MPCController(
            m["mpc"], None, self.actuation, lf=ego["front_wheelbase"], lr=ego["rear_wheelbase"],
            ego_length=ego["length"], ego_width=ego["width"], max_steering_rad=core.max_steering_rad,
            v_cap=core.v_cap, v_target=core.v_target)
        self.safe_cfg = dict(m["safe_set"])
        self.last_projection = None

    def on_reset(self):
        self.mpc.set_road(self.core.road)
        self.mpc.reset()

    def decode(self, action) -> tuple[np.ndarray, int]:
        road = self.core.road
        margin = self.m["lateral_margin"]
        lo, hi = self.m["ds_range"]
        k = waypoint_count(action[-1], self.n_slots)
        ds = np.array([affine(action[2 * i], lo, hi) for i in range(self.n_slots)])
        d = np.array([affine(action[2 * i + 1], road.d_min - margin, road.d_max + margin)
                      for i in range(self.n_slots)])
        s = self.core.ego.s + np.cumsum(ds)
        return np.stack([s, d], axis=1), k

    def encode(self, ds: list[float], d: list[float]) -> np.ndarray:
        """Inverse of ``decode`` (for scripted / hand-placed waypoints): Δs and d per slot -> action."""
        road = self.core.road
        margin = self.m["lateral_margin"]
        lo, hi = self.m["ds_range"]
        k = len(ds)
        a = np.zeros(2 * self.n_slots + 1)
        for i in range(self.n_slots):
            dsi = ds[min(i, k - 1)]
            di = d[min(i, k - 1)]
            a[2 * i] = 2 * (dsi - lo) / (hi - lo) - 1
            a[2 * i + 1] = 2 * (di - (road.d_min - margin)) / ((road.d_max + margin) - (road.d_min - margin)) - 1
        a[-1] = 2 * (k - 0.5) / self.n_slots - 1
        return np.clip(a, -1, 1)

    def decide(self, action):
        wps_all, k = self.decode(action)
        raw = wps_all[:k]
        if self.safe_cfg.get("enabled", True):
            res = project_waypoints_detailed(raw, self.core.ego, self.core.objects, self.core.road, self.safe_cfg,
                                             self.core.v_cap)
        else:  # diagnostics only (spec §17.4: MPC standalone without projection)
            res = ProjectionResult(projected=raw.copy(), distances=np.zeros(len(raw)),
                                   valid=np.ones(len(raw), bool), boxes=[[] for _ in raw], all_infeasible=False)
        self.last_projection = res
        if res.all_infeasible:
            self.reference = lane_hold_reference(self.core.ego, self.core.road, self.core.v_target,
                                                 ds=self.m["spline"]["ds"])
        else:
            self.reference = build_m3_reference(self.core.ego, self.core.road, res.projected[res.valid],
                                                self.core.v_target, ds=self.m["spline"]["ds"],
                                                extend=self.m["spline"]["extend"])
        vec = np.concatenate([wps_all.ravel(), [k]])
        return {"vector": vec, "k": k, "waypoints_raw": raw, "waypoints_projected": res.projected,
                "projection_distances": res.distances, "valid": res.valid, "all_infeasible": res.all_infeasible,
                "boxes": res.boxes}

    def control(self, j):
        return self.mpc.step(self.core.ego, self.reference, self.core.objects)
