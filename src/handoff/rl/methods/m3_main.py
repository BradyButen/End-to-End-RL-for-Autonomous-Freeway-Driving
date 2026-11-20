"""M3 — RL picks up to 5 Frenet waypoints and a target speed every 10 control steps; the pipeline below turns
them into controls (same pipeline as scripts/m3_harness.py).

    action [-1,1]^12 = [Δs_1, d_1, ..., Δs_5, d_5, v*, count] --decode--> k waypoints (s, d) + v*
        1. safe_set.py  project waypoints off cars / road edges (off by default: safe_set.enabled)  (per decision)
        2. spline.py    piecewise-quintic path through the waypoints, speed v*                  (per decision)
        3. corridor.py  road-aligned corridor around the initial trajectory (ramp to v*)        (every step)
        4. mpc.py       hard-constrained MPC in the tube + fallback chain; v* is also the MPC's speed ceiling
           gate: if a car in the planned path (line ego -> next waypoint crosses its keep-out zone) is being
           closed on faster than the ego can stop for (Δv²/2g > ebrake_gate_decel), emergency-brake instead
           of solving (corridor.path_blocked; method.ebrake_gate). Following a lead at its speed never fires it.
    Episode end (method.ebrake_ends_episode): the gate, or an emergency brake after an INFEASIBLE MPC, ends the
    episode as "emergency_brake" (collision penalty). Solver timeouts / numerical failures never do.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.rl.env.core import CoreEnv
from handoff.rl.env.decision import DecisionWrapper, affine
from handoff.rl.methods.common import ActuationMap
from handoff.rl.methods.m3_corridor import path_blocked
from handoff.rl.methods.m3_mpc import MPCController
from handoff.rl.methods.m3_safe_set import ProjectionResult, project_waypoints_detailed
from handoff.rl.methods.m3_spline import build_m3_reference, lane_hold_reference


def waypoint_count(c: float, n_slots: int = 5) -> int:
    """Count channel c ∈ [-1, 1] -> k = clip(1 + floor((c + 1)/2 · n), 1, n) (spec §7.3)."""
    c = float(np.clip(c, -1.0, 1.0))
    return int(np.clip(1 + np.floor((c + 1.0) / 2.0 * n_slots), 1, n_slots))


class MPCWrapper(DecisionWrapper):
    """M3: RL chooses up to 5 Frenet waypoints (Δs, d), a target speed v* and a count; spline; MPC."""

    def __init__(self, core: CoreEnv, cfg: dict):
        m = cfg["method"]
        super().__init__(core, m["decision_period"])
        self.m = m
        self.n_slots = int(m["n_slots"])
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2 * self.n_slots + 2,), dtype=np.float32)
        v_lo, v_hi = m.get("speed_range_km_h", [0.0, cfg["sim"]["v_cap_km_h"]])
        self.v_range = (float(v_lo) / 3.6, min(float(v_hi) / 3.6, core.v_cap))
        corr = m["mpc"]["corridor"]
        self.gate = bool(m.get("ebrake_gate", True))
        self.ebrake_ends = bool(m.get("ebrake_ends_episode", True))
        self.keep_s, self.keep_d = float(corr.get("keepout_s", 1.0)), float(corr["margin_d"])
        self.gate_decel = float(m.get("ebrake_gate_decel", 6.0))
        self.ego_length = float(cfg["sim"]["ego"]["length"])
        self.v_star = core.v_target
        self._gate_wps = np.zeros((0, 2))
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
        self.mpc.set_speed_limit(None)

    def decode(self, action) -> tuple[np.ndarray, int, float]:
        road = self.core.road
        margin = self.m["lateral_margin"]
        lo, hi = self.m["ds_range"]
        k = waypoint_count(action[-1], self.n_slots)
        ds = np.array([affine(action[2 * i], lo, hi) for i in range(self.n_slots)])
        d = np.array([affine(action[2 * i + 1], road.d_min - margin, road.d_max + margin)
                      for i in range(self.n_slots)])
        s = self.core.ego.s + np.cumsum(ds)
        v_star = affine(action[2 * self.n_slots], *self.v_range)
        return np.stack([s, d], axis=1), k, v_star

    def encode(self, ds: list[float], d: list[float], v: float | None = None) -> np.ndarray:
        """Inverse of ``decode`` (for scripted / hand-placed waypoints): Δs and d per slot (+ speed v in m/s,
        default the global target) -> action."""
        road = self.core.road
        margin = self.m["lateral_margin"]
        lo, hi = self.m["ds_range"]
        k = len(ds)
        a = np.zeros(2 * self.n_slots + 2)
        for i in range(self.n_slots):
            dsi = ds[min(i, k - 1)]
            di = d[min(i, k - 1)]
            a[2 * i] = 2 * (dsi - lo) / (hi - lo) - 1
            a[2 * i + 1] = 2 * (di - (road.d_min - margin)) / ((road.d_max + margin) - (road.d_min - margin)) - 1
        v = self.core.v_target if v is None else float(v)
        a[2 * self.n_slots] = 2 * (v - self.v_range[0]) / (self.v_range[1] - self.v_range[0]) - 1
        a[-1] = 2 * (k - 0.5) / self.n_slots - 1
        return np.clip(a, -1, 1)

    def decide(self, action):
        wps_all, k, v_star = self.decode(action)
        self.v_star = v_star
        self.mpc.set_speed_limit(v_star)
        raw = wps_all[:k]
        if self.safe_cfg.get("enabled", True):
            res = project_waypoints_detailed(raw, self.core.ego, self.core.objects, self.core.road, self.safe_cfg,
                                             self.core.v_cap)
        else:  # diagnostics only (spec §17.4: MPC standalone without projection)
            res = ProjectionResult(projected=raw.copy(), distances=np.zeros(len(raw)),
                                   valid=np.ones(len(raw), bool), boxes=[[] for _ in raw], all_infeasible=False)
        self.last_projection = res
        if res.all_infeasible:
            self.reference = lane_hold_reference(self.core.ego, self.core.road, v_star, ds=self.m["spline"]["ds"])
        else:
            self.reference = build_m3_reference(self.core.ego, self.core.road, res.projected[res.valid],
                                                v_star, ds=self.m["spline"]["ds"],
                                                extend=self.m["spline"]["extend"])
        self._gate_wps = res.projected[res.valid] if not res.all_infeasible else np.zeros((0, 2))
        vec = np.concatenate([wps_all.ravel(), [v_star, k]])
        return {"vector": vec, "k": k, "v_target": v_star, "waypoints_raw": raw, "waypoints_projected": res.projected,
                "projection_distances": res.distances, "valid": res.valid, "all_infeasible": res.all_infeasible,
                "boxes": res.boxes}

    def control(self, j):
        ego, objects = self.core.ego, self.core.objects
        blocked = self.gate and path_blocked(ego, self._gate_wps, objects, self.keep_s, self.keep_d,
                                             road=self.core.road, max_decel=self.gate_decel,
                                             ego_length=self.ego_length) is not None
        if blocked:
            u, diag = self.mpc.gated_brake(ego, self.reference)
        else:
            u, diag = self.mpc.step(ego, self.reference, objects)
        if self.ebrake_ends and (diag.get("gated_brake") or diag.get("fallback_terminal")):
            diag["ebrake_terminal"] = True       # CoreEnv ends the episode after this step (collision penalty)
        return u, diag
