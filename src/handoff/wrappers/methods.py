"""Method wrappers (spec §3.1, §7, §8). Each wraps a CoreEnv; one RL action → up to N control steps.

    step(action):
        decision, reference = decide(action)               # M3: + safe-set projection
        for j in range(N):
            u = controller.step(ego, reference, objects)
            obs, r, terminated, truncated, info = core.control_step(u, ...)
            R += r                                         # undiscounted sum within a decision
            if terminated or truncated: break
        core.end_decision(...)                             # decision-level log (+ episode summary if done)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.controllers.actuation import ActuationMap
from handoff.controllers.base import Reference
from handoff.controllers.mpc import MPCController
from handoff.controllers.pid import CascadedPIDController
from handoff.controllers.safe_set import ProjectionResult, project_waypoints_detailed
from handoff.controllers.spline import build_m3_reference, lane_hold_reference
from handoff.controllers.trajectory import build_m2_reference
from handoff.wrappers.core import CoreEnv


def affine(a: float, lo: float, hi: float) -> float:
    """[-1, 1] -> [lo, hi]."""
    return lo + (float(a) + 1.0) * 0.5 * (hi - lo)


def waypoint_count(c: float, n_slots: int = 5) -> int:
    """Count channel c ∈ [-1, 1] -> k = clip(1 + floor((c + 1)/2 · n), 1, n) (spec §7.3)."""
    c = float(np.clip(c, -1.0, 1.0))
    return int(np.clip(1 + np.floor((c + 1.0) / 2.0 * n_slots), 1, n_slots))


class DecisionWrapper(gym.Wrapper):
    period: int = 1

    def __init__(self, core: CoreEnv, period: int):
        super().__init__(core)
        self.core = core
        self.period = int(period)
        self.reference: Reference | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.core.reset(seed=seed, options=options)
        self.reference = None
        self.on_reset()
        return obs, info

    def on_reset(self) -> None:
        pass

    def decide(self, action: np.ndarray) -> dict:
        raise NotImplementedError

    def control(self, j: int) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        decision = self.decide(action)
        start = self.core.episode_control_steps
        ctx = {"first_step": True, "projection_distances": decision.get("projection_distances"),
               "decision": decision, "reference": self.reference}
        R, n = 0.0, 0
        obs = info = None
        terminated = truncated = False
        for j in range(self.period):
            u, diag = self.control(j)
            obs, r, terminated, truncated, info = self.core.control_step(u, decision=ctx, diag=diag)
            ctx = {**ctx, "first_step": False}
            R += r
            n += 1
            if terminated or truncated:
                break
        info = self.core.end_decision(action, decision, R, n, start, self.reference, terminated, truncated, info)
        return obs, R, terminated, truncated, info


class M1Wrapper(DecisionWrapper):
    """End-to-end: action = [steering, throttle_brake] every control step."""

    def __init__(self, core: CoreEnv, cfg: dict):
        super().__init__(core, cfg["method"].get("decision_period", 1))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._u = np.zeros(2)

    def decide(self, action):
        self._u = np.clip(action, -1.0, 1.0)
        return {"vector": self._u.copy(), "k": 0}

    def control(self, j):
        return self._u, {}


class B0Wrapper(M1Wrapper):
    """Rule-based baseline: MetaDrive's IDMPolicy drives the ego (agent_policy); the action is ignored."""

    def decide(self, action):
        self._u = np.zeros(2)
        return {"vector": self._u.copy(), "k": 0}


class PIDWrapper(DecisionWrapper):
    """M2: RL chooses (d*, v*); quintic lateral + speed profile; cascaded PID tracks it. No safety layer."""

    def __init__(self, core: CoreEnv, cfg: dict):
        m = cfg["method"]
        super().__init__(core, m["decision_period"])
        self.m = m
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.actuation = ActuationMap(cfg["actuation"]["table"])
        self.controller = CascadedPIDController.from_config(m["pid"], self.actuation, core.dt)
        self._elapsed = 0.0

    def on_reset(self):
        self.controller.reset()
        self._elapsed = 0.0

    def encode(self, d_star: float, v_star: float) -> np.ndarray:
        """Inverse of ``decode`` (for scripted actions)."""
        road = self.core.road
        margin = self.m["lateral_margin"]
        lo, hi = road.d_min - margin, road.d_max + margin
        return np.array([2 * (d_star - lo) / (hi - lo) - 1, 2 * v_star / self.core.v_cap - 1])

    def decode(self, action) -> tuple[float, float]:
        road = self.core.road
        margin = self.m["lateral_margin"]
        d_star = affine(action[0], road.d_min - margin, road.d_max + margin)   # deliberately not clipped to road
        v_star = affine(action[1], 0.0, self.core.v_cap)
        return d_star, v_star

    def decide(self, action):
        d_star, v_star = self.decode(action)
        tr = self.m["trajectory"]
        lat_init = None
        if tr.get("replan_from", "reference") == "reference" and self.reference is not None:
            r = self.reference.at_time(self._elapsed)
            lat_init = (r.get("d_dot", 0.0), r.get("d_ddot", 0.0))
        self.reference = build_m2_reference(
            self.core.ego, self.core.road, d_star, v_star, tr["horizon"], tr["dt"], tr["accel_limit"],
            tr["decel_limit"], lateral_init=lat_init, min_duration=(self.period + 1) * self.core.dt)
        self.controller.new_reference()
        self._elapsed = 0.0
        return {"vector": np.array([d_star, v_star]), "k": 0, "d_target": d_star, "v_target": v_star}

    def control(self, j):
        u, diag = self.controller.step(self.core.ego, self.reference, self.core.objects)
        self._elapsed += self.core.dt
        return u, diag


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


METHOD_WRAPPERS = {"m1": M1Wrapper, "m2": PIDWrapper, "m3": MPCWrapper, "b0": B0Wrapper}
