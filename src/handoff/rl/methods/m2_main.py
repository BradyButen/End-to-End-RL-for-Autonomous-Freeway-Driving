"""M2 — RL picks a target lateral offset d* and speed v* every 5 control steps.

    action [-1,1]^2 --decode--> (d*, v*) --trajectory.py--> Reference (quintic lateral + speed profile)
                                         --pid.py--------> [steering, throttle_brake] every control step
No safety layer.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.rl.env.core import CoreEnv
from handoff.rl.env.decision import DecisionWrapper, affine
from handoff.rl.methods.common import ActuationMap
from handoff.rl.methods.m2_pid import CascadedPIDController
from handoff.rl.methods.m2_trajectory import build_m2_reference


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
