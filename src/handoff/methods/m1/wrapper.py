"""M1 — end-to-end: the RL action is [steering, throttle_brake], applied every control step (no controller)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.env.core import CoreEnv
from handoff.env.decision import DecisionWrapper


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
