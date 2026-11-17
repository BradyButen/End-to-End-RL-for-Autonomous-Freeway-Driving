"""DecisionWrapper: the loop shared by every method (spec §3.1, §8). One RL action → up to N control steps.

    step(action):
        decision = decide(action)                          # method-specific: build the reference
        for j in range(N):
            u = control(j)                                 # method-specific: controller output
            obs, r, terminated, truncated, info = core.control_step(u, ...)
            R += r                                         # undiscounted sum within a decision
            if terminated or truncated: break
        core.end_decision(...)                             # decision-level log (+ episode summary if done)

The method wrappers (``handoff/methods/<m>/wrapper.py``) subclass this and implement ``decide`` / ``control``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.env.core import CoreEnv
from handoff.methods.common.reference import Reference


def affine(a: float, lo: float, hi: float) -> float:
    """[-1, 1] -> [lo, hi]."""
    return lo + (float(a) + 1.0) * 0.5 * (hi - lo)


class DecisionWrapper(gym.Wrapper):
    period: int = 1
    agent_policy = None
    """MetaDrive ``agent_policy`` class for the ego; None = the env input (our action) drives it. B0 sets IDM."""

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
