"""M1 (end-to-end RL) and B0 (rule-based baseline).

M1: the RL action is [steering, throttle_brake], applied every control step. No controller.
B0: MetaDrive's stock IDMPolicy drives the ego at the 70 km/h target speed (car following + its own lane changes);
    the action is ignored. Evaluation only. Its ``cmd_*`` log columns are the dummy [0, 0]; IDM's real commands
    are the ego's ``steering`` / ``throttle_brake``.
"""

from __future__ import annotations

import numpy as np

from handoff.env import Method


class M1(Method):
    def __init__(self, cfg: dict, env):
        super().__init__(env, period=cfg["method"].get("decision_period", 1), action_dim=2)
        self.u = np.zeros(2)

    def decide(self, action):
        self.u = np.clip(action, -1.0, 1.0)
        return {"vector": self.u.copy(), "k": 0}

    def control(self):
        return self.u, {}


class B0(M1):
    @staticmethod
    def agent_policy():
        from metadrive.policy.idm_policy import IDMPolicy

        return IDMPolicy

    def reset(self):
        super().reset()
        md = self.env.md
        pol = md.engine.get_policy(md.agent.id)
        if pol is not None:
            pol.NORMAL_SPEED = pol.target_speed = self.env.v_target * 3.6

    def decide(self, action):
        return super().decide(np.zeros(2))
