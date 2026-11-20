"""B0 — rule-based baseline: MetaDrive's stock IDMPolicy drives the ego; the RL action is ignored (evaluation only).

The ego's IDM cruises at the global target speed (sim.target_speed_km_h, 70 km/h), set on its policy instance at
every reset; traffic IDM cruises at scenario.traffic.speed_km_h. IDM overtakes slower traffic by itself.

Runs through the same CoreEnv as M1–M3 so scenarios, rewards, metrics and logs are identical.
Note: ``cmd_*`` log columns are the dummy [0, 0]; IDM's actual commands are in ``applied_*``.
"""

from __future__ import annotations

import numpy as np
from metadrive.policy.idm_policy import IDMPolicy

from handoff.rl.methods.m1_main import M1Wrapper


class B0Wrapper(M1Wrapper):
    """Same interface as M1 (one control step per decision), but MetaDrive's IDMPolicy produces the controls."""

    agent_policy = IDMPolicy

    def on_reset(self):
        super().on_reset()
        md = self.core.md
        pol = md.engine.get_policy(md.agent.id)
        if pol is not None:
            v = self.core.v_target * 3.6
            pol.NORMAL_SPEED = v
            pol.target_speed = v

    def decide(self, action):
        self._u = np.zeros(2)
        return {"vector": self._u.copy(), "k": 0}
