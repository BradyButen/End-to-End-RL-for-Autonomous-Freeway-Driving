"""B0 — rule-based baseline: MetaDrive's IDM drives the ego; the RL action is ignored (evaluation only).

Runs through the same CoreEnv as M1–M3 so scenarios, rewards, metrics and logs are identical.
Note: ``cmd_*`` log columns are the dummy [0, 0]; IDM's actual commands are in ``applied_*``.
"""

from __future__ import annotations

import numpy as np

from handoff.methods.b0.idm import EgoIDMPolicy
from handoff.methods.m1.wrapper import M1Wrapper


class B0Wrapper(M1Wrapper):
    """Same interface as M1 (one control step per decision), but MetaDrive's IDMPolicy produces the controls."""

    agent_policy = EgoIDMPolicy

    def decide(self, action):
        self._u = np.zeros(2)
        return {"vector": self._u.copy(), "k": 0}
