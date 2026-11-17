"""B0 — rule-based baseline: MetaDrive's stock IDMPolicy drives the ego; the RL action is ignored (evaluation only).

Stock IDM cruises at its built-in 30 km/h (IDMPolicy.NORMAL_SPEED), the same as the traffic.

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

    def decide(self, action):
        self._u = np.zeros(2)
        return {"vector": self._u.copy(), "k": 0}
