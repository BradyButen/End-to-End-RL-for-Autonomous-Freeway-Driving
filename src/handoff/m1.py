"""M1 (end-to-end RL) and B0 (rule-based baseline).

M1: the RL action is [steering, longitudinal accel], applied every control step. No controller. The accel channel
    is a target acceleration (0 = hold speed), mapped to MetaDrive's throttle_brake through the measured actuation
    table: raw throttle_brake is ~5x stronger on the brake side (-0.5 -> about -10 m/s^2, +0.5 -> +1.5), so
    zero-mean exploration noise on it nets out to hard braking and the car crawls.
B0: MetaDrive's stock IDMPolicy drives the ego at the 70 km/h target speed (car following + its own lane changes);
    the action is ignored. Evaluation only. Its ``cmd_*`` log columns are the dummy [0, 0]; IDM's real commands
    are the ego's ``steering`` / ``throttle_brake``.
"""

from __future__ import annotations

import numpy as np

from handoff.env import Method
from handoff.sim import ActuationMap


class M1(Method):
    def __init__(self, cfg: dict, env):
        super().__init__(env, period=cfg["method"].get("decision_period", 1), action_dim=2)
        rng = cfg["method"].get("accel_range")        # None: raw throttle_brake (runs from before the re-map)
        self.a_min, self.a_max = (float(x) for x in rng) if rng is not None else (None, None)
        self.actuation = ActuationMap(cfg["actuation"]["table"])
        self.u = np.zeros(2)

    def decide(self, action):
        steer, x = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.a_max is None:
            self.u = np.array([steer, x])
            return {"vector": self.u.copy(), "k": 0}
        acc = x * (self.a_max if x >= 0 else -self.a_min)        # piecewise linear, 0 -> 0 m/s^2 (hold speed)
        self.u = np.array([steer, np.clip(self.actuation.accel_to_command(acc), -1.0, 1.0)])
        return {"vector": np.array([steer, acc]), "k": 0}

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
        self.u = np.zeros(2)          # ignored: IDMPolicy drives
        return {"vector": self.u.copy(), "k": 0}
