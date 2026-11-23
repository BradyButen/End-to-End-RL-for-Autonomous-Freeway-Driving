"""M1 (end-to-end RL) and B0 (rule-based baseline).

M1: the RL action is [lateral accel, longitudinal accel], applied every control step. No controller. Both are in
    physical units, so the same action (and the same exploration noise) means the same thing at every speed:
    * lateral: a target lateral acceleration, +-method.steer_lat_accel (0 = straight), turned into a wheel angle for
      the current speed with the kinematic bicycle, delta = atan(L a_lat / v^2) (speeds below V_STEER_MIN use
      V_STEER_MIN). Raw steering (+-1 = +-40 deg of wheel) made nearly the whole range "leave the road" at speed:
      the warm start's noise (std 0.135 = 5.4 deg) asked for ~19 m/s^2 at 80 km/h, twice the grip.
    * accel: a target acceleration (0 = hold speed), mapped to throttle_brake through the measured actuation table:
      raw throttle_brake is ~5x stronger on the brake side, so zero-mean noise on it netted out to hard braking.
    ``steer_lat_accel`` / ``accel_range`` null = raw [steering, throttle_brake] (scripts/build_demos.py replays the
    logged human commands that way).
B0: MetaDrive's stock IDMPolicy drives the ego at the 70 km/h target speed (car following + its own lane changes);
    the action is ignored. Evaluation only. Its ``cmd_*`` log columns are the dummy [0, 0]; IDM's real commands
    are the ego's ``steering`` / ``throttle_brake``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.env import Method
from handoff.sim import ActuationMap

V_STEER_MIN = 5.0   # m/s: the steering map uses at least this speed (finite wheel angle when slow / stopped)


def lat_accel_to_steer(a_lat, v, wheelbase: float, max_steering_rad: float):
    """Lateral acceleration (m/s^2) at speed v -> normalised steering command (kinematic bicycle)."""
    v = np.maximum(np.asarray(v, dtype=float), V_STEER_MIN)
    return np.clip(np.arctan(wheelbase * np.asarray(a_lat, dtype=float) / v**2) / max_steering_rad, -1.0, 1.0)


def steer_to_lat_accel(steer, v, wheelbase: float, max_steering_rad: float):
    """Inverse of ``lat_accel_to_steer`` (human demos -> M1 labels)."""
    v = np.maximum(np.asarray(v, dtype=float), V_STEER_MIN)
    return v**2 * np.tan(np.asarray(steer, dtype=float) * max_steering_rad) / wheelbase


class M1(Method):
    def __init__(self, cfg: dict, env):
        super().__init__(env, period=cfg["method"].get("decision_period", 1),
                         action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32))
        m = cfg["method"]
        rng = m.get("accel_range")
        self.a_min, self.a_max = (float(x) for x in rng) if rng is not None else (None, None)
        lat = m.get("steer_lat_accel")
        self.lat_max = float(lat) if lat is not None else None
        ego = cfg["sim"]["ego"]
        self.wheelbase = float(ego["front_wheelbase"]) + float(ego["rear_wheelbase"])
        self.actuation = ActuationMap(cfg["actuation"]["table"])
        self.u = np.zeros(2)

    def decide(self, action):
        s, x = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.lat_max is None:
            steer = a_lat = s
        else:
            a_lat = s * self.lat_max
            steer = float(lat_accel_to_steer(a_lat, self.env.ego.v, self.wheelbase, self.env.max_steering_rad))
        if self.a_max is None:
            self.u = np.array([steer, x])
            return {"vector": np.array([a_lat, x]), "k": 0}
        acc = x * (self.a_max if x >= 0 else -self.a_min)        # piecewise linear, 0 -> 0 m/s^2 (hold speed)
        self.u = np.array([steer, np.clip(self.actuation.accel_to_command(acc), -1.0, 1.0)])
        return {"vector": np.array([a_lat, acc]), "k": 0}

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
