"""M2 — RL + PID. Every 5 control steps (0.5 s) the RL picks a lane command and a speed command:

    action MultiDiscrete [3, 3] = [lateral command, speed command]; [0, 0] = keep lane, hold speed (the default)
      lateral  KEEP / LEFT / RIGHT moves the target lane (reference.LaneTarget: one lane at a time, saturating at the
               edges; ignored while a lane change is still in progress) -> d* = the target lane's centre
      speed    HOLD / FASTER / SLOWER moves the target speed v* by 10 km/h (reference.SpeedTarget; sticky, starts at
               the spawn speed): the speed the PID tracks
      -> reference: quintic lateral profile to d* over 2 s (starting from the previous reference's lateral rate and
         acceleration, for continuity) + speed ramp to v* (+2 / -4 m/s^2)
      -> cascaded PID every control step:
             steering       = PID_heading(psi_ref - psi) + PID_lateral(d_ref - d)    reference at the ego's s
             throttle_brake = PID_speed(v_ref - v) + table^-1(a_ref)                  reference at the next instant
No safety layer: the RL owns the speed and must learn to brake for cars (M3's MPC does that for it).
Gains: configs/m2.yaml (tuned with scripts/matlab/m2_pid_tuning.m).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from handoff.env import Method
from handoff.reference import KEEP, LaneTarget, Reference, SpeedTarget, eval_quintic, quintic_coefficients
from handoff.sim import ActuationMap, EgoState, RoadFrame, wrap_angle


# ======================================================================================= reference
def speed_profile(v0: float, v1: float, t: np.ndarray, accel: float, decel: float):
    """Constant-acceleration ramp from v0 to v1, then hold. Returns v(t), a(t), s(t) (s from 0)."""
    a = accel if v1 >= v0 else -decel
    t_ramp = abs(v1 - v0) / max(abs(a), 1e-9)
    tr = np.minimum(t, t_ramp)
    v = np.where(t < t_ramp, v0 + a * t, v1)
    acc = np.where(t < t_ramp, a, 0.0)
    s = v0 * tr + 0.5 * a * tr**2 + v1 * np.maximum(t - t_ramp, 0.0)
    return v, acc, s


def build_m2_reference(ego: EgoState, road: RoadFrame, d_target: float, v_target: float, tr_cfg: dict,
                       lateral_init: tuple[float, float] | None, min_duration: float) -> Reference:
    """Reference from the ego's current state to (d*, v*). ``lateral_init`` = (d_dot, d_ddot) at the replan instant
    from the previous reference; None (first decision) = estimated from the ego (v sin(psi_err), 0)."""
    T = max(float(tr_cfg["horizon"]), 1e-3)
    dt = float(tr_cfg["dt"])
    t = np.arange(int(round(max(T, min_duration) / dt)) + 1) * dt
    dd0, ddd0 = (ego.v * np.sin(ego.psi_err), 0.0) if lateral_init is None else lateral_init
    d, d_dot, d_ddot = eval_quintic(quintic_coefficients(ego.d, dd0, ddd0, d_target, T), t, T)
    v, acc, s_rel = speed_profile(max(ego.v, 0.0), max(v_target, 0.0), t, float(tr_cfg["accel_limit"]),
                                  float(tr_cfg["decel_limit"]))
    s = ego.s + s_rel        # arc length along the road (lateral motion adds negligible length)
    x, y = road.to_world(s, d)
    psi = road.heading_at(s) + np.arctan2(d_dot, np.maximum(v, 0.5))
    return Reference(t=t, s=s, d=d, x=np.asarray(x), y=np.asarray(y), psi=np.asarray(psi), v=v, a=acc,
                     d_dot=d_dot, d_ddot=d_ddot)


# ======================================================================================= PID
class PID:
    """Discrete PID with per-step sums / differences and a clamped integral."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit = float(integral_limit)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error: float | None = None

    def __call__(self, error: float) -> float:
        self.integral = float(np.clip(self.integral + error, -self.integral_limit, self.integral_limit))
        deriv = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


class CascadedPID:
    """Positive steering turns left and d is positive left, so both lateral errors map directly to + steering."""

    def __init__(self, pid_cfg: dict, actuation: ActuationMap, dt: float):
        lim = float(pid_cfg.get("integral_limit", 5.0))
        self.heading = PID(*pid_cfg["heading"], integral_limit=lim)
        self.lateral = PID(*pid_cfg["lateral"], integral_limit=lim)
        self.speed = PID(*pid_cfg["speed"], integral_limit=lim)
        self.feedforward = actuation if pid_cfg.get("speed_feedforward", True) else None
        self.dt = dt

    def reset(self) -> None:
        for p in (self.heading, self.lateral, self.speed):
            p.reset()

    def step(self, ego: EgoState, ref: Reference, t: float) -> tuple[np.ndarray, dict]:
        """``t`` = time since the reference was built."""
        r_s = ref.at_s(ego.s)                       # lateral: spatial tracking at the ego's arc length
        heading_err = float(wrap_angle(r_s["psi"] - ego.psi))
        cross_track = r_s["d"] - ego.d
        steering = self.heading(heading_err) + self.lateral(cross_track)
        r_t = ref.at_time(t + self.dt)              # longitudinal: time tracking at the next control instant
        speed_err = r_t["v"] - ego.v
        throttle = self.speed(speed_err)
        if self.feedforward is not None:            # command that produces a_ref (includes coast-drag compensation)
            throttle += self.feedforward.accel_to_command(r_t["a"])
        u = np.clip(np.array([steering, throttle], dtype=float), -1.0, 1.0)
        return u, {"tracking_error": abs(cross_track), "cross_track": cross_track, "heading_error": heading_err,
                   "speed_error": speed_err, "ref_d": r_s["d"], "ref_v": r_t["v"], "ref_psi": r_s["psi"],
                   "status": "ok"}


# ======================================================================================= method
class M2(Method):
    def __init__(self, cfg: dict, env):
        m = cfg["method"]
        super().__init__(env, period=m["decision_period"], action_space=self.ACTION_SPACE)
        self.lane = LaneTarget(cfg["lane_action"])
        self.speed = SpeedTarget(cfg["lane_action"], env.v_cap)
        self.tr_cfg = m["trajectory"]
        self.pid = CascadedPID(m["pid"], ActuationMap(cfg["actuation"]["table"]), env.dt)
        self.t = 0.0    # time since the current reference was built

    ACTION_SPACE = gym.spaces.MultiDiscrete([3, 3])   # [lane command, speed command]
    COMMAND_FEATURES = ("target_speed",)   # the sticky target speed / v_max ("cmd"; target lane is an ego feature)

    def target_lane_offset(self) -> int:
        return self.lane.offset(self.env.ego, self.env.road)

    def command_state(self) -> list[float]:
        return [self.speed.v / self.env.v_max]

    def reset(self):
        super().reset()
        self.pid.reset()
        self.lane.reset(self.env.ego, self.env.road)
        self.speed.reset(self.env.ego)
        self.t = 0.0

    def decide(self, action):
        cmd, speed_cmd = (int(x) for x in np.asarray(action).ravel())
        self.lane.update(cmd, self.env.ego, self.env.road)
        return {**self.set_target(self.lane.lane, self.speed.update(speed_cmd), cmd), "speed_cmd": speed_cmd}

    def set_target(self, lane: int, v_star: float, cmd: int = KEEP) -> dict:
        """Reference to (target lane centre, v*) from here (also scripts/validate.py check m2: scripted targets)."""
        env = self.env
        self.lane.lane = int(lane)
        d_star = float(env.road.lane_centers[self.lane.lane])
        lat_init = None
        if self.reference is not None:
            r = self.reference.at_time(self.t)
            lat_init = (r["d_dot"], r["d_ddot"])
        self.reference = build_m2_reference(env.ego, env.road, d_star, v_star, self.tr_cfg, lat_init,
                                            min_duration=(self.period + 1) * env.dt)
        self.t = 0.0
        return {"vector": np.array([cmd, self.lane.lane, d_star, v_star]), "k": 0, "lane": self.lane.lane,
                "d_target": d_star, "v_target": v_star}

    def control(self):
        u, diag = self.pid.step(self.env.ego, self.reference, self.t)
        self.t += self.env.dt
        return u, diag
