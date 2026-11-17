"""Cascaded PID tracking controller for M2 (spec §7.2).

Lateral: heading PID on (ψ_ref − ψ) + lateral PID on cross-track (d_ref − d), summed into steering.
Longitudinal: speed PID on (v_ref − v) [+ optional accel feed-forward via the actuation map].
Sign convention: positive steering turns left (+ψ), d is positive to the left — so both errors map
directly to positive steering. Initial lateral gains are MetaDrive LaneChangePolicy's.
"""

from __future__ import annotations

import numpy as np

from handoff.methods.common.actuation import ActuationMap
from handoff.methods.common.reference import Reference
from handoff.sim.road import wrap_angle
from handoff.sim.state import EgoState, ObjectList


class PID:
    """Discrete PID (per-step sums/differences, like MetaDrive's PIDController, but with +error sign)."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float | None = None):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit = integral_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error: float | None = None

    def __call__(self, error: float) -> float:
        self.integral += error
        if self.integral_limit is not None:
            self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        deriv = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


class CascadedPIDController:
    def __init__(self, heading_gains, lateral_gains, speed_gains, integral_limit: float = 5.0,
                 actuation: ActuationMap | None = None, dt: float = 0.1):
        self.heading_pid = PID(*heading_gains, integral_limit=integral_limit)
        self.lateral_pid = PID(*lateral_gains, integral_limit=integral_limit)
        self.speed_pid = PID(*speed_gains, integral_limit=integral_limit)
        self.actuation = actuation
        self.dt = dt
        self.t_since_plan = 0.0

    @classmethod
    def from_config(cls, pid_cfg: dict, actuation: ActuationMap | None, dt: float) -> CascadedPIDController:
        return cls(pid_cfg["heading"], pid_cfg["lateral"], pid_cfg["speed"], pid_cfg.get("integral_limit", 5.0),
                   actuation if pid_cfg.get("speed_feedforward", True) else None, dt)

    def reset(self) -> None:
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.speed_pid.reset()
        self.t_since_plan = 0.0

    def new_reference(self) -> None:
        """Called at each replan; the time index restarts (PID integrators are kept for continuity)."""
        self.t_since_plan = 0.0

    def step(self, ego: EgoState, reference: Reference, objects: ObjectList) -> tuple[np.ndarray, dict]:
        # Lateral: reference evaluated at the ego's current arc length (spatial tracking).
        r_s = reference.at_s(ego.s)
        heading_err = float(wrap_angle(r_s["psi"] - ego.psi))
        cross_track = r_s["d"] - ego.d
        steering = self.heading_pid(heading_err) + self.lateral_pid(cross_track)

        # Longitudinal: reference evaluated at the next control instant (time tracking).
        r_t = reference.at_time(self.t_since_plan + self.dt)
        speed_err = r_t["v"] - ego.v
        throttle = self.speed_pid(speed_err)
        if self.actuation is not None:
            # feed-forward: command that produces the reference accel (includes coast-drag compensation)
            throttle += self.actuation.accel_to_command(r_t.get("a", 0.0))
        self.t_since_plan += self.dt

        u = np.clip(np.array([steering, throttle], dtype=float), -1.0, 1.0)
        diag = {
            "tracking_error": float(abs(cross_track)),
            "cross_track": float(cross_track),
            "heading_error": heading_err,
            "speed_error": float(speed_err),
            "ref_d": r_s["d"],
            "ref_v": r_t["v"],
            "ref_psi": r_s["psi"],
            "solve_time": 0.0,
            "fallback": False,
            "status": "ok",
        }
        return u, diag
