"""M2 reference generation (spec §7.2): quintic lateral profile + acceleration-limited speed profile."""

from __future__ import annotations

import numpy as np

from handoff.methods.common.quintic import eval_quintic, quintic_coefficients
from handoff.methods.common.reference import Reference
from handoff.sim.road import RoadFrame
from handoff.sim.state import EgoState


def speed_profile(v0: float, v_target: float, t: np.ndarray, accel_limit: float, decel_limit: float):
    """Piecewise-constant-acceleration ramp from v0 to v_target. Returns (v(t), a(t), s(t) from 0)."""
    dv = v_target - v0
    a = accel_limit if dv >= 0 else -decel_limit
    t_ramp = abs(dv) / max(abs(a), 1e-9)
    v = np.where(t < t_ramp, v0 + a * t, v_target)
    acc = np.where(t < t_ramp, a, 0.0)
    s_ramp = v0 * np.minimum(t, t_ramp) + 0.5 * a * np.minimum(t, t_ramp) ** 2
    s = s_ramp + v_target * np.maximum(t - t_ramp, 0.0)
    return v, acc, s


def build_m2_reference(
    ego: EgoState,
    road: RoadFrame,
    d_target: float,
    v_target: float,
    horizon: float,
    dt: float,
    accel_limit: float,
    decel_limit: float,
    lateral_init: tuple[float, float] | None = None,
    min_duration: float = 0.0,
) -> Reference:
    """Reference from the current Frenet state to (d_target, v_target).

    ``lateral_init`` = (d_dot, d_ddot) at the replan instant (from the previous reference, for continuity);
    if None they are estimated from the measured state (d_dot = v sin(psi_err), d_ddot = 0).
    """
    T = max(horizon, 1e-3)
    duration = max(T, min_duration)
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    if lateral_init is None:
        dd0, ddd0 = ego.v * np.sin(ego.psi_err), 0.0
    else:
        dd0, ddd0 = lateral_init
    c = quintic_coefficients(ego.d, dd0, ddd0, d_target, T)
    d, d_dot, d_ddot = eval_quintic(c, t, T)
    v, acc, s_rel = speed_profile(max(ego.v, 0.0), max(v_target, 0.0), t, accel_limit, decel_limit)
    # Arc length along the reference line (lateral motion adds negligible path length at these slopes).
    s = ego.s + s_rel
    x, y = road.to_world(s, d)
    s_dot = np.maximum(v, 0.5)
    psi = road.heading_at(s) + np.arctan2(d_dot, s_dot)
    return Reference(t=t, s=s, d=d, x=np.asarray(x), y=np.asarray(y), psi=np.asarray(psi), v=v, a=acc,
                     d_dot=d_dot, d_ddot=d_ddot)
