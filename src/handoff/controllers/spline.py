"""M3 path reference: smooth spline from the ego pose through the k projected waypoints (spec §7.3).

Piecewise quintic Hermite in s: the first segment starts at the ego's (d, slope = tan ψ_err, curvature 0);
every waypoint is passed with zero slope and curvature, which makes the path C² and free of overshoot
between waypoints (d stays within each segment's endpoint range). The path is extended flat beyond the
last waypoint so the MPC horizon is always covered; waypoints beyond the MPC horizon still shape it.
"""

from __future__ import annotations

import numpy as np

from handoff.controllers.base import Reference
from handoff.controllers.trajectory import quintic_coefficients
from handoff.sim.road import RoadFrame
from handoff.sim.state import EgoState


def _segment(d0, slope0, d1, L, s_local):
    c = quintic_coefficients(d0, slope0, 0.0, d1, L)
    x = np.clip(s_local, 0.0, L)
    d = sum(c[i] * x**i for i in range(6))
    dp = sum(i * c[i] * x ** (i - 1) for i in range(1, 6))
    return d, dp


def build_m3_reference(
    ego: EgoState,
    road: RoadFrame,
    waypoints: np.ndarray,
    v_target: float,
    ds: float = 1.0,
    extend: float = 80.0,
    max_start_slope: float = 0.5,
) -> Reference:
    """``waypoints``: (k, 2) array of (s, d), strictly increasing in s, all ahead of the ego."""
    wps = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    knots_s = np.concatenate([[ego.s], wps[:, 0]])
    knots_d = np.concatenate([[ego.d], wps[:, 1]])
    s_end = knots_s[-1] + extend
    s = np.arange(ego.s, s_end + 1e-9, ds)
    d = np.full_like(s, knots_d[-1])
    dp = np.zeros_like(s)
    slope0 = float(np.clip(np.tan(ego.psi_err), -max_start_slope, max_start_slope))
    for j in range(len(knots_s) - 1):
        s0, s1 = knots_s[j], knots_s[j + 1]
        m = (s >= s0) & (s < s1) if j < len(knots_s) - 2 else (s >= s0) & (s <= s1)
        if not np.any(m):
            continue
        d_seg, dp_seg = _segment(knots_d[j], slope0 if j == 0 else 0.0, knots_d[j + 1], s1 - s0, s[m] - s0)
        d[m] = d_seg
        dp[m] = dp_seg
    x, y = road.to_world(s, d)
    psi = road.heading_at(s) + np.arctan(dp)
    v_ref = max(v_target, 0.1)
    t = (s - ego.s) / v_ref
    v = np.full_like(s, v_target)
    return Reference(t=t, s=s, d=d, x=np.asarray(x), y=np.asarray(y), psi=np.asarray(psi), v=v,
                     a=np.zeros_like(s))


def lane_hold_reference(ego: EgoState, road: RoadFrame, v_target: float, length: float = 120.0,
                        ds: float = 1.0) -> Reference:
    """Fallback reference: centre of the ego's current (nearest) lane straight ahead."""
    d_lane = road.lane_centers[road.nearest_lane(ego.d)]
    wp = np.array([[ego.s + 20.0, d_lane]])
    return build_m3_reference(ego, road, wp, v_target, ds=ds, extend=length)
