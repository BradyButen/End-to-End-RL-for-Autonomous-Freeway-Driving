"""What M2 and M3 share: the RL's lane choice (``LaneTarget``), the reference path their controllers track (M2:
quintic trajectory in time; M3: ``LanePlan``, a quintic in arc length, or a waypoint spline in validation), and the
quintic polynomial both are built from. Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    """Samples along the reference. ``t`` is time since the reference was built (M3: s / v*, nominal)."""

    t: np.ndarray
    s: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    a: np.ndarray                    # longitudinal acceleration along the reference
    d_dot: np.ndarray | None = None  # M2: lateral rate / acceleration, for replan continuity
    d_ddot: np.ndarray | None = None

    def at_time(self, t: float) -> dict:
        """Every channel linearly interpolated at time ``t`` (held at the ends)."""
        out = {n: float(np.interp(t, self.t, getattr(self, n)))
               for n in ("s", "d", "x", "y", "v", "a", "d_dot", "d_ddot") if getattr(self, n) is not None}
        out["psi"] = float(np.interp(t, self.t, np.unwrap(self.psi)))
        return out

    def at_s(self, s: float) -> dict:
        """d, psi, v and t interpolated at arc length ``s``."""
        out = {n: float(np.interp(s, self.s, getattr(self, n))) for n in ("t", "d", "x", "y", "v", "a")}
        out["psi"] = float(np.interp(s, self.s, np.unwrap(self.psi)))
        out["s"] = float(s)
        return out

    def as_array(self) -> np.ndarray:
        """(n, 7) [t, s, d, x, y, psi, v] for logging."""
        return np.stack([self.t, self.s, self.d, self.x, self.y, self.psi, self.v], axis=1).astype(np.float32)


def quintic_coefficients(d0: float, dd0: float, ddd0: float, d1: float, T: float) -> np.ndarray:
    """c0..c5 of d(t) = sum c_i t^i with d(0) = d0, d'(0) = dd0, d''(0) = ddd0, d(T) = d1, d'(T) = d''(T) = 0."""
    c0, c1, c2 = d0, dd0, 0.5 * ddd0
    A = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
    b = np.array([d1 - (c0 + c1 * T + c2 * T**2), -(c1 + 2 * c2 * T), -(2 * c2)])
    return np.array([c0, c1, c2, *np.linalg.solve(A, b)])


def eval_quintic(c: np.ndarray, t: np.ndarray, T: float):
    """d, d', d'' at ``t``; held at the end value (zero rates) beyond T."""
    tt = np.minimum(t, T)
    d = sum(c[i] * tt**i for i in range(6))
    dd = sum(i * c[i] * tt ** (i - 1) for i in range(1, 6))
    ddd = sum(i * (i - 1) * c[i] * tt ** (i - 2) for i in range(2, 6))
    beyond = np.asarray(t) > T
    return d, np.where(beyond, 0.0, dd), np.where(beyond, 0.0, ddd)


# ======================================================================================= lane choice (M2, M3)
KEEP, LEFT, RIGHT = 0, 1, 2     # the RL's lateral command; KEEP is index 0, the default action


class LaneTarget:
    """The target lane the RL steers with relative commands (configs/base.yaml ``lane_action``).

    LEFT / RIGHT move the target one lane (lane 0 = next to the centre line = leftmost; saturates at the edge
    lanes), KEEP holds it. A lane change, once started, runs to completion: while the ego is more than ``settle`` m
    from the target lane's centre, new commands are ignored (no dithering mid-manoeuvre)."""

    def __init__(self, cfg: dict):
        self.settle = float(cfg["settle"])
        self.lane = 0

    def reset(self, ego, road) -> None:
        self.lane = road.nearest_lane(ego.d)

    def changing(self, ego, road) -> bool:
        return abs(ego.d - road.lane_centers[self.lane]) > self.settle

    def update(self, cmd: int, ego, road) -> bool:
        """Apply one command; True when the target lane changed."""
        if cmd == KEEP or self.changing(ego, road):
            return False
        new = int(np.clip(self.lane + (-1 if cmd == LEFT else 1), 0, len(road.lane_centers) - 1))
        changed, self.lane = new != self.lane, new
        return changed

    def offset(self, ego, road) -> int:
        """Target lane minus the ego's current lane, in lanes (+ = to the left; observation feature)."""
        return road.nearest_lane(ego.d) - self.lane


@dataclass
class LanePlan:
    """M3's lateral plan: a quintic d(s) from (d0, slope0) at s0 to the target lane centre ``d1`` at s0 + length,
    flat after."""

    s0: float
    length: float
    d1: float
    c: np.ndarray

    @classmethod
    def build(cls, ego, d1: float, cfg: dict) -> LanePlan:
        """length = speed x ``duration``, lengthened until the peak lateral acceleration (5.77 |dd| v^2 / length^2)
        is within ``max_lat_accel``, and at least ``min_length``. Starts at the ego's d with slope tan(psi_err)."""
        v = max(float(ego.v), 1.0)
        dd = abs(d1 - ego.d)
        length = max(v * float(cfg["duration"]), v * np.sqrt(5.77 * dd / float(cfg["max_lat_accel"])),
                     float(cfg["min_length"]))
        slope = float(np.tan(np.clip(ego.psi_err, -0.5, 0.5)))
        return cls(s0=float(ego.s), length=float(length), d1=float(d1),
                   c=quintic_coefficients(float(ego.d), slope, 0.0, float(d1), float(length)))

    def at(self, s):
        """d, d'(s), d''(s) along the road (held at d1 beyond the end, and at the start before s0)."""
        return eval_quintic(self.c, np.maximum(np.asarray(s, dtype=float) - self.s0, 0.0), self.length)

    def peak_lat_accel(self, v: float) -> float:
        return 5.77 * abs(self.d1 - float(self.c[0])) * float(v) ** 2 / self.length**2

    def reference(self, ego, road, v: float, ds: float, ahead: float) -> Reference:
        """The plan from the ego to ``ahead`` m past the plan's end, at constant speed v (t = s / v)."""
        s = np.arange(ego.s, max(self.s0 + self.length, ego.s) + ahead + 1e-9, ds)
        d, slope, _ = self.at(s)
        x, y = road.to_world(s, d)
        v = max(float(v), 0.1)
        return Reference(t=(s - ego.s) / v, s=s, d=d, x=np.asarray(x), y=np.asarray(y),
                         psi=np.asarray(road.heading_at(s) + np.arctan(slope)), v=np.full_like(s, v),
                         a=np.zeros_like(s))
