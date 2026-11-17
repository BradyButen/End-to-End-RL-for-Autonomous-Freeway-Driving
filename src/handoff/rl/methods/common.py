"""Pieces shared by M2 and M3: the Reference type and Controller interface (spec §16), quintic polynomial
helpers (M2 lateral profile in time, M3 spline segments in arc length), and the throttle/brake lookup table.
No MetaDrive / SB3 dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from handoff.simulator.state import EgoState, ObjectList


# --------------------------------------------------------------------------------------- reference
@dataclass
class Reference:
    """Time-indexed reference samples. ``t`` is relative to the reference's creation time.

    Both M2 (quintic trajectory) and M3 (spline path through projected waypoints) produce this; for M3,
    ``t`` / ``v`` are filled assuming the global target speed and the path is primarily used spatially.
    """

    t: np.ndarray
    s: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    a: np.ndarray | None = None      # longitudinal accel along the reference (optional)
    d_dot: np.ndarray | None = None  # lateral velocity (optional, for replan continuity)
    d_ddot: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.t)

    def at_time(self, t: float) -> dict:
        """Linear interpolation of every channel at relative time ``t`` (held at the ends)."""
        out = {}
        for name in ("s", "d", "x", "y", "v", "a", "d_dot", "d_ddot"):
            arr = getattr(self, name)
            if arr is not None:
                out[name] = float(np.interp(t, self.t, arr))
        out["psi"] = float(_interp_angle(t, self.t, self.psi))
        return out

    def at_s(self, s: float) -> dict:
        """Interpolation by arc length (s must be increasing along the reference)."""
        out = {}
        for name in ("t", "d", "x", "y", "v", "a"):
            arr = getattr(self, name)
            if arr is not None:
                out[name] = float(np.interp(s, self.s, arr))
        out["psi"] = float(_interp_angle(s, self.s, self.psi))
        out["s"] = float(s)
        return out

    def as_array(self) -> np.ndarray:
        """(n, 7) array [t, s, d, x, y, psi, v] for logging."""
        return np.stack([self.t, self.s, self.d, self.x, self.y, self.psi, self.v], axis=1).astype(np.float32)


def _interp_angle(q, xs, angles):
    u = np.unwrap(angles)
    return float(np.interp(q, xs, u))


class Controller(Protocol):
    def reset(self) -> None: ...

    def step(self, ego: EgoState, reference: Reference, objects: ObjectList) -> tuple[np.ndarray, dict]:
        """Returns ([steering, throttle_brake] in [-1,1]^2, diagnostics dict e.g.
        {'solve_time': ..., 'status': ..., 'fallback': bool, 'tracking_error': ...})."""
        ...


# --------------------------------------------------------------------------------------- quintic
def quintic_coefficients(d0: float, dd0: float, ddd0: float, d1: float, T: float) -> np.ndarray:
    """Coefficients c0..c5 of d(t) = Σ c_i t^i with d(0)=d0, d'(0)=dd0, d''(0)=ddd0, d(T)=d1, d'(T)=d''(T)=0."""
    c0, c1, c2 = d0, dd0, 0.5 * ddd0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    A = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
    b = np.array([d1 - (c0 + c1 * T + c2 * T2), -(c1 + 2 * c2 * T), -(2 * c2)])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5])


def eval_quintic(c: np.ndarray, t: np.ndarray, T: float):
    """d, d', d'' at times t; held constant (at the target, zero rates) for t > T."""
    tt = np.minimum(t, T)
    d = sum(c[i] * tt**i for i in range(6))
    dd = sum(i * c[i] * tt ** (i - 1) for i in range(1, 6))
    ddd = sum(i * (i - 1) * c[i] * tt ** (i - 2) for i in range(2, 6))
    beyond = t > T
    dd = np.where(beyond, 0.0, dd)
    ddd = np.where(beyond, 0.0, ddd)
    return d, dd, ddd


# --------------------------------------------------------------------------------------- actuation
class ActuationMap:
    """Static mapping between longitudinal acceleration and MetaDrive ``throttle_brake`` (spec §19).

    No system identification: a monotone lookup table of (throttle_brake, steady acceleration) pairs measured
    once open-loop on the bare map (``scripts/calibrate_actuation.py``) and stored in ``configs/base.yaml``.
    Brake response is strongly nonlinear in MetaDrive (≈ -8 m/s² at -0.3, -15 m/s² at -1), hence a table
    rather than two gains.
    """

    def __init__(self, table):
        arr = np.asarray(table, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        if not np.all(np.diff(arr[:, 1]) > 0):
            raise ValueError("actuation table must be strictly increasing in acceleration")
        self.tb = arr[:, 0]
        self.acc = arr[:, 1]

    @property
    def a_min(self) -> float:
        return float(self.acc[0])

    @property
    def a_max(self) -> float:
        return float(self.acc[-1])

    def accel_to_command(self, a: float) -> float:
        """Desired accel (m/s²) -> throttle_brake in [-1, 1] (clipped to the table range)."""
        return float(np.interp(a, self.acc, self.tb))

    def command_to_accel(self, u: float) -> float:
        return float(np.interp(u, self.tb, self.acc))
