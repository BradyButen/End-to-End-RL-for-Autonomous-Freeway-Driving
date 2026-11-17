"""The Reference type shared by M2 and M3 (spec §16), plus the Controller interface. No MetaDrive / SB3 dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from handoff.sim.state import EgoState, ObjectList


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
