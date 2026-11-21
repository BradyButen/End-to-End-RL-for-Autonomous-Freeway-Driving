"""The reference path both controllers track (M2: quintic trajectory in time, M3: spline in arc length), and the
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
