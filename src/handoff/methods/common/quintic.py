"""Quintic polynomial helpers shared by M2 (lateral profile in time) and M3 (spline segments in arc length)."""

from __future__ import annotations

import numpy as np


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
