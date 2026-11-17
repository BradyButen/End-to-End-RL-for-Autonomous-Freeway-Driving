"""Static mapping between longitudinal acceleration and MetaDrive ``throttle_brake`` (spec §19).

No system identification: a monotone lookup table of (throttle_brake, steady acceleration) pairs measured
once open-loop on the bare map (``scripts/calibrate_actuation.py``) and stored in ``configs/base.yaml``.
Brake response is strongly nonlinear in MetaDrive (≈ -8 m/s² at -0.3, -15 m/s² at -1), hence a table
rather than two gains.
"""

from __future__ import annotations

import numpy as np


class ActuationMap:
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
