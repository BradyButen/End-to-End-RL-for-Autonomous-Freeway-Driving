"""B0 driver: MetaDrive's rule-based IDMPolicy (IDM car following + lane changes) at our global target speed.

Installed as MetaDrive's ``agent_policy`` for the ego, so MetaDrive calls it every step and ignores the env action.
"""

from __future__ import annotations

# imported via sim.metadrive_env so its headless SDL env vars are set before MetaDrive loads
from handoff.sim.metadrive_env import IDMPolicy


class EgoIDMPolicy(IDMPolicy):
    """Stock IDMPolicy (car following + lane changes) at the global target speed."""

    def __init__(self, control_object, random_seed):
        super().__init__(control_object=control_object, random_seed=random_seed)
        cfg = self.engine.global_config
        self.NORMAL_SPEED = float(cfg["handoff_target_speed_km_h"])
        self.target_speed = self.NORMAL_SPEED
        self.enable_lane_change = bool(cfg["handoff_idm_lane_change"])
