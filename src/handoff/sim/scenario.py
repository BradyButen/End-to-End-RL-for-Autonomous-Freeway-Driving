"""Per-episode scenario sampling (spec §4.6). Pure numpy; MetaDrive-free.

Positions are Frenet ``s`` (centre of the vehicle, m from route start) and MetaDrive lane index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from handoff.config import kmh


@dataclass
class VehicleSpec:
    lane: int
    s: float
    speed: float            # m/s
    model: str = "default"  # MetaDrive vehicle type key: default, s, m, l, xl
    lane_change: bool = False
    role: str = "background"


@dataclass
class ScenarioSpec:
    seed: int
    ego: VehicleSpec
    lead: VehicleSpec
    background: list[VehicleSpec] = field(default_factory=list)

    @property
    def traffic(self) -> list[VehicleSpec]:
        return [self.lead, *self.background]

    def as_dict(self) -> dict:
        return asdict(self)


class Scenario(Protocol):
    def sample(self, rng: np.random.Generator, seed: int = -1) -> ScenarioSpec: ...


# Vehicle lengths used for spacing / gap computations (MetaDrive vehicle_type DEFAULT_LENGTH).
VEHICLE_LENGTHS = {"default": 4.515, "s": 4.3, "m": 4.6, "l": 4.87, "xl": 5.74}


class FreewayPassScenario:
    """Ego behind a slow XL lead vehicle in the same lane; IDM background traffic in the other lanes."""

    def __init__(self, cfg: dict, lane_num: int = 3):
        self.cfg = cfg
        self.lane_num = lane_num

    def sample(self, rng: np.random.Generator, seed: int = -1) -> ScenarioSpec:
        c = self.cfg
        ego_lane = int(rng.choice(c["ego"]["lanes"]))
        ego_s = float(c["ego"]["s0"])
        ego = VehicleSpec(lane=ego_lane, s=ego_s, speed=kmh(rng.uniform(*c["ego"]["speed_km_h"])),
                          model="default", role="ego")

        lead_model = c["lead"].get("vehicle_model", "xl")
        gap = float(rng.uniform(*c["lead"]["gap"]))
        lead_s = ego_s + 0.5 * VEHICLE_LENGTHS["default"] + gap + 0.5 * VEHICLE_LENGTHS[lead_model]
        lead = VehicleSpec(lane=ego_lane, s=lead_s, speed=kmh(rng.uniform(*c["lead"]["speed_km_h"])),
                           model=lead_model, lane_change=bool(c["lead"].get("lane_change", False)), role="lead")

        bg_cfg = c["background"]
        lo, hi = bg_cfg["count"]
        n_bg = int(rng.integers(lo, hi + 1))
        lane_mode = bg_cfg.get("lanes", "other")
        if lane_mode == "other":
            lanes = [i for i in range(self.lane_num) if i != ego_lane]
        elif lane_mode == "all":
            lanes = list(range(self.lane_num))
        else:
            lanes = list(lane_mode)
        occupied: dict[int, list[float]] = {ego_lane: [ego_s, lead_s]}
        background = []
        s_lo, s_hi = bg_cfg.get("s_bounds", [15.0, 900.0])
        for _ in range(n_bg):
            for _attempt in range(30):
                lane = int(rng.choice(lanes))
                if bg_cfg.get("s_range") is not None:   # absolute placement along the route
                    s = float(rng.uniform(*bg_cfg["s_range"]))
                else:                                   # relative to the ego
                    s = float(np.clip(ego_s + rng.uniform(*bg_cfg["s_offset"]), s_lo, s_hi))
                taken = occupied.get(lane, [])
                if all(abs(s - o) >= bg_cfg["min_spacing"] for o in taken):
                    occupied.setdefault(lane, []).append(s)
                    model = str(rng.choice(bg_cfg.get("vehicle_models", ["default"])))
                    background.append(VehicleSpec(lane=lane, s=s, speed=kmh(rng.uniform(*bg_cfg["speed_km_h"])),
                                                  model=model, lane_change=bool(bg_cfg.get("lane_change", False))))
                    break
        return ScenarioSpec(seed=int(seed), ego=ego, lead=lead, background=background)


def make_scenario(cfg: dict) -> Scenario:
    name = cfg["scenario"]["name"]
    if name in ("freeway_pass", "freeway_dense"):
        return FreewayPassScenario(cfg["scenario"], lane_num=cfg["sim"]["map"]["lane_num"])
    raise ValueError(f"Unknown scenario {name}")
