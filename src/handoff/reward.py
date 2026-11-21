"""Reward (configs/r*.yaml): a weighted sum of raw terms, computed every control step and summed over a
decision by the env. Terminal terms are added on the step the episode ends.

    progress      Δs since the previous step (m)
    time          -1 per step
    success       1 on route completion (MetaDrive arrive_dest)
    collision     1 on a crash, or an M3 failed run (emergency brake / infeasible waypoints ended the episode)
    off_road      1 when out of road (incl. crossing the centre line / road edge)
    lane_change   lanes changed this step: counts once the ego centre settles within 0.3 lane widths of a new
                  lane's centre (two lanes at once = 2)
    heading       -(|heading error| - 0.5 deg), when positive (rad): straight driving costs nothing, oscillation
                  does. Summed over a lane change it is about lateral distance / speed whatever the manoeuvre's
                  length (3.5 m at 20 m/s: ~1.5 rad-steps after the deadband), so a pass stays cheap.
    speed_gap     -((v - v_max) / v_max)^2: quadratic shortfall from the 80 km/h cap (0 at 80, -0.25 at 40 km/h)
    lane_keep     -1 while the ego centre is more than 0.57 m from its nearest lane centre, else 0 (on 3.5 m lanes
                  the body then stays ~25 cm inside the lane lines, so riding a line is charged). A lane change
                  crosses it in ~1-1.5 s.
    proximity     -(G - gap) / G for the closest car whose body overlaps the ego's sideways (same lane, or mid lane
                  change), when the bumper-to-bumper gap is under G: G = 5 m for a car ahead (following), 2 m for a
                  car behind (e.g. cutting in after a pass). 0 at G, -1 at contact. Cars beside the ego never count.
"""

from __future__ import annotations

import numpy as np

from handoff.sim import EgoState, ObjectList, RoadFrame

TERMS = ("progress", "time", "success", "collision", "off_road", "lane_change", "heading", "speed_gap",
         "lane_keep", "proximity")
TERMINAL = ("success", "collision", "off_road")   # charged once, on the step the episode ends
LANE_SETTLE = 0.3   # fraction of a lane width
HEADING_DEADBAND = float(np.deg2rad(0.5))   # rad; heading error below this is free
LANE_ZONE = 0.57        # m, in-lane zone half-width around each lane centre (ego centre)
PROXIMITY_GAP_AHEAD = 5.0    # m, bumper-to-bumper gap to a car ahead below which proximity is charged
PROXIMITY_GAP_BEHIND = 2.0   # m, same for a car behind


def crashed(info: dict) -> bool:
    """Every MetaDrive crash that ends the episode (a sidewalk hit ends it as out_of_road instead)."""
    return bool(info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_human")
                or info.get("crash_building"))


class Reward:
    def __init__(self, reward_cfg: dict, v_max: float):
        self.v_max = float(v_max)     # m/s, the ego's speed cap (sim.ego.max_speed_km_h)
        weights = reward_cfg["weights"]
        unknown = [n for n, w in weights.items() if n not in TERMS and float(w) != 0.0]
        if unknown:   # weight-0 entries for removed terms (old run configs) are ignored
            raise ValueError(f"Unknown reward terms with non-zero weight: {unknown}")
        self.weights = {n: float(weights.get(n, 0.0)) for n in TERMS}
        self.road: RoadFrame | None = None
        self.lane: int | None = None

    def reset(self, road: RoadFrame) -> None:
        self.road = road
        self.lane = None

    def step(self, prev: EgoState, ego: EgoState, info: dict, objects: ObjectList) -> tuple[float, dict, dict]:
        """(total, raw terms, weighted terms) for one control step."""
        raw = {
            "progress": ego.s - prev.s,
            "time": -1.0,
            "success": float(bool(info.get("arrive_dest"))),
            "collision": float(crashed(info) or bool(info.get("emergency_brake") or info.get("infeasible_plan"))),
            "off_road": float(bool(info.get("out_of_road"))),
            "lane_change": self._lane_change(prev, ego),
            "heading": -max(abs(ego.psi_err) - HEADING_DEADBAND, 0.0),
            "speed_gap": -((ego.v - self.v_max) / self.v_max) ** 2,
            "lane_keep": -float(self._off_lane(ego)),
            "proximity": -self._proximity(ego, objects),
        }
        weighted = {n: self.weights[n] * raw[n] for n in TERMS}
        return float(sum(weighted.values())), raw, weighted

    def _off_lane(self, ego: EgoState) -> bool:
        road = self.road
        return abs(ego.d - road.lane_centers[road.nearest_lane(ego.d)]) > LANE_ZONE

    @staticmethod
    def _proximity(ego: EgoState, objects: ObjectList) -> float:
        worst = 0.0
        for o in objects:
            if abs(o.d - ego.d) >= 0.5 * (o.width + ego.width):    # beside the ego: fine
                continue
            gap = abs(o.s - ego.s) - 0.5 * (o.length + ego.length)
            g = PROXIMITY_GAP_AHEAD if o.s >= ego.s else PROXIMITY_GAP_BEHIND
            worst = max(worst, min(max((g - gap) / g, 0.0), 1.0))
        return worst

    def _lane_change(self, prev: EgoState, ego: EgoState) -> float:
        road = self.road
        if self.lane is None:
            self.lane = road.nearest_lane(prev.d)
        j = road.nearest_lane(ego.d)
        if j != self.lane and abs(ego.d - road.lane_centers[j]) < LANE_SETTLE * road.lane_width:
            n, self.lane = abs(j - self.lane), j
            return float(n)
        return 0.0
