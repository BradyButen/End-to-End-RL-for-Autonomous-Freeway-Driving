"""Reward: a weighted sum of raw terms, computed every control step and summed over a decision by the env. Terminal
terms are added on the step the episode ends. Weights live in configs/r5.yaml.

The idea behind r5: progress and speed earn reward on every step of normal driving, so ending the episode early
never pays; a quadratic shortfall from the speed cap makes crawling expensive; a crash costs more than any finish.

Raw terms, per control step:
    progress      distance travelled along the road since the previous step (m)
    speed         (v / v_max)^2: 1 at 80 km/h, 0.32 at 45 km/h, 0 stopped
    speed_gap     -((v - v_max) / v_max)^2: -1 stopped, -0.25 at 40 km/h, 0 at 80 km/h. Steepest where the car is
                  slowest, so crawling is the costliest way to drive
    success       1 on reaching the end of the route
    collision     1 on a crash
    off_road      1 on leaving the road (including crossing the centre line)
    lane_keep     -(offset / half a lane width)^2, offset = distance from the nearest lane centre: 0 centred,
                  -1 on a lane line. Smooth, so every correction toward the centre pays
    heading       -(|heading error| - 0.5 deg) in rad, when positive: straight driving is free, weaving is not
    steer_rate    -|change in the lateral acceleration the steering asks for| (m/s^2), with
                  a_lat = v^2 tan(steering angle) / wheelbase. Speed-aware: the same twitch costs more at speed
    headway       -(1 - headway / 0.5 s) when positive, for the nearest car in line with the ego, ahead (gap / ego
                  speed: tailgating) or behind (gap / its speed: cutting in). 0 at 0.5 s or more, -1 at contact
    lane_change   lanes changed this step, counted once the ego settles near the new lane's centre (logged only:
                  feeds the lane_changes metric)
"""

from __future__ import annotations

import numpy as np

from handoff.sim import EgoState, ObjectList, RoadFrame

TERMS = ("progress", "speed", "speed_gap", "success", "collision", "off_road", "lane_keep", "heading",
         "steer_rate", "headway", "lane_change")
LANE_SETTLE = 0.3   # fraction of a lane width
HEADING_DEADBAND = float(np.deg2rad(0.5))   # rad; heading error below this is free
HEADWAY_S = 0.5     # s; time headway below which following (or being followed after a cut-in) is charged
V_HEADWAY_MIN = 1.0  # m/s; speed floor in the headway (a car stopped behind a stopped car is not tailgating)


def crashed(info: dict) -> bool:
    """Every MetaDrive crash that ends the episode (a sidewalk hit ends it as out_of_road instead)."""
    return bool(info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_human")
                or info.get("crash_building"))


class Reward:
    def __init__(self, reward_cfg: dict, v_max: float, wheelbase: float, max_steering_rad: float):
        self.v_max = float(v_max)     # m/s, the ego's speed cap (sim.ego.max_speed_km_h)
        self.wheelbase, self.max_steering_rad = float(wheelbase), float(max_steering_rad)
        weights = reward_cfg["weights"]
        unknown = [n for n, w in weights.items() if n not in TERMS and float(w) != 0.0]
        if unknown:   # zero-weight entries for unknown terms are ignored
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
            "speed": (ego.v / self.v_max) ** 2,
            "speed_gap": -((ego.v - self.v_max) / self.v_max) ** 2,
            "success": float(bool(info.get("arrive_dest"))),
            "collision": float(crashed(info)),
            "off_road": float(bool(info.get("out_of_road"))),
            "lane_keep": -self._lane_offset(ego) ** 2,
            "heading": -max(abs(ego.psi_err) - HEADING_DEADBAND, 0.0),
            "steer_rate": -abs(self._lat_accel(ego) - self._lat_accel(prev)),
            "headway": -self._headway(ego, objects),
            "lane_change": self._lane_change(prev, ego),
        }
        weighted = {n: self.weights[n] * raw[n] for n in TERMS}
        return float(sum(weighted.values())), raw, weighted

    def _lane_offset(self, ego: EgoState) -> float:
        """Distance from the nearest lane centre / half a lane width (0 centred, 1 on a lane line)."""
        road = self.road
        return abs(ego.d - road.lane_centers[road.nearest_lane(ego.d)]) / (0.5 * road.lane_width)

    def _lat_accel(self, e: EgoState) -> float:
        """Lateral acceleration (m/s^2) the applied steering asks for at the current speed (kinematic bicycle)."""
        return e.v**2 * float(np.tan(e.steering * self.max_steering_rad)) / self.wheelbase

    @staticmethod
    def _headway(ego: EgoState, objects: ObjectList) -> float:
        """Worst (1 - headway / HEADWAY_S) over cars overlapping the ego sideways, ahead or behind; 0 = none."""
        worst = 0.0
        for o in objects:
            if abs(o.d - ego.d) >= 0.5 * (o.width + ego.width):    # beside the ego: not in its path
                continue
            gap = max(abs(o.s - ego.s) - 0.5 * (o.length + ego.length), 0.0)
            v = ego.v if o.s >= ego.s else o.v                       # the follower's speed
            worst = max(worst, 1.0 - min(gap / max(v, V_HEADWAY_MIN) / HEADWAY_S, 1.0))
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
