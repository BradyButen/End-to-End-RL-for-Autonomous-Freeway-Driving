"""Reward (configs/r3_fast.yaml): a weighted sum of six raw terms, computed every control step and summed over a
decision by the env. Terminal terms are added on the step the episode ends.

    progress      Δs since the previous step (m)
    time          -1 per step
    success       1 on route completion (MetaDrive arrive_dest)
    collision     1 on a crash, or on an M3 emergency brake that ended the episode (a predicted crash)
    off_road      1 when out of road (incl. crossing the centre line / road edge)
    lane_change   lanes changed this step: counts once the ego centre settles within 0.3 lane widths of a new
                  lane's centre (two lanes at once = 2)
"""

from __future__ import annotations

from handoff.sim import EgoState, RoadFrame

TERMS = ("progress", "time", "success", "collision", "off_road", "lane_change")
LANE_SETTLE = 0.3   # fraction of a lane width


def crashed(info: dict) -> bool:
    return bool(info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_human"))


class Reward:
    def __init__(self, reward_cfg: dict):
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

    def step(self, prev: EgoState, ego: EgoState, info: dict) -> tuple[float, dict, dict]:
        """(total, raw terms, weighted terms) for one control step."""
        raw = {
            "progress": ego.s - prev.s,
            "time": -1.0,
            "success": float(bool(info.get("arrive_dest"))),
            "collision": float(crashed(info) or bool(info.get("emergency_brake"))),
            "off_road": float(bool(info.get("out_of_road"))),
            "lane_change": self._lane_change(prev, ego),
        }
        weighted = {n: self.weights[n] * raw[n] for n in TERMS}
        return float(sum(weighted.values())), raw, weighted

    def _lane_change(self, prev: EgoState, ego: EgoState) -> float:
        road = self.road
        if self.lane is None:
            self.lane = road.nearest_lane(prev.d)
        j = road.nearest_lane(ego.d)
        if j != self.lane and abs(ego.d - road.lane_centers[j]) < LANE_SETTLE * road.lane_width:
            n, self.lane = abs(j - self.lane), j
            return float(n)
        return 0.0
