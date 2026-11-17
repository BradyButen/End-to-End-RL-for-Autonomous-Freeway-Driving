"""Rewards (spec §9): the term catalog (each term: name, docstring definition, units) and the config-driven
composer (weighted sum of the terms, weights from configs/reward/<variant>.yaml).

Terms are evaluated per control step. ``decision`` carries per-decision data (M3 projection distances)
and a ``first_step`` flag so per-decision terms are charged exactly once.
Raw values are logged for every term at every control step regardless of weight (spec §9.1).
"""

from __future__ import annotations

import numpy as np

from handoff.simulator.road import RoadFrame
from handoff.simulator.state import EgoState, ObjectList


class RewardTerm:
    name: str = ""
    units: str = ""
    terminal: bool = False

    def __init__(self, ctx: dict):
        self.ctx = ctx  # constants: dt, v_max, v_target, road (bound at reset)
        self.reset()

    def reset(self) -> None:
        pass

    def compute(self, prev: EgoState, curr: EgoState, objects: ObjectList, info: dict,
                decision: dict | None) -> float:
        raise NotImplementedError


class Progress(RewardTerm):
    """Δs along the route since the previous control step. Units: m."""
    name, units = "progress", "m"

    def compute(self, prev, curr, objects, info, decision):
        return float(curr.s - prev.s)


class Speed(RewardTerm):
    """v / v_max (v_max = vehicle cap, 80 km/h). Units: dimensionless."""
    name, units = "speed", "-"

    def compute(self, prev, curr, objects, info, decision):
        return float(curr.v / self.ctx["v_max"])


class SpeedTracking(RewardTerm):
    """−|v − v_target| / v_target. Units: dimensionless."""
    name, units = "speed_tracking", "-"

    def compute(self, prev, curr, objects, info, decision):
        return -abs(curr.v - self.ctx["v_target"]) / self.ctx["v_target"]


class Success(RewardTerm):
    """Terminal indicator: route completed (MetaDrive arrive_dest). Units: 1/0."""
    name, units, terminal = "success", "1", True

    def compute(self, prev, curr, objects, info, decision):
        return float(bool(info.get("arrive_dest", False)))


class Collision(RewardTerm):
    """Terminal indicator: crash with a vehicle or object. Units: 1/0."""
    name, units, terminal = "collision", "1", True

    def compute(self, prev, curr, objects, info, decision):
        return float(bool(info.get("crash_vehicle", False) or info.get("crash_object", False)
                          or info.get("crash_human", False)))


class OffRoad(RewardTerm):
    """Terminal indicator: out of road (incl. crossing continuous edge lines / sidewalk). Units: 1/0."""
    name, units, terminal = "off_road", "1", True

    def compute(self, prev, curr, objects, info, decision):
        return float(bool(info.get("out_of_road", False)))


class LaneCenter(RewardTerm):
    """−|d − nearest lane-centre d| / (lane_width / 2). Units: dimensionless (0 at centre, −1 at a lane line)."""
    name, units = "lane_center", "-"

    def compute(self, prev, curr, objects, info, decision):
        road: RoadFrame = self.ctx["road"]
        dev = min(abs(curr.d - c) for c in road.lane_centers)
        return -dev / (0.5 * road.lane_width)


class Jerk(RewardTerm):
    """−|longitudinal jerk|, from finite differences of speed. Units: m/s³."""
    name, units = "jerk", "m/s^3"

    def reset(self):
        self.prev_acc = None

    def compute(self, prev, curr, objects, info, decision):
        dt = self.ctx["dt"]
        acc = (curr.v - prev.v) / dt
        jerk = 0.0 if self.prev_acc is None else (acc - self.prev_acc) / dt
        self.prev_acc = acc
        return -abs(jerk)


class SteerRate(RewardTerm):
    """−|Δsteering| / dt on the applied normalised steering command. Units: 1/s."""
    name, units = "steer_rate", "1/s"

    def compute(self, prev, curr, objects, info, decision):
        return -abs(curr.steering - prev.steering) / self.ctx["dt"]


class Time(RewardTerm):
    """Constant −1 per control step (weight sets the per-step cost). Units: 1/step."""
    name, units = "time", "1"

    def compute(self, prev, curr, objects, info, decision):
        return -1.0


class ProjectionPenalty(RewardTerm):
    """M3 only, per decision (charged on the decision's first control step): −Σ waypoint projection
    distances (incl. the penalty distance for dropped waypoints). Units: m."""
    name, units = "projection_penalty", "m"

    def compute(self, prev, curr, objects, info, decision):
        if not decision or not decision.get("first_step", False):
            return 0.0
        dists = decision.get("projection_distances")
        return -float(np.sum(dists)) if dists is not None else 0.0


ALL_TERMS = [Progress, Speed, SpeedTracking, Success, Collision, OffRoad, LaneCenter, Jerk, SteerRate, Time,
             ProjectionPenalty]
TERM_NAMES = [t.name for t in ALL_TERMS]


# --------------------------------------------------------------------------------------- composer
class RewardComposer:
    def __init__(self, terms: list[RewardTerm], weights: dict[str, float], terminal_mode: str = "add",
                 terminal_priority: list[str] | None = None):
        self.terms = terms
        self.names = [t.name for t in terms]
        unknown = set(weights) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown reward terms in weights: {sorted(unknown)}")
        self.weights = {n: float(weights.get(n, 0.0)) for n in self.names}
        if terminal_mode not in ("add", "replace"):
            raise ValueError(terminal_mode)
        self.terminal_mode = terminal_mode
        self.terminal_names = [t.name for t in terms if t.terminal]
        self.terminal_priority = terminal_priority or self.terminal_names

    @classmethod
    def from_config(cls, reward_cfg: dict, ctx: dict) -> RewardComposer:
        terms = [T(ctx) for T in ALL_TERMS]
        return cls(terms, reward_cfg["weights"], reward_cfg.get("terminal_mode", "add"),
                   reward_cfg.get("terminal_priority"))

    def reset(self) -> None:
        for t in self.terms:
            t.reset()

    def compute(self, prev: EgoState, curr: EgoState, objects: ObjectList, info: dict,
                decision: dict | None = None) -> tuple[float, dict[str, float]]:
        """Returns (weighted total, raw values of all terms)."""
        raw = {t.name: float(t.compute(prev, curr, objects, info, decision)) for t in self.terms}
        return self.total(raw), raw

    def total(self, raw: dict[str, float]) -> float:
        if self.terminal_mode == "replace":
            for name in self.terminal_priority:
                if raw.get(name, 0.0) != 0.0:
                    return self.weights[name] * raw[name]
        return float(sum(self.weights[n] * raw[n] for n in self.names))

    def weighted(self, raw: dict[str, float]) -> dict[str, float]:
        """Per-term weighted contributions consistent with ``total`` (for logging)."""
        if self.terminal_mode == "replace":
            for name in self.terminal_priority:
                if raw.get(name, 0.0) != 0.0:
                    return {n: (self.weights[n] * raw[n] if n == name else 0.0) for n in self.names}
        return {n: self.weights[n] * raw[n] for n in self.names}

