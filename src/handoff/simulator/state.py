"""Shared state representations (spec §5, §16). Plain dataclasses; no MetaDrive import at module level."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from handoff.simulator.road import RoadFrame, wrap_angle


@dataclass
class EgoState:
    t: float
    x: float
    y: float
    psi: float
    v: float
    s: float
    d: float
    psi_err: float
    steering: float          # last applied, normalised [-1, 1]
    throttle_brake: float    # last applied, normalised [-1, 1]
    lane_index: int
    length: float
    width: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackedObject:
    id: str
    x: float
    y: float
    psi: float
    v: float
    vx: float
    vy: float
    s: float
    d: float
    length: float
    width: float
    type: str

    def as_dict(self) -> dict:
        return asdict(self)


ObjectList = list[TrackedObject]

# Column order used when objects are packed into fixed-size arrays for logging.
OBJECT_FIELDS = ("x", "y", "psi", "v", "vx", "vy", "s", "d", "length", "width")
EGO_FIELDS = ("t", "x", "y", "psi", "v", "s", "d", "psi_err", "steering", "throttle_brake", "lane_index",
              "length", "width")


def make_ego_state(t, x, y, psi, v, steering, throttle_brake, length, width, road: RoadFrame) -> EgoState:
    s, d = road.to_frenet(x, y)
    psi_err = float(wrap_angle(psi - road.heading_at(s)))
    return EgoState(t=float(t), x=float(x), y=float(y), psi=float(psi), v=float(v), s=s, d=d, psi_err=psi_err,
                    steering=float(steering), throttle_brake=float(throttle_brake),
                    lane_index=road.nearest_lane(d), length=float(length), width=float(width))


def make_tracked_object(oid, x, y, psi, vx, vy, length, width, otype, road: RoadFrame) -> TrackedObject:
    s, d = road.to_frenet(x, y)
    return TrackedObject(id=str(oid), x=float(x), y=float(y), psi=float(psi), v=float(np.hypot(vx, vy)),
                         vx=float(vx), vy=float(vy), s=s, d=d, length=float(length), width=float(width),
                         type=str(otype))


def frenet_velocity(obj_or_ego, road: RoadFrame) -> tuple[float, float]:
    """(s_dot, d_dot) of a TrackedObject or EgoState."""
    h = road.heading_at(obj_or_ego.s)
    if isinstance(obj_or_ego, TrackedObject):
        vx, vy = obj_or_ego.vx, obj_or_ego.vy
    else:
        vx, vy = obj_or_ego.v * np.cos(obj_or_ego.psi), obj_or_ego.v * np.sin(obj_or_ego.psi)
    c, s_ = np.cos(h), np.sin(h)
    return float(c * vx + s_ * vy), float(-s_ * vx + c * vy)


def pack_objects(objects: ObjectList, max_n: int) -> np.ndarray:
    """(max_n, len(OBJECT_FIELDS)) array, NaN-padded, nearest-first order preserved."""
    arr = np.full((max_n, len(OBJECT_FIELDS)), np.nan, dtype=np.float32)
    for i, o in enumerate(objects[:max_n]):
        arr[i] = [getattr(o, f) for f in OBJECT_FIELDS]
    return arr


def frenet_box_gap(ego: EgoState, obj: TrackedObject) -> float:
    """Clearance (m) between ego and object footprints, approximated as axis-aligned boxes in the
    Frenet frame (exact on the straight freeway with near-aligned headings). 0 when overlapping."""
    lon = abs(obj.s - ego.s) - 0.5 * (obj.length + ego.length)
    lat = abs(obj.d - ego.d) - 0.5 * (obj.width + ego.width)
    return float(np.hypot(max(lon, 0.0), max(lat, 0.0)))


def time_to_collision(ego: EgoState, obj: TrackedObject, road: RoadFrame) -> float:
    """Longitudinal TTC (s) for objects laterally overlapping the ego footprint; inf otherwise."""
    lat = abs(obj.d - ego.d) - 0.5 * (obj.width + ego.width)
    if lat > 0:
        return float("inf")
    gap = abs(obj.s - ego.s) - 0.5 * (obj.length + ego.length)
    ego_sdot, _ = frenet_velocity(ego, road)
    obj_sdot, _ = frenet_velocity(obj, road)
    closing = (ego_sdot - obj_sdot) if obj.s >= ego.s else (obj_sdot - ego_sdot)
    if closing <= 1e-6:
        return float("inf")
    return float(max(gap, 0.0) / closing)
