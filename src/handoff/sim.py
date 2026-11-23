"""Everything about the simulator: MetaDrive setup, the road (Frenet) frame, ego / car state, traffic speeds,
scripted cars and the throttle/brake <-> acceleration table. No RL and no controllers here.

MetaDrive is used stock (configured, never patched): MetaDriveEnv (its own observation is ignored: the policy's is
built from ground-truth state in env.py), its PG
map ("S" blocks = straight road, block lengths drawn from the seed), its PGTrafficManager (IDM cars) and its own
success / crash / out-of-road rules.

Frenet conventions (used everywhere):
  * s = arc length (m) along the route; d = lateral offset (m), positive to the LEFT of the driving direction;
  * reference line = centre of MetaDrive lane 0 (the lane next to the centre line), so lane centres are
    d = 0, -3.5, -7.0; ``d_max`` / ``d_min`` are the left (centre line) and right road edges.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no audio device on servers / WSL

import numpy as np  # noqa: E402


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ======================================================================================= MetaDrive setup
def make_metadrive_config(cfg: dict, render: bool = False, agent_policy=None) -> dict:
    """MetaDrive config dict from our YAML. ``agent_policy``: MetaDrive policy class driving the ego (None = our
    env action; B0 passes IDMPolicy)."""
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.manager.traffic_manager import TrafficMode
    from metadrive.policy.env_input_policy import EnvInputPolicy

    sim, traffic = cfg["sim"], cfg["scenario"]["traffic"]
    seeds = cfg["scenario"]["seeds"]
    spawn_lane = sim["ego"].get("spawn_lane")            # None = random lane every episode
    clean = render and bool(sim.get("render_clean", False))   # no MetaDrive navigation marks / dashboard
    md = dict(
        use_render=render,
        log_level=sim.get("log_level", 50),
        physics_world_step_size=sim["physics_world_step_size"],
        decision_repeat=sim["decision_repeat"],
        horizon=sim["horizon"],
        start_seed=0,
        store_map=bool(sim.get("store_map", False)),   # MetaDrive default True keeps every visited map (~10 MB each,
                                                       # up to num_scenarios per process): the memory growth that
                                                       # OOM-killed runs. False rebuilds the seed's map on reset.
        num_scenarios=max(int(seeds["train_high"]), max(int(s) for s in seeds["eval"]) + 1),
        map=sim["map"]["blocks"],
        map_config={"lane_num": sim["map"]["lane_num"], "lane_width": sim["map"]["lane_width"]},
        traffic_density=float(traffic["density"]),
        traffic_mode=getattr(TrafficMode, traffic.get("mode", "Trigger")),
        random_traffic=False,                        # traffic tied to the seed (reproducible episodes)
        need_inverse_traffic=False,
        random_spawn_lane_index=spawn_lane is None,
        vehicle_config=dict(
            vehicle_model=sim["ego"]["vehicle_model"],
            spawn_velocity=[sim["ego"]["spawn_speed_km_h"] / 3.6, 0.0],
            spawn_velocity_car_frame=True,
            lidar=dict(num_lasers=0, distance=50),   # IDM finds neighbours through the lidar's range query only
            show_navi_mark=render and not clean,
            show_dest_mark=False,
            show_navigation_arrow=render and not clean,
        ),
        on_continuous_line_done=sim["on_continuous_line_done"],
        out_of_route_done=False,
        agent_policy=agent_policy or EnvInputPolicy,
        show_logo=False,
        show_fps=render and not clean,
        preload_models=False,
    )
    if clean:
        md["show_interface"] = False
    if spawn_lane is not None:
        md["agent_configs"] = {"default_agent": {
            "use_special_color": True,
            "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, int(spawn_lane))}}
    md.update(sim.get("metadrive_overrides") or {})   # raw MetaDrive passthrough
    return md


def make_metadrive_env(cfg: dict, render: bool = False, agent_policy=None):
    from metadrive.envs.metadrive_env import MetaDriveEnv

    return MetaDriveEnv(make_metadrive_config(cfg, render=render, agent_policy=agent_policy))


def build_road_frame(md) -> RoadFrame:
    """Frenet frame of the current episode's map along the ego's route (rebuild after every reset)."""
    m = md.engine.current_map
    return RoadFrame.from_metadrive(m.road_network, md.agent.navigation.checkpoints,
                                    m.config["lane_num"], m.config["lane_width"])


# ======================================================================================= road frame
class RoadFrame:
    """Polyline reference line with s / d conversions. Pure numpy."""

    def __init__(self, points: np.ndarray, lane_centers: list[float], d_min: float, d_max: float,
                 lane_width: float):
        pts = np.asarray(points, dtype=float)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])  # drop duplicates
        pts = pts[keep]
        seg = np.diff(pts, axis=0)
        self.points = pts
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._seg_dir = seg / self._seg_len[:, None]
        self._seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        self.s_points = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self.length = float(self.s_points[-1])
        self.lane_centers = [float(c) for c in lane_centers]
        self.d_min, self.d_max = float(d_min), float(d_max)
        self.lane_width = float(lane_width)

    @classmethod
    def from_metadrive(cls, road_network, checkpoints: list[str], lane_num: int, lane_width: float,
                       spacing: float = 2.0) -> RoadFrame:
        """Stitch the lane-0 centrelines of consecutive roads along ``checkpoints`` (the navigation route)."""
        pts = []
        for a, b in zip(checkpoints[:-1], checkpoints[1:]):
            lane = road_network.graph[a][b][0]
            for lon in np.linspace(0.0, lane.length, max(2, int(np.ceil(lane.length / spacing)) + 1)):
                pts.append(np.asarray(lane.position(lon, 0.0), float)[:2])
        frame = cls(np.array(pts), [0.0], 0.0, 0.0, lane_width)   # provisional, to measure the lane offsets
        lanes = road_network.graph[checkpoints[0]][checkpoints[1]]
        centers = []
        for i in range(lane_num):
            p = np.asarray(lanes[i].position(lanes[i].length / 2, 0.0), float)[:2]
            centers.append(frame.to_frenet(p[0], p[1])[1])
        return cls(frame.points, centers, d_min=min(centers) - lane_width / 2, d_max=max(centers) + lane_width / 2,
                   lane_width=lane_width)

    def to_frenet(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=float)
        rel = p - self.points[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, self._seg_dir), 0.0, self._seg_len)
        proj = self.points[:-1] + self._seg_dir * t[:, None]
        i = int(np.argmin(np.sum((p - proj) ** 2, axis=1)))
        last = len(self._seg_len) - 1
        if i in (0, last):   # extrapolate beyond the ends along the first / last segment
            t_raw = float(np.dot(rel[i], self._seg_dir[i]))
            if (i == 0 and t_raw < 0) or (i == last and t_raw > self._seg_len[i]):
                t[i] = t_raw
        dvec = rel[i] - self._seg_dir[i] * t[i]
        d = self._seg_dir[i, 0] * dvec[1] - self._seg_dir[i, 1] * dvec[0]
        return float(self.s_points[i] + t[i]), float(d)

    def _locate(self, s):
        s = np.asarray(s, dtype=float)
        i = np.clip(np.searchsorted(self.s_points, s, side="right") - 1, 0, len(self._seg_len) - 1)
        return i, s - self.s_points[i]

    def to_world(self, s, d):
        """Scalars -> (x, y) floats; arrays -> (x array, y array)."""
        i, t = self._locate(s)
        base = self.points[i] + self._seg_dir[i] * np.asarray(t)[..., None]
        normal = np.stack([-self._seg_dir[i, 1], self._seg_dir[i, 0]], axis=-1)
        p = base + normal * np.asarray(d, dtype=float)[..., None]
        if p.ndim == 1:
            return float(p[0]), float(p[1])
        return p[..., 0], p[..., 1]

    def heading_at(self, s):
        h = self._seg_heading[self._locate(s)[0]]
        return float(h) if np.ndim(h) == 0 else h

    def nearest_lane(self, d: float) -> int:
        return int(np.argmin([abs(d - c) for c in self.lane_centers]))

    @property
    def d_center(self) -> float:
        return 0.5 * (self.d_min + self.d_max)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.d_max - self.d_min)


# ======================================================================================= state
@dataclass
class EgoState:
    t: float
    x: float
    y: float
    psi: float
    v: float
    s: float
    d: float
    psi_err: float           # heading relative to the road
    steering: float          # last applied, normalised [-1, 1]
    throttle_brake: float    # last applied, normalised [-1, 1]
    lane_index: int
    length: float
    width: float


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


ObjectList = list[TrackedObject]
EGO_FIELDS = ("t", "x", "y", "psi", "v", "s", "d", "psi_err", "steering", "throttle_brake", "lane_index",
              "length", "width")
OBJECT_FIELDS = ("x", "y", "psi", "v", "vx", "vy", "s", "d", "length", "width")   # packed-array column order


def extract_ego(md, road: RoadFrame, t: float) -> EgoState:
    a = md.agent
    x, y = a.position
    s, d = road.to_frenet(x, y)
    psi = float(a.heading_theta)
    return EgoState(t=float(t), x=float(x), y=float(y), psi=psi, v=float(a.speed), s=s, d=d,
                    psi_err=float(wrap_angle(psi - road.heading_at(s))), steering=float(a.steering),
                    throttle_brake=float(a.throttle_brake), lane_index=road.nearest_lane(d),
                    length=float(a.LENGTH), width=float(a.WIDTH))


def extract_objects(md, road: RoadFrame, radius: float, extra_vehicles=()) -> ObjectList:
    """Traffic (+ scripted) cars within ``radius`` of the ego, nearest first (ground truth)."""
    ex, ey = md.agent.position
    out = []
    for veh in list(md.engine.traffic_manager._traffic_vehicles) + list(extra_vehicles):
        x, y = veh.position
        dist = float(np.hypot(x - ex, y - ey))
        if dist > radius:
            continue
        vx, vy = veh.velocity
        s, d = road.to_frenet(x, y)
        out.append((dist, TrackedObject(id=str(veh.name), x=float(x), y=float(y), psi=float(veh.heading_theta),
                                        v=float(np.hypot(vx, vy)), vx=float(vx), vy=float(vy), s=s, d=d,
                                        length=float(veh.LENGTH), width=float(veh.WIDTH))))
    out.sort(key=lambda p: p[0])
    return [o for _, o in out]


def frenet_velocity(obj, road: RoadFrame) -> tuple[float, float]:
    """(s_dot, d_dot) of a TrackedObject or EgoState."""
    h = road.heading_at(obj.s)
    if isinstance(obj, TrackedObject):
        vx, vy = obj.vx, obj.vy
    else:
        vx, vy = obj.v * np.cos(obj.psi), obj.v * np.sin(obj.psi)
    c, s_ = np.cos(h), np.sin(h)
    return float(c * vx + s_ * vy), float(-s_ * vx + c * vy)


def pack_objects(objects: ObjectList, max_n: int) -> np.ndarray:
    """(max_n, len(OBJECT_FIELDS)) array, NaN-padded, nearest first."""
    arr = np.full((max_n, len(OBJECT_FIELDS)), np.nan, dtype=np.float32)
    for i, o in enumerate(objects[:max_n]):
        arr[i] = [getattr(o, f) for f in OBJECT_FIELDS]
    return arr


def box_gap(ego: EgoState, obj: TrackedObject) -> float:
    """Clearance (m) between the ego and car footprints as road-aligned boxes; 0 when overlapping."""
    lon = abs(obj.s - ego.s) - 0.5 * (obj.length + ego.length)
    lat = abs(obj.d - ego.d) - 0.5 * (obj.width + ego.width)
    return float(np.hypot(max(lon, 0.0), max(lat, 0.0)))


def time_to_collision(ego: EgoState, obj: TrackedObject, road: RoadFrame) -> float:
    """Longitudinal TTC (s) for cars overlapping the ego laterally; inf otherwise."""
    if abs(obj.d - ego.d) - 0.5 * (obj.width + ego.width) > 0:
        return float("inf")
    gap = abs(obj.s - ego.s) - 0.5 * (obj.length + ego.length)
    ego_sdot, obj_sdot = frenet_velocity(ego, road)[0], frenet_velocity(obj, road)[0]
    closing = (ego_sdot - obj_sdot) if obj.s >= ego.s else (obj_sdot - ego_sdot)
    return float(max(gap, 0.0) / closing) if closing > 1e-6 else float("inf")


# ======================================================================================= traffic
class TrafficSpeeds:
    """Traffic cruise speeds: ``scenario.traffic.speed_km_h`` + a per-lane offset (``lane_speed_offset_km_h``,
    indexed by MetaDrive lane, 0 = next to the centre line), re-read from each car's current lane every step."""

    def __init__(self, traffic_cfg: dict):
        ts = traffic_cfg.get("speed_km_h")
        self.base = None if ts is None else float(ts)
        self.offsets = [float(x) for x in traffic_cfg.get("lane_speed_offset_km_h") or []]
        self.release_all_at_reset = bool(traffic_cfg.get("release_all_at_reset", False))

    def speed_kmh(self, veh) -> float:
        v = self.base
        try:
            lane = int(veh.lane_index[-1])
        except (AttributeError, TypeError, IndexError):
            lane = None
        if lane is not None and 0 <= lane < len(self.offsets):
            v += self.offsets[lane]
        return v

    def on_reset(self, md) -> None:
        """Trigger-mode placement (random positions over every block but the first), but every car drives from
        t = 0 instead of waiting for the ego to reach its block, starting at its cruise speed."""
        if self.release_all_at_reset:
            tm = md.engine.traffic_manager
            for group in getattr(tm, "block_triggered_vehicles", []):
                tm._traffic_vehicles += list(tm.get_objects(group.vehicles).values())
            tm.block_triggered_vehicles = []
            if self.base is not None:
                for veh in tm._traffic_vehicles:
                    v, h = self.speed_kmh(veh) / 3.6, float(veh.heading_theta)
                    veh.set_velocity([v * np.cos(h), v * np.sin(h)])
        self.apply(md)

    def apply(self, md) -> None:
        """Set each traffic car's IDM cruise speed from its current lane. IDMPolicy resets ``target_speed`` from
        ``NORMAL_SPEED`` every step; set per policy instance, so B0's ego IDM keeps its own."""
        if self.base is None:
            return
        engine = md.engine
        for veh in engine.traffic_manager._traffic_vehicles:
            pol = engine.get_policy(veh.id)
            v = self.speed_kmh(veh)
            if pol is not None and getattr(pol, "NORMAL_SPEED", None) != v:
                pol.NORMAL_SPEED = v
                pol.target_speed = v


def spawn_car(md, road: RoadFrame, s: float, d: float, static: bool = True):
    """A scripted car (same body as the ego) at (s, d), facing along the road. Static cars are frozen; moving ones
    are driven by DrivingEnv (``add_scripted_car``)."""
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

    x, y = road.to_world(float(s), float(d))
    veh = md.engine.spawn_object(TrafficDefaultVehicle, vehicle_config={"spawn_velocity": None},
                                 position=[x, y], heading=float(road.heading_at(float(s))))
    veh.set_velocity([0.0, 0.0])
    if static:
        veh.set_static(True)
    return veh


# ======================================================================================= actuation
class ActuationMap:
    """Longitudinal acceleration (m/s^2) <-> MetaDrive ``throttle_brake``, as a measured monotone lookup table
    (``actuation.table`` in base.yaml, from scripts/calibrate_actuation.py). Brake response is strongly nonlinear
    (about -8 m/s^2 at -0.3, -15.8 at -1), hence a table rather than gains."""

    def __init__(self, table):
        arr = np.asarray(table, dtype=float)
        arr = arr[np.argsort(arr[:, 0])]
        if not np.all(np.diff(arr[:, 1]) > 0):
            raise ValueError("actuation table must be strictly increasing in acceleration")
        self.tb, self.acc = arr[:, 0], arr[:, 1]

    def accel_to_command(self, a: float) -> float:
        return float(np.interp(a, self.acc, self.tb))

    def command_to_accel(self, u: float) -> float:
        return float(np.interp(u, self.tb, self.acc))
