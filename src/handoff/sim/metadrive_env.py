"""MetaDrive integration (spec §4): fixed freeway map, pinned ego dynamics, custom scenario manager.

MetaDrive source is not modified; customisation is via config, subclassing and managers.

One MetaDrive instance per process (MetaDrive limitation), so the per-episode ``ScenarioSpec`` is handed
to the managers through the module-level ``_EPISODE`` holder, set immediately before ``reset``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for top-down observations
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no audio device on servers/WSL (silences ALSA errors)

import numpy as np  # noqa: E402
from metadrive.component.map.pg_map import MapGenerateMethod, PGMap  # noqa: E402
from metadrive.component.pgblock.first_block import FirstPGBlock  # noqa: E402
from metadrive.component.vehicle.vehicle_type import vehicle_type as MD_VEHICLE_TYPES  # noqa: E402
from metadrive.constants import DEFAULT_AGENT  # noqa: E402
from metadrive.envs.base_env import BaseEnv  # noqa: E402
from metadrive.envs.varying_dynamics_env import (  # noqa: E402
    VaryingDynamicsAgentManager,
    VaryingDynamicsEnv,
)
from metadrive.manager.pg_map_manager import PGMapManager  # noqa: E402
from metadrive.manager.traffic_manager import PGTrafficManager, TrafficMode  # noqa: E402
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel  # noqa: E402
from metadrive.policy.env_input_policy import EnvInputPolicy  # noqa: E402
from metadrive.policy.idm_policy import IDMPolicy  # noqa: E402

from handoff.sim.scenario import ScenarioSpec, VehicleSpec  # noqa: E402

_EPISODE: dict = {"spec": None, "segments": None, "road": None}


def get_road_frame(engine):
    """RoadFrame of the (fixed) map, built once per process and cached."""
    if _EPISODE["road"] is None:
        from handoff.sim.road import RoadFrame

        net = engine.current_map.road_network
        cfg = engine.current_map.config
        _EPISODE["road"] = RoadFrame.from_metadrive(net, route_nodes(net), cfg["lane_num"], cfg["lane_width"])
    return _EPISODE["road"]


def geometric_off_road(road, position, margin: float = 0.0) -> bool:
    """Vehicle centre outside the drivable road (Frenet bounds) or beyond the route ends.

    Replaces MetaDrive's ``not vehicle.on_lane``: that flag comes from a single vertical raycast against the
    lane-surface meshes, which sporadically misses at mesh seams (≈0.1 % of road points beyond x≈150 m),
    producing spurious off-road terminations mid-lane and silently deleting traffic (DECISIONS D-10).
    """
    s, d = road.to_frenet(position[0], position[1])
    return d < road.d_min - margin or d > road.d_max + margin or s < -1.0 or s > road.length + 1.0


# --------------------------------------------------------------------------------------- map helpers
def build_block_config(map_cfg: dict) -> list[dict]:
    """PG_MAP_FILE block list: FirstPGBlock + straight blocks with pinned lengths (MetaDrive samples
    straight-block lengths in [40, 80] m by default)."""
    blocks = [{"id": "I", "pre_block_socket_index": None}]
    for i, length in enumerate(map_cfg["straight_block_lengths"]):
        prev = "I" if i == 0 else "S"
        blocks.append({"id": "S", "pre_block_socket_index": f"{i}{prev}-socket0", "length": float(length)})
    return blocks


def route_nodes(road_network) -> list[str]:
    """Forward node sequence from the map start (unique on a straight map)."""
    nodes = [FirstPGBlock.NODE_1]
    while True:
        nxt = [n for n in road_network.graph.get(nodes[-1], {}) if not n.startswith("-")]
        if not nxt:
            return nodes
        nodes.append(nxt[0])


def road_segments(road_network) -> list[tuple[str, str, float, float]]:
    """[(from, to, s_start, length)] along the route, measured on lane 0."""
    nodes = route_nodes(road_network)
    out, s = [], 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        length = float(road_network.graph[a][b][0].length)
        out.append((a, b, s, length))
        s += length
    return out


def lane_at(segments, s: float, lane: int) -> tuple[tuple[str, str, int], float]:
    """MetaDrive (lane index, longitudinal) for route arc length ``s`` (straight roads: all lanes share s)."""
    for a, b, s0, length in segments:
        if s < s0 + length or (a, b) == segments[-1][:2]:
            return (a, b, int(lane)), float(np.clip(s - s0, 0.0, length))
    raise ValueError(s)


def _spawn_config(spec: VehicleSpec, segments, road_network) -> dict:
    lane_index, lon = lane_at(segments, spec.s, spec.lane)
    lane = road_network.get_lane(lane_index)
    h = lane.heading_theta_at(lon)
    return {
        "spawn_lane_index": lane_index,
        "spawn_longitude": lon,
        "spawn_lateral": 0.0,
        "spawn_velocity": [spec.speed * np.cos(h), spec.speed * np.sin(h)],
        "spawn_velocity_car_frame": False,
    }


# --------------------------------------------------------------------------------------- one-way map
class OneWayPGMap(PGMap):
    """PG map built from a block list with ``remove_negative_lanes=True`` on every block: a one-way road
    (no oncoming carriageway). MetaDrive's blocks support the flag but PGMap never passes it through."""

    def _config_generate(self, blocks_config, parent_node_path, physics_world):
        assert len(self.road_network.graph) == 0, "map must be empty"
        last_block = FirstPGBlock(
            global_network=self.road_network,
            lane_width=self._config.get(self.LANE_WIDTH, 3.5),
            lane_num=self._config.get(self.LANE_NUM, 2),
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=self._config.get("exit_length", 50),
            start_point=self._config.get("start_position", [0, 0]),
            ignore_intersection_checking=True,
            remove_negative_lanes=True,
        )
        self.blocks.append(last_block)
        for block_index, b in enumerate(blocks_config[1:], 1):
            b = dict(b)
            block_type = self.engine.global_config["block_dist_config"].get_block(b.pop(self.BLOCK_ID))
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index,
                last_block.get_socket(pre_block_socket_index),
                self.road_network,
                random_seed=self.engine.global_random_seed,
                ignore_intersection_checking=True,
                remove_negative_lanes=True,
            )
            last_block.construct_from_config(b, parent_node_path, physics_world)
            self.blocks.append(last_block)


class FreewayMapManager(PGMapManager):
    """PGMapManager that builds ``OneWayPGMap`` when ``handoff_one_way`` is set (map fixed, cached per seed)."""

    def reset(self):
        if not self.engine.global_config.get("handoff_one_way", False):
            return super().reset()
        seed = self.engine.global_seed
        if self.maps[seed] is None:
            map_config = self.engine.global_config.copy()["map_config"]
            map_config.update({"seed": seed})
            m = self.spawn_object(OneWayPGMap, map_config=map_config, random_seed=None)
            self.current_map = m
            if self.engine.global_config["store_map"]:
                self.maps[seed] = m
        else:
            self.load_map(self.maps[seed])


# --------------------------------------------------------------------------------------- policies
class ScenarioIDMPolicy(IDMPolicy):
    """MetaDrive IDM with a per-vehicle cruise speed and lane-change switch (stock default: 30 km/h)."""

    def __init__(self, control_object, random_seed, target_speed_km_h: float = 30.0, lane_change: bool = True):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED = float(target_speed_km_h)  # instance attr shadows the class constant
        self.target_speed = self.NORMAL_SPEED
        self.enable_lane_change = bool(lane_change)

    def acceleration(self, front_obj, dist_to_front) -> float:
        if self.target_speed < 1.0:  # stopped vehicle (stock IDM divides by the target speed)
            return -1.0
        return super().acceleration(front_obj, dist_to_front)


class EgoIDMPolicy(IDMPolicy):
    """B0 baseline: stock IDMPolicy (car following + lane changes) at the global target speed."""

    def __init__(self, control_object, random_seed):
        super().__init__(control_object=control_object, random_seed=random_seed)
        cfg = self.engine.global_config
        self.NORMAL_SPEED = float(cfg["handoff_target_speed_km_h"])
        self.target_speed = self.NORMAL_SPEED
        self.enable_lane_change = bool(cfg["handoff_idm_lane_change"])


# --------------------------------------------------------------------------------------- managers
class ScenarioTrafficManager(PGTrafficManager):
    """Spawns exactly the lead + background vehicles of the current ScenarioSpec (spec §4.4)."""

    def __init__(self):
        super().__init__()
        self.roles: dict[str, str] = {}

    def reset(self):
        self.mode = TrafficMode.Basic
        self.block_triggered_vehicles = []
        self.roles = {}
        spec: ScenarioSpec | None = _EPISODE["spec"]
        if spec is None:
            return
        net = self.current_map.road_network
        segments = _EPISODE["segments"]
        for vs in spec.traffic:
            v_config = _spawn_config(vs, segments, net)
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            v_config["enable_reverse"] = False
            veh = self.spawn_object(MD_VEHICLE_TYPES[vs.model], vehicle_config=v_config)
            self.add_policy(veh.id, ScenarioIDMPolicy, veh, self.generate_seed(),
                            vs.speed * 3.6, vs.lane_change)
            self._traffic_vehicles.append(veh)
            self.roles[veh.name] = vs.role

    def after_step(self, *args, **kwargs):
        """As PGTrafficManager.after_step (Basic mode, no respawn) but removes a vehicle only when it is
        geometrically off the road / past the route end, not on raycast ``on_lane`` misses."""
        road = get_road_frame(self.engine)
        to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if geometric_off_road(road, v.position):
                to_remove.append(v)
        for v in to_remove:
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
        return dict()


class ScenarioAgentManager(VaryingDynamicsAgentManager):
    """Ego spawn (lane, s, speed) from the ScenarioSpec; dynamics pinned via ``random_dynamics``
    constant ranges (VaryingDynamicsAgentManager), which also serves reach goal 1."""

    def reset(self):
        spec: ScenarioSpec | None = _EPISODE["spec"]
        if spec is not None and self.engine.current_map is not None:
            net = self.engine.current_map.road_network
            if _EPISODE["segments"] is None:
                _EPISODE["segments"] = road_segments(net)
            get_road_frame(self.engine)
            cfg = _spawn_config(spec.ego, _EPISODE["segments"], net)
            self.engine.global_config["agent_configs"][DEFAULT_AGENT].update(cfg)
        super().reset()


# --------------------------------------------------------------------------------------- env
class FreewayMetaDriveEnv(VaryingDynamicsEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "resolution_size": 84,
                "distance": 100.0,
                "handoff_target_speed_km_h": 70.0,
                "handoff_idm_lane_change": True,
                "handoff_one_way": False,
            },
            allow_add_new_key=True,
        )
        return config

    def setup_engine(self):
        BaseEnv.setup_engine(self)
        self.engine.register_manager("map_manager", FreewayMapManager())
        self.engine.register_manager("traffic_manager", ScenarioTrafficManager())

    def _get_agent_manager(self):
        return ScenarioAgentManager(init_observations=self._get_observations())

    def _is_out_of_road(self, vehicle):
        """MetaDrive semantics (continuous-line / sidewalk contact when on_continuous_line_done) but with the
        raycast ``on_lane`` test replaced by the geometric road-bounds test (see geometric_off_road)."""
        ret = geometric_off_road(self.road_frame, vehicle.position)
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line \
                or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    @property
    def road_frame(self):
        return get_road_frame(self.engine)

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=False,
            clip_rgb=self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"],
        )

    def reset_with_spec(self, spec: ScenarioSpec):
        """Reset to the (single, fixed) map with this episode's initial conditions."""
        _EPISODE["spec"] = spec
        return super().reset(seed=self.config["start_seed"])

    @property
    def traffic_roles(self) -> dict[str, str]:
        return self.engine.traffic_manager.roles


def make_metadrive_config(cfg: dict, render: bool = False, b0: bool = False) -> dict:
    sim = cfg["sim"]
    ego = sim["ego"]
    bev = cfg["observation"]["bev"]
    md = dict(
        use_render=render,
        log_level=sim.get("log_level", 50),
        physics_world_step_size=sim["physics_world_step_size"],
        decision_repeat=sim["decision_repeat"],
        horizon=sim["horizon"],
        num_scenarios=1,
        start_seed=0,
        traffic_density=0.0,
        random_spawn_lane_index=False,
        map_config={
            "type": MapGenerateMethod.PG_MAP_FILE,
            "config": build_block_config(sim["map"]),
            "lane_num": sim["map"]["lane_num"],
            "lane_width": sim["map"]["lane_width"],
            "exit_length": sim["map"]["first_block_length"],
        },
        random_dynamics=dict(
            max_engine_force=(ego["max_engine_force"], ego["max_engine_force"]),
            max_brake_force=(ego["max_brake_force"], ego["max_brake_force"]),
            wheel_friction=(ego["wheel_friction"], ego["wheel_friction"]),
            max_steering=(ego["max_steering_deg"], ego["max_steering_deg"]),
            mass=None,
        ),
        vehicle_config=dict(
            vehicle_model="varying_dynamics",
            lidar=dict(num_lasers=0, distance=50),
            show_navi_mark=render,
            show_dest_mark=False,
            show_navigation_arrow=render,
        ),
        on_continuous_line_done=sim["on_continuous_line_done"],
        out_of_route_done=False,
        norm_pixel=True,
        frame_skip=bev["frame_skip"],
        frame_stack=bev["frame_stack"],
        post_stack=bev["post_stack"],
        resolution_size=bev["resolution"],
        distance=float(bev["distance"]),
        handoff_target_speed_km_h=float(sim["target_speed_km_h"]),
        handoff_idm_lane_change=bool(cfg.get("method", {}).get("idm_lane_change", True)),
        handoff_one_way=bool(sim["map"].get("one_way", False)),
        agent_policy=EgoIDMPolicy if b0 else EnvInputPolicy,
        show_logo=False,
        show_fps=render,
        preload_models=False,
    )
    md.update(sim.get("metadrive_overrides") or {})  # raw MetaDrive config passthrough (advanced)
    return md

