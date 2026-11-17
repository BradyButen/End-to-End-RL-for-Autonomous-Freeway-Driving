"""MetaDrive setup: the stock ``TopDownMetaDrive`` env (multi-channel BEV), configured from our YAML.

Nothing here subclasses or patches MetaDrive. Map, traffic, ego spawn, termination and observation are all
MetaDrive's own, selected through its config dict:
  * map:      a string of PG blocks (e.g. "SSSS" = straight blocks; MetaDrive draws each block's length from the
              seed, so each seed is a slightly different straight road), lane count/width from ``sim.map``;
  * traffic:  MetaDrive's PGTrafficManager (stock IDMPolicy cars, 30 km/h) at ``scenario.traffic.density``;
  * episodes: seed s in [start_seed, start_seed + num_scenarios) picks the map + traffic (MetaDrive's own seeding);
  * done:     MetaDrive's success / crash / out-of-road / horizon rules.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for top-down observations
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no audio device on servers/WSL (silences ALSA errors)

from metadrive.envs.top_down_env import TopDownMetaDrive  # noqa: E402
from metadrive.manager.traffic_manager import TrafficMode  # noqa: E402
from metadrive.policy.env_input_policy import EnvInputPolicy  # noqa: E402

from handoff.simulator.road import RoadFrame  # noqa: E402

__all__ = ["TopDownMetaDrive", "make_metadrive_config", "make_metadrive_env", "num_scenarios", "build_road_frame"]


def num_scenarios(cfg: dict) -> int:
    """MetaDrive seeds run from 0 to num_scenarios - 1; cover both the training range and the eval list."""
    seeds = cfg["scenario"]["seeds"]
    return max(int(seeds["train_high"]), max(int(s) for s in seeds["eval"]) + 1)


def make_metadrive_config(cfg: dict, render: bool = False, agent_policy=None) -> dict:
    """``agent_policy``: MetaDrive policy class driving the ego (None = our env action; B0 passes IDMPolicy)."""
    sim = cfg["sim"]
    bev = cfg["observation"]["bev"]
    traffic = cfg["scenario"]["traffic"]
    md = dict(
        use_render=render,
        log_level=sim.get("log_level", 50),
        physics_world_step_size=sim["physics_world_step_size"],
        decision_repeat=sim["decision_repeat"],
        horizon=sim["horizon"],
        start_seed=0,
        num_scenarios=num_scenarios(cfg),
        map=sim["map"]["blocks"],
        map_config={"lane_num": sim["map"]["lane_num"], "lane_width": sim["map"]["lane_width"]},
        traffic_density=float(traffic["density"]),
        traffic_mode=getattr(TrafficMode, traffic.get("mode", "Trigger")),   # "basic" | "trigger" | "respawn"
        random_traffic=False,                       # traffic tied to the seed (reproducible episodes)
        need_inverse_traffic=False,
        random_spawn_lane_index=bool(sim["ego"].get("random_spawn_lane", True)),
        vehicle_config=dict(
            spawn_velocity=[sim["ego"]["spawn_speed_km_h"] / 3.6, 0.0],
            spawn_velocity_car_frame=True,
            lidar=dict(num_lasers=0, distance=50),  # IDM finds neighbours through the lidar's range query only
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
        agent_policy=agent_policy or EnvInputPolicy,
        show_logo=False,
        show_fps=render,
        preload_models=False,
    )
    md.update(sim.get("metadrive_overrides") or {})  # raw MetaDrive config passthrough (advanced)
    return md


def make_metadrive_env(cfg: dict, render: bool = False, agent_policy=None) -> TopDownMetaDrive:
    return TopDownMetaDrive(make_metadrive_config(cfg, render=render, agent_policy=agent_policy))


def build_road_frame(env: TopDownMetaDrive) -> RoadFrame:
    """Frenet frame of the current episode's map along the ego's navigation route (rebuild after every reset:
    each seed has its own map)."""
    m = env.engine.current_map
    return RoadFrame.from_metadrive(m.road_network, env.agent.navigation.checkpoints,
                                    m.config["lane_num"], m.config["lane_width"])
