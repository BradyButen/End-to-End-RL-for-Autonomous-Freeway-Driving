"""DrivingEnv: the one Gymnasium env every method runs in.

One RL step = one decision:

    decision = method.decide(action)              # M1: the controls; M2: target lane + speed; M3: target lane
    for j in range(method.period):                # M1 1, M2 5, M3 10 control steps (0.1 s each)
        u, diag = method.control()                # [steering, throttle_brake] (+ controller diagnostics)
        MetaDrive step -> ego / cars -> reward -> log
        if the episode ended: break
    return obs, reward summed over the decision, ...

Same observation and reward for every method:
  * obs (ground truth, like a tracking stack's output; ``observation_space``):
      "ego":   EGO_FEATURES (speed, position in the road and in the lane, heading error, last commands, the target
               lane relative to the current one (M2 / M3 lane changes in progress, 0 for M1), time left)
      "lanes": for the lanes left of / at / right of the ego: exists, then the nearest car ahead and behind in that
               lane (LANE_SLOT each), out to observation.range (100 m)
  * reward = reward.py (configs/r5.yaml by default), summed undiscounted over the decision's control steps.

A method is a small class (m1.py, m2.py, m3.py) with ``period``, ``action_space``, ``reset()``, ``decide(action)``
and ``control()``; it reads the world from the env (``env.ego``, ``env.objects``, ``env.road``).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from handoff.config import kmh
from handoff.logs import EpisodeRecorder, termination_reason
from handoff.reward import Reward
from handoff.sim import (
    EgoState,
    ObjectList,
    RoadFrame,
    TrafficSpeeds,
    box_gap,
    build_road_frame,
    extract_ego,
    extract_objects,
    frenet_velocity,
    make_metadrive_env,
    spawn_car,
    time_to_collision,
)

# ======================================================================================= observation
EGO_FEATURES = (
    "v",               # speed / v_max
    "d",               # lateral position, road-centred / half the road width
    "lane_offset",     # offset from the nearest lane centre / lane width (+ left)
    "psi_err",         # heading error to the road (rad)
    "steering",        # last applied steering command
    "throttle_brake",  # last applied throttle / brake command
    "target_lane",     # target lane - current lane, in lanes (+ left); M2 / M3 mid lane change, M1 always 0
    "t_left",          # time left / episode cap (sim.horizon): with a ~50 s discount the value depends on it
)
LANE_SLOT = (          # the nearest car ahead (or behind) in one lane
    "present",         # 1 = a car, 0 = none within range (then: gap 1, dd the lane's offset, dv 0, inv_ttc 0)
    "gap",             # bumper-to-bumper distance along the road / observation.range, 0 when overlapping
    "dd",              # its lateral offset from the ego / lane width (+ left): shows cars mid lane change
    "dv",              # its speed along the road - the ego's / v_max (+ = faster)
    "inv_ttc",         # closing speed / gap (1/s, 0 when not closing), clipped to 5
)
LANES = (-1, 0, 1)     # lane index offsets from the ego's lane: left, own, right
LANE_OVERLAP = 0.3     # m: a car occupies a lane when it reaches this far into it (straddling cars occupy both)


def observation_space(cfg: dict) -> gym.spaces.Dict:
    """The policy's observation space (also used by scripts/bc.py to build the same network)."""
    n_lanes = len(LANES) * (1 + 2 * len(LANE_SLOT))
    return gym.spaces.Dict({
        "ego": gym.spaces.Box(-5.0, 5.0, shape=(len(EGO_FEATURES),), dtype=np.float32),
        "lanes": gym.spaces.Box(-5.0, 5.0, shape=(n_lanes,), dtype=np.float32),
    })


def lane_features(ego: EgoState, objects: ObjectList, road: RoadFrame, radius: float, v_max: float) -> np.ndarray:
    """Per lane in LANES: [exists, ahead slot, behind slot] (LANE_SLOT each), relative to the ego's nearest lane."""
    w = road.lane_width
    own = road.nearest_lane(ego.d)
    ego_sdot = frenet_velocity(ego, road)[0]
    out = []
    for rel in LANES:
        i = own + rel
        empty = [0.0, 1.0, float(-rel), 0.0, 0.0]          # lane index +1 = one lane to the right = d - w
        if not 0 <= i < len(road.lane_centers):
            out += [0.0] + empty + empty
            continue
        c = road.lane_centers[i]
        near = {1: None, -1: None}                         # ahead, behind: (gap, car)
        for o in objects:
            if abs(o.d - c) >= 0.5 * w + 0.5 * o.width - LANE_OVERLAP:
                continue
            side = 1 if o.s > ego.s else -1
            gap = max(abs(o.s - ego.s) - 0.5 * (o.length + ego.length), 0.0)
            if near[side] is None or gap < near[side][0]:
                near[side] = (gap, o)
        out.append(1.0)
        for side in (1, -1):
            if near[side] is None:
                out += empty
                continue
            gap, o = near[side]
            sdot = frenet_velocity(o, road)[0]
            closing = (ego_sdot - sdot) if side > 0 else (sdot - ego_sdot)
            out += [1.0, min(gap / radius, 1.0), (o.d - ego.d) / w, (sdot - ego_sdot) / v_max,
                    min(max(closing, 0.0) / max(gap, 0.5), 5.0)]
    return np.asarray(out, np.float32)


class Method:
    """Base class of M1 / M2 / M3 / B0: one RL action -> ``decide`` once, then ``control`` every control step."""

    def __init__(self, env: DrivingEnv, period: int, action_space: gym.Space):
        self.env = env
        self.period = int(period)
        self.action_space = action_space
        self.reference = None     # what the controller tracks (M2 / M3), drawn by the viewer and logged

    def target_lane_offset(self) -> int:
        """Target lane - current lane (observation feature ``target_lane``); M2 / M3 override it."""
        return 0

    @staticmethod
    def agent_policy():
        """MetaDrive policy class driving the ego; None = our action drives it (B0 uses IDM)."""
        return None

    def reset(self) -> None:
        self.reference = None

    def decide(self, action: np.ndarray) -> dict:
        """Turn the RL action into this decision's plan. Returns a dict for logs / viewer; keys used by the logs:
        ``vector`` (decoded action), ``k`` (M3 waypoint count), ``waypoints`` (M3 (k, 2) s, d)."""
        raise NotImplementedError

    def control(self) -> tuple[np.ndarray, dict]:
        """[steering, throttle_brake] in [-1, 1]^2 for this control step, and a diagnostics dict (see
        ``DrivingEnv._log_step`` for the keys that are logged)."""
        raise NotImplementedError


def method_class(name: str) -> type[Method]:
    from handoff.m1 import B0, M1
    from handoff.m2 import M2
    from handoff.m3 import M3

    return {"m1": M1, "m2": M2, "m3": M3, "b0": B0}[name]


class DrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
                 render: bool = False):
        super().__init__()
        self.cfg, self.rank, self.is_eval = cfg, rank, is_eval
        sim = cfg["sim"]
        self.dt = float(sim["control_dt"])
        self.v_max = kmh(sim["ego"]["max_speed_km_h"])
        self.v_target = kmh(sim["target_speed_km_h"])
        self.v_cap = kmh(sim["v_cap_km_h"])
        self.horizon_steps = max(int(sim["horizon"]), 1)
        self.radius = float(sim["perception_range"])              # controllers, reward, logs: env.objects
        self.obs_range = float(cfg["observation"]["range"])        # observation: env.seen
        self.max_steering_rad = float(np.deg2rad(sim["ego"]["max_steering_deg"]))

        cls = method_class(cfg["method"]["name"])
        self.md = make_metadrive_env(cfg, render=render, agent_policy=cls.agent_policy())
        self.traffic = TrafficSpeeds(cfg["scenario"]["traffic"])
        ego = sim["ego"]
        self.reward = Reward(cfg["reward"], self.v_max, float(ego["front_wheelbase"]) + float(ego["rear_wheelbase"]),
                             self.max_steering_rad)
        log = cfg["logging"]
        self.full_log_eval = bool(log["full_log_eval"])
        self.full_every = int(log["train_full_log_every"])
        self.tube_every = max(1, int(log["tube_every"]))
        self.recorder = EpisodeRecorder((Path(run_dir) / "episodes") if run_dir else None,
                                        int(log["max_logged_objects"]), self.dt, self.max_steering_rad,
                                        weights=self.reward.weights)
        seeds = cfg["scenario"]["seeds"]
        self.train_seeds = (int(seeds["train_low"]), int(seeds["train_high"]))
        self._seed_rng = np.random.default_rng([int(cfg["experiment"]["seed"]), rank, int(is_eval)])

        self.road: RoadFrame | None = None
        self.ego: EgoState | None = None
        self.objects: ObjectList = []   # cars within perception_range (controllers, reward, logs), nearest first
        self.seen: ObjectList = []      # cars within observation.range (the policy's observation)
        self.scripted_cars: list = []   # (MetaDrive vehicle, speed m/s) spawned by validation; perceived like traffic
        self.decision: dict = {}        # the current decision (method.decide output)
        self.on_step = None             # viewer hook: callable(env, diag) after every control step
        self.total_control_steps = self.total_decisions = self.total_episodes = 0
        self.episode_control_steps = self.episode_decisions = 0
        self.episode_open = False

        self.method = cls(cfg, self)
        self.action_space = self.method.action_space
        self.observation_space = observation_space(cfg)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """``options``: ``episode_seed`` (MetaDrive seed = map + traffic; default: drawn from the training range),
        ``tag`` (prefix of the episode id)."""
        if seed is not None:
            self._seed_rng = np.random.default_rng([int(seed), self.rank, int(self.is_eval)])
        options = options or {}
        ep_seed = options.get("episode_seed")
        self.episode_seed = int(self._seed_rng.integers(*self.train_seeds) if ep_seed is None else ep_seed)
        self.scripted_cars = []
        self.md.reset(seed=self.episode_seed)
        self.traffic.on_reset(self.md)
        self.road = build_road_frame(self.md)   # each seed has its own map
        self.reward.reset(self.road)
        self.episode_control_steps = self.episode_decisions = 0
        self.ego = extract_ego(self.md, self.road, 0.0)
        self._extract_objects()
        kind = "eval" if self.is_eval else "train"
        full = (self.is_eval and self.full_log_eval) or (
            not self.is_eval and self.full_every > 0 and self.total_episodes % self.full_every == 0)
        tag = options.get("tag")
        self.episode_id = ((f"{tag}_" if tag else "")
                           + f"{kind}_r{self.rank}_e{self.total_episodes:06d}_s{self.episode_seed}")
        self.recorder.start(self.episode_id, self.episode_seed, full,
                            meta={"method": self.cfg["experiment"]["method"],
                                  "reward_variant": self.cfg["experiment"]["reward_variant"],
                                  "is_eval": int(self.is_eval), "rank": self.rank})
        self.total_episodes += 1
        self.episode_open = True
        self.decision = {}
        self.method.reset()
        return self._obs(), {"episode_seed": self.episode_seed, "episode_id": self.episode_id}

    def step(self, action):
        action = np.asarray(action)
        self.decision = self.method.decide(action)
        start = self.episode_control_steps
        total, n = 0.0, 0
        for _ in range(self.method.period):
            u, diag = self.method.control()
            obs, r, terminated, truncated, info = self._control_step(u, diag)
            total += r
            n += 1
            if terminated or truncated:
                break
        self.recorder.record_decision(action, self.decision, total, n, start, self.method.reference)
        self.episode_decisions += 1
        self.total_decisions += 1
        info["decision_control_steps"] = n
        info["counters"] = self.counters()
        if terminated or truncated:
            info["termination"] = termination_reason(info, truncated and not terminated)
            info["episode_summary"] = self.end_episode(info["termination"])
        return obs, total, terminated, truncated, info

    def end_episode(self, reason: str) -> dict:
        """Close the episode's log (also for scripts that stop an episode early) -> episode summary."""
        self.episode_open = False
        return self.recorder.finish(reason, self.counters())

    def close(self):
        self.md.close()

    def counters(self) -> dict:
        return {"total_control_steps": self.total_control_steps, "total_decisions": self.total_decisions,
                "total_sim_time": self.total_control_steps * self.dt, "total_episodes": self.total_episodes}

    def add_scripted_car(self, s: float, d: float, speed_kmh: float = 0.0):
        """Validation: a car at (s, d); speed 0 = static, otherwise held at that speed along the road."""
        veh = spawn_car(self.md, self.road, s, d, static=speed_kmh <= 0)
        self.scripted_cars.append((veh, kmh(speed_kmh)))
        self._drive_scripted_cars()
        self._extract_objects()
        return veh

    # ------------------------------------------------------------------ one control step
    def _control_step(self, u: np.ndarray, diag: dict):
        prev = self.ego
        u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        _, _, terminated, truncated, info = self.md.step(u)
        info = dict(info)
        self.traffic.apply(self.md)
        self._drive_scripted_cars()
        if diag.get("ebrake_terminal"):          # M3: emergency brake ends the run, scored as a collision
            info["emergency_brake"] = True
            terminated = True
        self.episode_control_steps += 1
        self.total_control_steps += 1
        self.ego = extract_ego(self.md, self.road, self.episode_control_steps * self.dt)
        self._extract_objects()
        total, raw, weighted = self.reward.step(prev, self.ego, info, self.objects)
        self._log_step(u, diag, total, raw, weighted)
        info["reward_terms"] = raw
        if self.on_step is not None:
            self.on_step(self, diag)
        return self._obs(), total, bool(terminated), bool(truncated), info

    def _log_step(self, u, diag: dict, total: float, raw: dict, weighted: dict) -> None:
        gaps = [box_gap(self.ego, o) for o in self.objects]
        ttcs = [time_to_collision(self.ego, o, self.road) for o in self.objects]
        scalars = {
            "reward": total,
            "tracking_error": diag.get("tracking_error"),
            "solve_time": diag.get("solve_time", 0.0),
            "cpu_time": diag.get("cpu_time", 0.0),
            "fallback": float(bool(diag.get("fallback"))),
            "gated_brake": float(bool(diag.get("gated_brake"))),
            "min_gap": min(gaps) if gaps else self.radius,
            "min_ttc": min(ttcs) if ttcs else np.inf,
            "ref_d": diag.get("ref_d"),
            "ref_v": diag.get("ref_v"),
            "ref_psi": diag.get("ref_psi"),
            "cmd_steering": u[0],
            "cmd_throttle": u[1],
            "decision_index": self.episode_decisions,
        }
        status = str(diag.get("status", ""))
        if diag.get("fallback"):
            status = f"fallback:{diag.get('fallback_reason', '')}:{diag.get('fallback_mode', '')}"
        tube = diag.get("tube") if self.episode_control_steps % self.tube_every == 0 else None
        self.recorder.record_step(self.ego, self.objects, raw, weighted, scalars, status,
                                  pred_path=diag.get("pred_xy"), tube=tube)

    def _drive_scripted_cars(self) -> None:
        for veh, v in self.scripted_cars:
            if v > 0:
                h = float(self.road.heading_at(self.road.to_frenet(*veh.position)[0]))
                veh.set_velocity([v * np.cos(h), v * np.sin(h)])

    def _extract_objects(self) -> None:
        """env.seen (observation range) and env.objects (perception range: what the controllers see)."""
        e = self.ego
        self.seen = extract_objects(self.md, self.road, max(self.obs_range, self.radius),
                                    [veh for veh, _ in self.scripted_cars])
        self.objects = [o for o in self.seen if np.hypot(o.x - e.x, o.y - e.y) <= self.radius]

    def _obs(self) -> dict:
        e, road = self.ego, self.road
        nearest = road.lane_centers[road.nearest_lane(e.d)]
        values = {"v": e.v / self.v_max, "d": (e.d - road.d_center) / road.half_width,
                  "lane_offset": (e.d - nearest) / road.lane_width, "psi_err": e.psi_err, "steering": e.steering,
                  "throttle_brake": e.throttle_brake, "target_lane": float(self.method.target_lane_offset()),
                  "t_left": 1.0 - self.episode_control_steps / self.horizon_steps}
        return {"ego": np.clip(np.asarray([values[f] for f in EGO_FEATURES], np.float32), -5.0, 5.0),
                "lanes": np.clip(lane_features(e, self.seen, road, self.obs_range, self.v_max), -5.0, 5.0)}


def make_env(cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
             render: bool = False) -> DrivingEnv:
    return DrivingEnv(cfg, rank=rank, run_dir=run_dir, is_eval=is_eval, render=render)
