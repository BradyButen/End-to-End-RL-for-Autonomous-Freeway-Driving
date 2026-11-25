"""DrivingEnv: the one Gymnasium env every method runs in.

One RL step = one decision:

    decision = method.decide(action)              # M1: the controls; M2 / M3: target lane + target speed
    for j in range(method.period):                # M1 1, M2 5, M3 10 control steps (0.1 s each)
        u, diag = method.control()                # [steering, throttle_brake] (+ controller diagnostics)
        MetaDrive step -> ego / cars -> reward -> log
        if the episode ended: break
    return obs, reward summed over the decision, ...

Every method sees the same world and earns the same reward:
  * observation: ground truth, like the output of a tracking stack (``observation_space``):
      "ego":   EGO_FEATURES. Frozen: data/bc/m1_clone.zip was trained on exactly this vector, so do not reorder or
               extend it
      "lanes": for the lane to the left, the ego's lane and the lane to the right: whether it exists, then the
               nearest car ahead and behind (LANE_SLOT each) within observation.range
      "cmd":   a method's own command state, ``Method.COMMAND_FEATURES`` (M2 / M3: the target speed); absent for
               M1 / B0. New method-specific inputs go here so M1's inputs never change
  * reward: reward.py, summed (undiscounted) over the decision's control steps.

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
    "t_left",          # time left / episode cap: with a ~100 s discount horizon, the value depends on it
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
    """The policy's observation space for this config's method (also used by scripts/warm_start/bc.py)."""
    n_lanes = len(LANES) * (1 + 2 * len(LANE_SLOT))
    spaces = {"ego": gym.spaces.Box(-5.0, 5.0, shape=(len(EGO_FEATURES),), dtype=np.float32),
              "lanes": gym.spaces.Box(-5.0, 5.0, shape=(n_lanes,), dtype=np.float32)}
    n_cmd = len(method_class(cfg["method"]["name"]).COMMAND_FEATURES)
    if n_cmd:
        spaces["cmd"] = gym.spaces.Box(-5.0, 5.0, shape=(n_cmd,), dtype=np.float32)
    return gym.spaces.Dict(spaces)


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

    COMMAND_FEATURES: tuple = ()   # extra command state in the observation ("cmd"); none for M1 / B0
    ACTION_SPACE: gym.Space | None = None   # class-level, so a checkpoint can be checked against it without an
                                            # env; None = not checked (M1 / B0)

    def target_lane_offset(self) -> int:
        """Target lane - current lane (ego feature ``target_lane``); M2 / M3 override it, M1 / B0 0."""
        return 0

    def command_state(self) -> list[float]:
        """Values of COMMAND_FEATURES for the observation's "cmd" entry (M2 / M3)."""
        return []

    @staticmethod
    def agent_policy():
        """MetaDrive policy class driving the ego; None = our action drives it (B0 uses IDM)."""
        return None

    def reset(self) -> None:
        self.reference = None

    def decide(self, action: np.ndarray) -> dict:
        """Turn the RL action into this decision's plan. Returns a dict the viewer shows (``vector`` = the decoded
        action, plus method-specific targets)."""
        raise NotImplementedError

    def control(self) -> tuple[np.ndarray, dict]:
        """[steering, throttle_brake] in [-1, 1]^2 for this control step, and a diagnostics dict for the logs
        (``DrivingEnv._log_step``) and the viewer."""
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
        self.recorder = EpisodeRecorder((Path(run_dir) / "episodes") if run_dir else None, self.dt,
                                        self.max_steering_rad)
        seeds = cfg["scenario"]["seeds"]
        self.train_seeds = (int(seeds["train_low"]), int(seeds["train_high"]))
        self._seed_rng = np.random.default_rng([int(cfg["experiment"]["seed"]), rank, int(is_eval)])

        self.road: RoadFrame | None = None
        self.ego: EgoState | None = None
        self.objects: ObjectList = []   # cars within perception_range (controllers, reward, logs), nearest first
        self.seen: ObjectList = []      # cars within observation.range (the policy's observation)
        self.decision: dict = {}        # the current decision (method.decide output)
        self.on_step = None             # viewer hook: callable(env, diag) after every control step
        self.total_control_steps = self.total_decisions = self.total_episodes = 0
        self.episode_control_steps = self.episode_decisions = 0

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
        self.decision = {}
        self.method.reset()
        return self._obs(), {"episode_seed": self.episode_seed, "episode_id": self.episode_id}

    def step(self, action):
        action = np.asarray(action)
        self.decision = self.method.decide(action)
        total, n = 0.0, 0
        for _ in range(self.method.period):
            u, diag = self.method.control()
            obs, r, terminated, truncated, info = self._control_step(u, diag)
            total += r
            n += 1
            if terminated or truncated:
                break
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
        return self.recorder.finish(reason, self.episode_decisions, self.counters())

    def close(self):
        self.md.close()

    def counters(self) -> dict:
        return {"total_control_steps": self.total_control_steps, "total_decisions": self.total_decisions,
                "total_sim_time": self.total_control_steps * self.dt, "total_episodes": self.total_episodes}

    # ------------------------------------------------------------------ one control step
    def _control_step(self, u: np.ndarray, diag: dict):
        prev = self.ego
        u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        _, _, terminated, truncated, info = self.md.step(u)
        info = dict(info)
        self.traffic.apply(self.md)
        self.episode_control_steps += 1
        self.total_control_steps += 1
        self.ego = extract_ego(self.md, self.road, self.episode_control_steps * self.dt)
        self._extract_objects()
        total, raw, weighted = self.reward.step(prev, self.ego, info, self.objects)
        self._log_step(diag, total, raw, weighted)
        info["reward_terms"] = raw
        if self.on_step is not None:
            self.on_step(self, diag)
        return self._obs(), total, bool(terminated), bool(truncated), info

    def _log_step(self, diag: dict, total: float, raw: dict, weighted: dict) -> None:
        gaps = [box_gap(self.ego, o) for o in self.objects]
        ttcs = [time_to_collision(self.ego, o, self.road) for o in self.objects]
        scalars = {
            "reward": total,
            "tracking_error": diag.get("tracking_error"),
            "solve_time": diag.get("solve_time", 0.0),
            "fallback": float(bool(diag.get("fallback"))),
            "min_gap": min(gaps) if gaps else self.radius,
            "min_ttc": min(ttcs) if ttcs else np.inf,
        }
        self.recorder.record_step(self.ego, raw, weighted, scalars)

    def _extract_objects(self) -> None:
        """env.seen (observation range) and env.objects (perception range: what the controllers see)."""
        e = self.ego
        self.seen = extract_objects(self.md, self.road, max(self.obs_range, self.radius))
        self.objects = [o for o in self.seen if np.hypot(o.x - e.x, o.y - e.y) <= self.radius]

    def _obs(self) -> dict:
        e, road = self.ego, self.road
        nearest = road.lane_centers[road.nearest_lane(e.d)]
        values = {"v": e.v / self.v_max, "d": (e.d - road.d_center) / road.half_width,
                  "lane_offset": (e.d - nearest) / road.lane_width, "psi_err": e.psi_err, "steering": e.steering,
                  "throttle_brake": e.throttle_brake, "target_lane": float(self.method.target_lane_offset()),
                  "t_left": max(1.0 - self.episode_control_steps / self.horizon_steps, 0.0)}
        obs = {"ego": np.clip(np.asarray([values[f] for f in EGO_FEATURES], np.float32), -5.0, 5.0),
               "lanes": np.clip(lane_features(e, self.seen, road, self.obs_range, self.v_max), -5.0, 5.0)}
        if self.method.COMMAND_FEATURES:
            obs["cmd"] = np.clip(np.asarray(self.method.command_state(), np.float32), -5.0, 5.0)
        return obs


def make_env(cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
             render: bool = False) -> DrivingEnv:
    return DrivingEnv(cfg, rank=rank, run_dir=run_dir, is_eval=is_eval, render=render)
