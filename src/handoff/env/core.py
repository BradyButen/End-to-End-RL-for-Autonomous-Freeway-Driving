"""CoreEnv: the Gymnasium environment shared by all methods (spec §3.1).

Owns the MetaDrive instance and everything common to M1/M2/M3/B0:
  * episode seeding + scenario sampling (fixed map, randomised initial conditions; disjoint train/eval seeds),
  * state extraction (EgoState, ObjectList within R, RoadFrame),
  * RL observation Dict{bev, ego} (spec §6),
  * reward composition (all terms computed + logged every control step, spec §9),
  * logging hooks (EpisodeRecorder, spec §12).

Method wrappers (``methods/<m>/wrapper.py`` via ``env/decision.py``) call ``control_step`` once per control step and
``end_decision`` once per RL decision. ``step`` = one control step + one decision (M1 semantics).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from handoff.config import kmh
from handoff.eval.metrics import termination_reason
from handoff.logging.recorder import EpisodeRecorder
from handoff.rewards.composer import RewardComposer
from handoff.sim.metadrive_env import FreewayMetaDriveEnv, make_metadrive_config
from handoff.sim.road import RoadFrame
from handoff.sim.scenario import ScenarioSpec, make_scenario
from handoff.sim.state import (
    EgoState,
    ObjectList,
    frenet_box_gap,
    make_ego_state,
    make_tracked_object,
    time_to_collision,
)


class CoreEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
                 render: bool = False, agent_policy=None):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.is_eval = is_eval
        self.md = FreewayMetaDriveEnv(make_metadrive_config(cfg, render=render, agent_policy=agent_policy))
        self.scenario = make_scenario(cfg)
        sim = cfg["sim"]
        self.dt = float(sim["control_dt"])
        self.v_max = kmh(sim["ego"]["max_speed_km_h"])
        self.v_target = kmh(sim["target_speed_km_h"])
        self.v_cap = kmh(sim["v_cap_km_h"])
        self.R = float(sim["perception_range"])
        self.max_steering_rad = float(np.deg2rad(sim["ego"]["max_steering_deg"]))

        bev = cfg["observation"]["bev"]
        n_ch = 2 + int(bev["frame_stack"])
        self.ego_features = list(cfg["observation"]["ego_features"])
        self.observation_space = gym.spaces.Dict({
            # channels-first (C, H, W): SB3's NatureCNN reads channels from shape[0] when normalize_images=False
            "bev": gym.spaces.Box(0.0, 1.0, shape=(n_ch, bev["resolution"], bev["resolution"]), dtype=np.float32),
            "ego": gym.spaces.Box(-5.0, 5.0, shape=(len(self.ego_features),), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.road: RoadFrame | None = None
        self.reward_ctx = {"dt": self.dt, "v_max": self.v_max, "v_target": self.v_target, "road": None}
        self.reward = RewardComposer.from_config(cfg["reward"], self.reward_ctx)

        log = cfg["logging"]
        self.full_log_eval = bool(log.get("full_log_eval", True))
        self.full_every = int(log.get("train_full_log_every", 0))
        self.recorder = EpisodeRecorder(
            out_dir=(Path(run_dir) / "episodes") if run_dir else None,
            max_objects=int(log["max_logged_objects"]), dt=self.dt,
            lane_centers_fn=lambda: self.road.lane_centers, lane_width=float(sim["map"]["lane_width"]),
            max_steering_rad=self.max_steering_rad)

        seeds = cfg["scenario"]["seeds"]
        self.train_seed_range = (int(seeds["train_low"]), int(seeds["train_high"]))
        self._seed_rng = np.random.default_rng([int(cfg["experiment"]["seed"]), rank, int(is_eval)])

        # counters (spec §11.2: env steps, decisions, simulated time — all logged)
        self.total_control_steps = 0
        self.total_decisions = 0
        self.total_episodes = 0
        self.episode_control_steps = 0
        self.episode_decisions = 0
        self.ego: EgoState | None = None
        self.objects: ObjectList = []
        self.spec: ScenarioSpec | None = None
        self.overlay_hook = None   # viz: callable(core, decision_ctx, diag) after every control step
        self._done = True

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed_rng = np.random.default_rng([int(seed), self.rank, int(self.is_eval)])
        options = options or {}
        ep_seed = options.get("episode_seed")
        if ep_seed is None:
            ep_seed = int(self._seed_rng.integers(*self.train_seed_range))
        self.episode_seed = int(ep_seed)
        self.spec = self.scenario.sample(np.random.default_rng(self.episode_seed), self.episode_seed)
        raw_obs, _ = self.md.reset_with_spec(self.spec)
        if self.road is None:
            self.road = self.md.road_frame
            self.reward_ctx["road"] = self.road
        self.episode_control_steps = 0
        self.episode_decisions = 0
        self.ego = self._extract_ego()
        self.objects = self._extract_objects()
        self.reward.reset()
        full = (self.is_eval and self.full_log_eval) or (
            not self.is_eval and self.full_every > 0 and self.total_episodes % self.full_every == 0)
        kind = "eval" if self.is_eval else "train"
        tag = options.get("tag")
        base_id = f"{kind}_r{self.rank}_e{self.total_episodes:06d}_s{self.episode_seed}"
        self.episode_id = (f"{tag}_" if tag else "") + base_id
        self.recorder.start(self.episode_id, self.episode_seed, full,
                            meta={"method": self.cfg["experiment"]["method"],
                                  "reward_variant": self.cfg["experiment"]["reward_variant"],
                                  "is_eval": int(self.is_eval), "rank": self.rank})
        self.total_episodes += 1
        self._done = False
        return self._obs(raw_obs), {"episode_seed": self.episode_seed, "episode_id": self.episode_id}

    def step(self, action):
        """Single control step treated as one decision (M1 semantics); prefer the method wrappers."""
        obs, r, term, trunc, info = self.control_step(np.asarray(action, float))
        info = self.end_decision(action, {"vector": action, "k": 0}, r, 1, self.episode_control_steps - 1,
                                 None, term, trunc, info)
        return obs, r, term, trunc, info

    def close(self):
        self.md.close()

    # ------------------------------------------------------------------ method-wrapper API
    def control_step(self, u: np.ndarray, decision: dict | None = None, diag: dict | None = None):
        diag = diag or {}
        prev = self.ego
        u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        raw_obs, _, terminated, truncated, info = self.md.step(u)
        self.episode_control_steps += 1
        self.total_control_steps += 1
        self.ego = self._extract_ego()
        self.objects = self._extract_objects()
        total, raw = self.reward.compute(prev, self.ego, self.objects, info, decision)
        weighted = self.reward.weighted(raw)
        gaps = [frenet_box_gap(self.ego, o) for o in self.objects]
        ttcs = [time_to_collision(self.ego, o, self.road) for o in self.objects]
        extra = {
            "tracking_error": diag.get("tracking_error", np.nan),
            "solve_time": diag.get("solve_time", 0.0),
            "fallback": float(bool(diag.get("fallback", False))),
            "min_gap": min(gaps) if gaps else self.R,
            "min_ttc": min(ttcs) if ttcs else np.inf,
            "ref_d": diag.get("ref_d", np.nan),
            "ref_v": diag.get("ref_v", np.nan),
            "ref_psi": diag.get("ref_psi", np.nan),
            "applied_steering": self.ego.steering,
            "applied_throttle": self.ego.throttle_brake,
            "cmd_steering": u[0],
            "cmd_throttle": u[1],
            "decision_index": self.episode_decisions,
            "tube_rebuilt": float(bool(diag.get("tube_rebuilt", False))),
            "tube_blocked": diag.get("tube_blocked", np.nan),
        }
        status = str(diag.get("status", ""))
        if diag.get("fallback"):
            status = f"fallback:{diag.get('fallback_reason', '')}:{diag.get('fallback_mode', '')}"
        self.recorder.record_control(self.ego, self.objects, raw, weighted, total, extra,
                                     pred_path=diag.get("pred_xy"), status=status, tube=diag.get("tube_full"))
        info = dict(info)
        info["reward_terms"] = raw
        if self.overlay_hook is not None:
            self.overlay_hook(self, decision, diag)
        return self._obs(raw_obs), float(total), bool(terminated), bool(truncated), info

    def end_decision(self, raw_action, decoded: dict, reward: float, n_steps: int, start_step: int,
                     reference, terminated: bool, truncated: bool, info: dict) -> dict:
        self.episode_decisions += 1
        self.total_decisions += 1
        self.recorder.record_decision(raw_action, decoded, reward, n_steps, start_step, reference)
        info["decision_control_steps"] = n_steps
        info["counters"] = self.counters()
        if terminated or truncated:
            reason = termination_reason(info, truncated and not terminated)
            info["termination"] = reason
            info["episode_summary"] = self.recorder.finish(reason, self.counters())
            self._done = True
        return info

    def counters(self) -> dict:
        return {"total_control_steps": self.total_control_steps, "total_decisions": self.total_decisions,
                "total_sim_time": self.total_control_steps * self.dt, "total_episodes": self.total_episodes}

    # ------------------------------------------------------------------ extraction
    def _extract_ego(self) -> EgoState:
        v = self.md.agent
        x, y = v.position
        return make_ego_state(self.episode_control_steps * self.dt, x, y, v.heading_theta, v.speed,
                              float(v.steering), float(v.throttle_brake), v.LENGTH, v.WIDTH, self.road)

    def _extract_objects(self) -> ObjectList:
        ego = self.md.agent
        ex, ey = ego.position
        roles = self.md.traffic_roles
        out = []
        for veh in self.md.engine.traffic_manager._traffic_vehicles:
            x, y = veh.position
            dist = float(np.hypot(x - ex, y - ey))
            if dist > self.R:
                continue
            vx, vy = veh.velocity
            out.append((dist, make_tracked_object(veh.name, x, y, veh.heading_theta, vx, vy, veh.LENGTH,
                                                  veh.WIDTH, roles.get(veh.name, "vehicle"), self.road)))
        out.sort(key=lambda p: p[0])
        return [o for _, o in out]

    def _obs(self, raw_obs) -> dict:
        e = self.ego
        feats = []
        for f in self.ego_features:
            if f == "v":
                feats.append(e.v / self.v_max)
            elif f == "d":
                feats.append((e.d - self.road.d_center) / self.road.half_width)
            elif f == "psi_err":
                feats.append(e.psi_err)
            elif f == "steering":
                feats.append(e.steering)
            elif f == "throttle_brake":
                feats.append(e.throttle_brake)
            else:
                raise ValueError(f"unknown ego feature {f}")
        ego_vec = np.clip(np.asarray(feats, np.float32), -5.0, 5.0)
        bev = np.ascontiguousarray(np.transpose(np.asarray(raw_obs, np.float32), (2, 0, 1)))
        return {"bev": bev, "ego": ego_vec}
