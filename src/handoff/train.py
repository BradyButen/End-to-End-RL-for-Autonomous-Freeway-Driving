"""PPO training (Stable-Baselines3, MultiInputPolicy), identical for M1 / M2 / M3.

* Budgets and cadences (total, eval, checkpoint) count CONTROL STEPS (0.1 s of simulated driving, summed over
  envs), so every method gets the same simulated experience whatever its decision period.
* ``max_wall_minutes`` is a wall-clock stop on top, for cluster time limits: after that long the training loop
  stops, then the final eval and final.zip still run, so leave room for them before the job's limit.
* gamma and GAE lambda are given per 0.1 s and scaled to each method's decision period (x ** (T_decision / 0.1 s)),
  so every method has the same lookahead and credit window in seconds.
* Failure handling: a crashed env or eval worker ends the job (FAILED.txt, checkpoints/crash.zip, all child
  processes killed, non-zero exit). A watchdog ends it with exit code 3 if nothing progresses for
  ``training.watchdog_minutes``; env start-up, every env step, every finished eval episode and the final shutdown
  count as progress.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

from handoff.evaluate import ParallelEvaluator, aggregate, policy_spec  # noqa: E402
from handoff.logs import CsvAppender, make_run_dir  # noqa: E402

EVAL_LOG_METRICS = ("success", "collision", "off_road", "timeout", "return", "mean_speed",
                    "min_gap", "min_ttc", "rms_long_jerk", "rms_lat_accel", "rms_steer_rate", "tracking_error_rms",
                    "solve_time_mean", "solve_time_p95", "fallback_count", "lane_changes",
                    "time_to_complete")


# ======================================================================================= settings
def _per_decision(cfg: dict, per_step_value: float) -> float:
    t_dec = int(cfg["method"].get("decision_period", 1)) * float(cfg["sim"]["control_dt"])
    return float(per_step_value) ** (t_dec / 0.1)


def method_gamma(cfg: dict) -> float:
    return _per_decision(cfg, cfg["training"]["gamma_ref"])      # M1 0.999, M2 0.995, M3 0.990


def method_gae_lambda(cfg: dict) -> float:
    return _per_decision(cfg, cfg["training"]["gae_lambda_ref"])  # M1 0.995, M2 0.975, M3 0.951


def ppo_kwargs(cfg: dict) -> dict:
    kw = dict(cfg["ppo"])
    kw["policy_kwargs"] = dict(kw.get("policy_kwargs") or {})
    kw["gamma"] = method_gamma(cfg)
    kw["gae_lambda"] = method_gae_lambda(cfg)
    return kw


# ======================================================================================= process handling
def _env_thunk(cfg: dict, rank: int, run_dir: Path):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor

        from handoff.env import make_env

        return Monitor(make_env(cfg, rank=rank, run_dir=run_dir, is_eval=False))

    return _thunk


def _rss_gb() -> float:
    """Resident memory (GB) of this process + every child (env and eval workers), from /proc (Linux)."""
    total = 0
    for pid in [os.getpid()] + [p.pid for p in mp.active_children()]:
        try:
            with open(f"/proc/{pid}/status") as f:
                total += next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))   # kB
        except (OSError, StopIteration, ValueError):
            pass
    return total / 1024**2


def _kill_children(timeout: float = 10.0) -> None:
    """Terminate (then kill) every child process (env and eval workers). Never blocks for long."""
    kids = mp.active_children()
    for p in kids:
        p.terminate()
    deadline = time.time() + timeout
    for p in kids:
        p.join(max(0.1, deadline - time.time()))
        if p.is_alive():
            p.kill()
            p.join(1.0)


class Watchdog:
    """Kills the job if nothing calls ``beat()`` for ``minutes`` (a hung worker blocks the main process forever)."""

    def __init__(self, minutes: float, run_dir: Path):
        self.limit = float(minutes) * 60.0
        self.run_dir = run_dir
        self.last = time.time()
        self._stop = threading.Event()
        if self.limit > 0:
            threading.Thread(target=self._run, daemon=True).start()

    def beat(self) -> None:
        self.last = time.time()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(30.0):
            idle = time.time() - self.last
            if idle > self.limit:
                msg = f"watchdog: no training progress for {idle / 60:.1f} min; killing workers and exiting\n"
                print(msg, file=sys.stderr, flush=True)
                (self.run_dir / "FAILED.txt").write_text(msg)
                _kill_children()
                os._exit(3)


# ======================================================================================= warm start
CRITIC_PREFIXES = ("mlp_extractor.value_net.", "value_net.")   # policy state-dict keys that belong to the critic


def actor_parameters(policy) -> list:
    """Parameters that set the action: features extractor, policy MLP, action head, log_std (continuous actions).
    Frozen during the critic warm-up."""
    params = (list(policy.features_extractor.parameters()) + list(policy.mlp_extractor.policy_net.parameters())
              + list(policy.action_net.parameters()))
    if hasattr(policy, "log_std"):
        params.append(policy.log_std)
    if not policy.share_features_extractor:
        params += list(policy.pi_features_extractor.parameters())
    return params


def warm_start(model, cfg: dict) -> str | None:
    """Load the actor of ``training.init_policy`` into ``model`` (critic and log_std stay as initialised; see
    ``init_exploration``). Returns a one-line description, or None when no warm start is set."""
    from handoff.config import REPO_ROOT

    tr = cfg["training"]
    path = tr.get("init_policy")
    if not path:
        return None
    path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"training.init_policy: {path} does not exist")
    src = PPO.load(str(path), device="cpu").policy.state_dict()
    dst = model.policy.state_dict()
    actor = {k: v for k, v in src.items() if not k.startswith(CRITIC_PREFIXES) and k != "log_std"}
    missing = [k for k in dst if not k.startswith(CRITIC_PREFIXES) and k != "log_std" and k not in actor]
    bad = [k for k, v in actor.items() if k not in dst or dst[k].shape != v.shape]
    if missing or bad:
        raise ValueError(f"training.init_policy {path} does not match this policy (wrong method / observation?): "
                         f"missing {missing[:3]}, mismatched {bad[:3]}")
    model.policy.load_state_dict(actor, strict=False)
    return f"warm start: actor from {path} ({len(actor)} tensors)"


def init_exploration(model, cfg: dict) -> str | None:
    """Initial exploration, by action type. Returns a one-line description (None if nothing was set).

    Box (M1): ``training.init_log_std``, one number or one per action dimension (null = ppo.policy_kwargs.
      log_std_init), so each dimension's noise can be sized from what it does to the car.
    Discrete / MultiDiscrete (M2, M3): ``training.init_default_prob``, one number or one per action component: the
      initial probability of index 0 (keep lane / hold speed), the rest shared equally by the other choices. Set
      through the action head's bias (its weights start near 0, so the initial policy is the same everywhere)."""
    import gymnasium as gym

    tr, policy = cfg["training"], model.policy
    space = model.action_space
    if isinstance(space, gym.spaces.Box):
        init = tr.get("init_log_std")
        if init is None:
            return None
        n = policy.log_std.numel()
        vals = [float(x) for x in init] if isinstance(init, (list, tuple)) else [float(init)] * n
        if len(vals) != n:
            raise ValueError(f"training.init_log_std {init}: the action has {n} dimensions")
        with torch.no_grad():
            policy.log_std.copy_(torch.as_tensor(vals, dtype=policy.log_std.dtype))
        return f"initial action std {policy.log_std.exp().detach().cpu().numpy().round(3).tolist()}"
    nvec = [int(space.n)] if isinstance(space, gym.spaces.Discrete) else [int(x) for x in space.nvec]
    init = tr.get("init_default_prob")
    if init is None:
        return None
    probs = [float(x) for x in init] if isinstance(init, (list, tuple)) else [float(init)] * len(nvec)
    if len(probs) != len(nvec):
        raise ValueError(f"training.init_default_prob {init}: the action has {len(nvec)} components")
    with torch.no_grad():
        bias, i = policy.action_net.bias, 0
        for n, p in zip(nvec, probs):
            bias[i:i + n] = 0.0
            bias[i] = float(np.log(p * (n - 1) / (1.0 - p)))   # softmax: p for index 0, (1 - p) / (n - 1) others
            i += n
    return f"initial probability of the default action (index 0) {probs}"


# ======================================================================================= callback
class TrainingCallback(BaseCallback):
    """Episode rows -> episodes.csv; checkpoints, evaluations (step 0, every eval.freq_control_steps, final) and
    the budget stop, all counted in control steps. best.zip = highest eval success, ties by return."""

    def __init__(self, cfg: dict, run_dir: Path, verbose: int, heartbeat):
        super().__init__(verbose)
        self.cfg, self.run_dir, self.heartbeat = cfg, run_dir, heartbeat
        tr, ev = cfg["training"], cfg["eval"]
        self.dt = float(cfg["sim"]["control_dt"])
        self.budget = int(float(tr["total_control_steps"]))
        self.eval_seeds = list(cfg["scenario"]["seeds"]["eval"][: int(ev["n_episodes"])])
        self.eval_freq, self.ckpt_freq = int(ev["freq_control_steps"]), int(tr["checkpoint_freq_control_steps"])
        self.n_eval_workers, self.deterministic = int(ev.get("n_workers", 1)), bool(ev.get("deterministic", True))
        self.progress_s = float(tr.get("progress_seconds") or 0)
        wall = tr.get("max_wall_minutes")
        self.wall_limit_s = float(wall) * 60.0 if wall else None   # stop the training loop after this long
        self.episodes_csv = CsvAppender(run_dir / "episodes.csv")
        self.control_steps = self.decisions = self.episodes = self.rollouts = 0
        self.next_eval = self.eval_freq if self.eval_freq > 0 else None
        self.next_ckpt = self.ckpt_freq if self.ckpt_freq > 0 else None
        self.best = (-np.inf, -np.inf)
        self.evaluator: ParallelEvaluator | None = None
        self.recent: list[dict] = []
        # critic warm-up (warm start only): the actor stays frozen for this many control steps
        self.warmup = int(float(tr.get("critic_warmup_control_steps") or 0)) if tr.get("init_policy") else 0
        self.actor_frozen = False

    def _freeze_actor(self, frozen: bool) -> None:
        for p in actor_parameters(self.model.policy):
            p.requires_grad_(not frozen)
        self.actor_frozen = frozen
        if self.verbose:
            state = (f"frozen: critic warm-up for {self.warmup} control steps" if frozen
                     else f"unfrozen at {self.control_steps} control steps: full PPO from here")
            print(f"[warm start] actor {state}", flush=True)

    def _counters(self) -> dict:
        return {"train_control_steps": self.control_steps, "train_decisions": self.decisions,
                "train_sim_time": self.control_steps * self.dt}

    def _on_training_start(self) -> None:
        self.t_start = time.time()
        self.next_progress = self.t_start + self.progress_s
        if self.warmup > 0:
            self._freeze_actor(True)
        if self.next_eval is not None:
            self.evaluator = ParallelEvaluator(self.cfg, self.run_dir, self.n_eval_workers, policy_spec(self.model))
            self._evaluate()   # step 0 (untrained policy) anchors the learning curve

    def _on_training_end(self) -> None:
        if self.evaluator is not None:
            self._evaluate(final=True)
            self.evaluator.close()
        self._save("final")

    def _on_rollout_end(self) -> None:
        self.rollouts += 1     # a PPO update follows every rollout
        if self.actor_frozen and self.control_steps >= self.warmup:
            self._freeze_actor(False)   # takes effect from the update that follows this rollout
        self.logger.record("train/actor_frozen", float(self.actor_frozen))

    def _on_step(self) -> bool:
        self.heartbeat()
        rows = []
        for info in self.locals["infos"]:
            self.control_steps += int(info.get("decision_control_steps", 1))
            self.decisions += 1
            summ = info.get("episode_summary")
            if summ is not None:
                self.episodes += 1
                rows.append({**summ, **self._counters()})
                self.recent = (self.recent + [summ])[-100:]
        self.episodes_csv.write(rows)
        if rows:
            for k in ("success", "collision", "off_road", "return", "mean_speed"):
                self.logger.record(f"train/{k}", float(np.mean([r[k] for r in self.recent])))
        self.logger.record("counters/control_steps", self.control_steps)
        self.logger.record("counters/decisions", self.decisions)
        self.logger.record("counters/sim_time", self.control_steps * self.dt)
        self.logger.record("counters/episodes", self.episodes)

        if self.next_ckpt is not None and self.control_steps >= self.next_ckpt:
            self._save(f"ckpt_{self.control_steps:010d}")
            self.next_ckpt += self.ckpt_freq
        if self.next_eval is not None and self.control_steps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.eval_freq
        if self.progress_s > 0 and time.time() >= self.next_progress:
            self._print_progress()
            self.next_progress = time.time() + self.progress_s
        if self.wall_limit_s is not None and time.time() - self.t_start > self.wall_limit_s:
            print(f"[stop] wall-clock limit {self.wall_limit_s / 3600:.2f} h reached at {self.control_steps} control "
                  "steps; final eval + save follow", flush=True)
            return False
        return self.control_steps < self.budget

    def _save(self, name: str) -> None:
        """Policy + reward-normalisation statistics (the latter only needed to resume training)."""
        ckpt = self.run_dir / "checkpoints"
        self.model.save(ckpt / f"{name}.zip")
        env = self.model.get_env()
        if hasattr(env, "ret_rms"):
            env.save(str(ckpt / f"vecnormalize_{name.removeprefix('ckpt_')}.pkl"))

    def _print_progress(self) -> None:
        el = time.time() - self.t_start
        msg = (f"[{self.cfg['method']['name']} {el / 60:5.1f} min] {self.control_steps / 1e3:7.1f}k steps "
               f"({self.control_steps / max(el, 1e-9):4.0f}/s, {self.control_steps * self.dt / 3600:.2f} h driven) | "
               f"{self.episodes} ep | {self.rollouts} rollouts | mem {_rss_gb():.1f} GB")
        recent = self.recent[-50:]
        if recent:
            crash = np.mean([max(r["collision"], r["off_road"]) for r in recent])
            msg += (f" | last {len(recent)} ep: return {np.mean([r['return'] for r in recent]):7.1f}, "
                    f"success {np.mean([r['success'] for r in recent]):.2f}, crash {crash:.2f}, "
                    f"{np.mean([r['mean_speed'] for r in recent]) * 3.6:4.1f} km/h, "
                    f"{np.mean([r['distance'] for r in recent]):4.0f} m, "
                    f"{np.mean([r['sim_time'] for r in recent]):4.1f} s")
        print(msg, flush=True)

    def _evaluate(self, final: bool = False) -> None:
        tag = f"eval_{self.control_steps:010d}" + ("_final" if final else "")
        if self.verbose:
            print(f"[eval {tag}] running {len(self.eval_seeds)} episodes on {self.n_eval_workers} workers ...",
                  flush=True)
        # CPU copy: eval workers run on CPU, and CUDA tensors must not be pickled across processes
        state = {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()}
        self.heartbeat()
        summaries = self.evaluator.evaluate(state, self.eval_seeds, self.deterministic, tag=tag,
                                            heartbeat=self.heartbeat)
        self.heartbeat()
        self.episodes_csv.write([{**s, **self._counters(), "eval_tag": tag} for s in summaries])
        agg = aggregate(summaries)
        for k in EVAL_LOG_METRICS:
            if k in agg:
                self.logger.record(f"eval/{k}", agg[k])
        key = (agg.get("success", 0.0), agg.get("return", -np.inf))
        if key > self.best:
            self.best = key
            self.model.save(self.run_dir / "checkpoints" / "best.zip")
            (self.run_dir / "checkpoints" / "best.txt").write_text(f"{tag} success={key[0]:.3f} return={key[1]:.2f}\n")
        if self.verbose:
            print(f"[eval {tag}] success={agg.get('success', 0):.2f} collision={agg.get('collision', 0):.2f} "
                  f"off_road={agg.get('off_road', 0):.2f} return={agg.get('return', 0):.1f} "
                  f"speed={agg.get('mean_speed', 0) * 3.6:.1f} km/h distance={agg.get('distance', 0):.0f} m")
        self.logger.dump(self.num_timesteps)


# ======================================================================================= entry point
def train(cfg: dict, run_dir: str | Path | None = None, verbose: int = 1) -> Path:
    if cfg["method"]["name"] == "b0":
        raise ValueError("B0 is not trained; watch it with scripts/view.py live --method b0")
    run_dir = make_run_dir(cfg, run_dir)
    tr = cfg["training"]
    seed = int(cfg["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if tr.get("torch_threads"):
        torch.set_num_threads(int(tr["torch_threads"]))
    watchdog = Watchdog(tr.get("watchdog_minutes", 15), run_dir)   # from here on: covers env start-up too
    n_envs = int(tr["n_envs"])
    fns = [_env_thunk(cfg, i, run_dir) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns, start_method="spawn")   # one MetaDrive per process
    vec.seed(seed)
    if tr.get("normalize_reward", False):
        # PPO sees rewards divided by a running std of the discounted return, clipped; observations untouched, and
        # the episode logs / train/* / eval/* metrics stay raw (Monitor + recorder sit inside the env).
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=method_gamma(cfg),
                           clip_reward=float(tr.get("reward_clip", 10.0)))
    model = PPO("MultiInputPolicy", vec, seed=seed, device=tr.get("device", "auto"), verbose=0, **ppo_kwargs(cfg))
    ws = warm_start(model, cfg)
    expl = init_exploration(model, cfg)
    if ws:
        (run_dir / "warm_start.txt").write_text(ws + (f", {expl}" if expl else "") + "\n")
    for line in (ws, expl):
        if line and verbose:
            print(line)
    (run_dir / "device.txt").write_text(f"{model.device} torch={torch.__version__} cuda_available="
                                        f"{torch.cuda.is_available()} threads={torch.get_num_threads()}\n")
    if verbose:
        print(f"PPO device: {model.device} | run dir: {run_dir}")
    outputs = [CSVOutputFormat(str(run_dir / "train_progress.csv"))]
    if verbose:
        outputs.append(HumanOutputFormat(sys.stdout))
    model.set_logger(Logger(str(run_dir), outputs))
    try:
        # stopped by the control-step budget in the callback
        model.learn(total_timesteps=10**12, callback=TrainingCallback(cfg, run_dir, verbose, watchdog.beat),
                    progress_bar=False)
    except BaseException:
        # A crashed worker (e.g. a native segfault) surfaces here as EOFError / BrokenPipeError. No graceful close
        # (it waits on the other workers and can hang): save what we have, kill every child, re-raise (exit != 0).
        tb = traceback.format_exc()
        print(f"training failed:\n{tb}", file=sys.stderr, flush=True)
        (run_dir / "FAILED.txt").write_text(tb)
        try:
            model.save(run_dir / "checkpoints" / "crash.zip")
        except Exception:  # noqa: BLE001
            pass
        watchdog.stop()
        _kill_children()
        raise
    _close_envs(vec)
    watchdog.stop()
    return run_dir


def _close_envs(vec, timeout: float = 120.0) -> None:
    """Close the env workers, but never wait on a stuck one for more than ``timeout`` (checkpoints are saved)."""
    t = threading.Thread(target=vec.close, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"env workers did not close within {timeout:.0f} s; killing them", file=sys.stderr, flush=True)
        _kill_children()
