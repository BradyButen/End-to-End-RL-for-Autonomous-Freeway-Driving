"""PPO training (Stable-Baselines3, MultiInputPolicy), identical for M1 / M2 / M3.

Budgets and cadences (total, eval, checkpoint) are counted in CONTROL STEPS (0.1 s of simulated driving, summed
over envs), so every method gets the same simulated experience whatever its decision period.
gamma and GAE lambda are set per 0.1 s and scaled to the decision period (x ** (T_decision / 0.1 s)) so every method
has the same lookahead / credit window in seconds. Rationale for every setting: docs/TRAINING.md.

Robustness: a crashed env / eval worker ends the job (FAILED.txt, checkpoints/crash.zip, all children killed,
non-zero exit); the watchdog kills it (exit 3) if no PPO step or eval completes for ``training.watchdog_minutes``.
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

EVAL_LOG_METRICS = ("success", "collision", "emergency_brake", "off_road", "timeout", "return", "mean_speed",
                    "min_gap", "min_ttc", "rms_long_jerk", "rms_lat_accel", "rms_steer_rate", "tracking_error_rms",
                    "solve_time_mean", "solve_time_p95", "fallback_count", "gated_brake_count", "lane_changes",
                    "time_to_complete")


# ======================================================================================= settings
def _per_decision(cfg: dict, per_step_value: float) -> float:
    t_dec = int(cfg["method"].get("decision_period", 1)) * float(cfg["sim"]["control_dt"])
    return float(per_step_value) ** (t_dec / 0.1)


def method_gamma(cfg: dict) -> float:
    return _per_decision(cfg, cfg["training"]["gamma_ref"])      # M1 0.99, M2 0.951, M3 0.904


def method_gae_lambda(cfg: dict) -> float:
    return _per_decision(cfg, cfg["training"]["gae_lambda_ref"])  # M1 0.995, M2 0.975, M3 0.951


def ppo_kwargs(cfg: dict) -> dict:
    kw = dict(cfg["ppo"])
    kw["policy_kwargs"] = dict(kw.get("policy_kwargs") or {})
    kw["policy_kwargs"].setdefault("normalize_images", False)   # BEV is already float in [0, 1]
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
        self.episodes_csv = CsvAppender(run_dir / "episodes.csv")
        self.control_steps = self.decisions = self.episodes = self.rollouts = 0
        self.next_eval = self.eval_freq if self.eval_freq > 0 else None
        self.next_ckpt = self.ckpt_freq if self.ckpt_freq > 0 else None
        self.best = (-np.inf, -np.inf)
        self.evaluator: ParallelEvaluator | None = None
        self.recent: list[dict] = []

    def _counters(self) -> dict:
        return {"train_control_steps": self.control_steps, "train_decisions": self.decisions,
                "train_sim_time": self.control_steps * self.dt}

    def _on_training_start(self) -> None:
        self.t_start = time.time()
        self.next_progress = self.t_start + self.progress_s
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
            for k in ("success", "collision", "emergency_brake", "off_road", "return", "mean_speed"):
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
               f"{self.episodes} ep | {self.rollouts} rollouts")
        recent = self.recent[-50:]
        if recent:
            crash = np.mean([max(r["collision"], r["emergency_brake"], r["off_road"]) for r in recent])
            msg += (f" | last {len(recent)} ep: return {np.mean([r['return'] for r in recent]):7.1f}, "
                    f"success {np.mean([r['success'] for r in recent]):.2f}, crash {crash:.2f}, "
                    f"{np.mean([r['mean_speed'] for r in recent]) * 3.6:4.1f} km/h")
        print(msg, flush=True)

    def _evaluate(self, final: bool = False) -> None:
        tag = f"eval_{self.control_steps:010d}" + ("_final" if final else "")
        if self.verbose:
            print(f"[eval {tag}] running {len(self.eval_seeds)} episodes on {self.n_eval_workers} workers ...",
                  flush=True)
        # CPU copy: eval workers run on CPU, and CUDA tensors must not be pickled across processes
        state = {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()}
        self.heartbeat()
        summaries = self.evaluator.evaluate(state, self.eval_seeds, self.deterministic, tag=tag)
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
                  f"return={agg.get('return', 0):.1f}")
        self.logger.dump(self.num_timesteps)


# ======================================================================================= entry point
def train(cfg: dict, run_dir: str | Path | None = None, verbose: int = 1) -> Path:
    if cfg["method"]["name"] == "b0":
        raise ValueError("B0 is not trained; use scripts/evaluate.py")
    run_dir = make_run_dir(cfg, run_dir)
    tr = cfg["training"]
    seed = int(cfg["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if tr.get("torch_threads"):
        torch.set_num_threads(int(tr["torch_threads"]))
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
    (run_dir / "device.txt").write_text(f"{model.device} torch={torch.__version__} cuda_available="
                                        f"{torch.cuda.is_available()} threads={torch.get_num_threads()}\n")
    if verbose:
        print(f"PPO device: {model.device} | run dir: {run_dir}")
    outputs = [CSVOutputFormat(str(run_dir / "train_progress.csv"))]
    if verbose:
        outputs.append(HumanOutputFormat(sys.stdout))
    model.set_logger(Logger(str(run_dir), outputs))
    watchdog = Watchdog(tr.get("watchdog_minutes", 30), run_dir)
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
    watchdog.stop()
    vec.close()
    return run_dir
