"""SB3 callbacks: episode logging, control-step-based eval/checkpoint cadence, budget stop (spec §10, §11, §12).

All cadences are counted in **control steps** (summed over envs) so methods with longer decision periods
are evaluated/checkpointed at the same simulated-experience points (DECISIONS D-07).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from handoff.rl.eval.harness import ParallelEvaluator, aggregate, policy_spec
from handoff.rl.records.run_dir import CsvAppender

LOG_METRICS = ("success", "collision", "emergency_brake", "off_road", "timeout", "return", "mean_speed", "min_gap",
               "min_ttc", "rms_long_jerk", "rms_lat_accel", "rms_steer_rate", "tracking_error_rms",
               "solve_time_mean", "solve_time_p95", "fallback_count", "gated_brake_count", "projection_distance_mean",
               "lane_changes", "time_to_complete")


class HandoffCallback(BaseCallback):
    def __init__(self, cfg: dict, run_dir: str | Path, eval_seeds: list[int], verbose: int = 0, heartbeat=None):
        super().__init__(verbose)
        self.heartbeat = heartbeat or (lambda: None)   # training watchdog (train.Watchdog.beat)
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.eval_seeds = list(eval_seeds)
        tr, ev = cfg["training"], cfg["eval"]
        self.budget = int(float(tr["total_control_steps"]))
        wall = tr.get("max_wall_minutes")
        self.wall_limit_s = float(wall) * 60.0 if wall else None   # optional wall-clock stop (laptop trials)
        self._t_start = None
        self.progress_s = float(tr.get("progress_seconds") or 0)   # one status line every N wall seconds (0 = off)
        self._next_progress = None
        self._updates = 0
        self.eval_freq = int(ev["freq_control_steps"])
        self.ckpt_freq = int(tr["checkpoint_freq_control_steps"])
        self.n_eval_workers = int(ev.get("n_workers", 1))
        self.deterministic = bool(ev.get("deterministic", True))
        self.episodes_csv = CsvAppender(self.run_dir / "episodes.csv")
        self.control_steps = 0
        self.decisions = 0
        self.episodes = 0
        self._next_eval = self.eval_freq if self.eval_freq > 0 else None
        self._next_ckpt = self.ckpt_freq if self.ckpt_freq > 0 else None
        self.best_key = (-np.inf, -np.inf)
        self.evaluator: ParallelEvaluator | None = None
        self._recent: list[dict] = []

    # --------------------------------------------------------------- lifecycle
    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self._next_progress = self._t_start + self.progress_s
        if self._next_eval is not None:
            self.evaluator = ParallelEvaluator(self.cfg, self.run_dir, self.n_eval_workers,
                                               policy_cls_kwargs=policy_spec(self.model))
            self._evaluate()  # step-0 evaluation (untrained policy) anchors the learning curve

    def _on_training_end(self) -> None:
        if self.evaluator is not None:
            self._evaluate(final=True)
            self.evaluator.close()
        self.model.save(self.run_dir / "checkpoints" / "final.zip")
        self._save_vecnorm("final")

    def _save_vecnorm(self, name: str) -> None:
        """Reward-normalisation statistics (only needed to resume training, not for eval)."""
        env = self.model.get_env()
        if hasattr(env, "ret_rms"):
            env.save(str(self.run_dir / "checkpoints" / f"vecnormalize_{name}.pkl"))

    # --------------------------------------------------------------- per step
    def _on_step(self) -> bool:
        self.heartbeat()
        infos = self.locals["infos"]
        rows = []
        for info in infos:
            self.control_steps += int(info.get("decision_control_steps", 1))
            self.decisions += 1
            summ = info.get("episode_summary")
            if summ is not None:
                self.episodes += 1
                row = {**summ, "train_control_steps": self.control_steps, "train_decisions": self.decisions,
                       "train_sim_time": self.control_steps * self.cfg["sim"]["control_dt"]}
                rows.append(row)
                self._recent.append(summ)
                del self._recent[:-100]
        self.episodes_csv.write(rows)
        if rows:
            recent = self._recent[-100:]
            for k in ("success", "collision", "emergency_brake", "off_road", "return", "mean_speed"):
                self.logger.record(f"train/{k}", float(np.mean([r[k] for r in recent])))
        self.logger.record("counters/control_steps", self.control_steps)
        self.logger.record("counters/decisions", self.decisions)
        self.logger.record("counters/sim_time", self.control_steps * self.cfg["sim"]["control_dt"])
        self.logger.record("counters/episodes", self.episodes)

        if self._next_ckpt is not None and self.control_steps >= self._next_ckpt:
            self.model.save(self.run_dir / "checkpoints" / f"ckpt_{self.control_steps:010d}.zip")
            self._save_vecnorm(f"{self.control_steps:010d}")
            self._next_ckpt += self.ckpt_freq
        if self._next_eval is not None and self.control_steps >= self._next_eval:
            self._evaluate()
            self._next_eval += self.eval_freq
        if self.progress_s > 0 and time.time() >= self._next_progress:
            self._print_progress()
            self._next_progress = time.time() + self.progress_s
        if self.wall_limit_s is not None and time.time() - self._t_start > self.wall_limit_s:
            if self.verbose:
                print(f"[stop] wall-clock limit {self.wall_limit_s / 60:.1f} min reached at {self.control_steps} "
                      "control steps")
            return False
        return self.control_steps < self.budget

    def _on_rollout_end(self) -> None:
        self._updates += 1      # a PPO update follows every rollout

    def _print_progress(self) -> None:
        el = time.time() - self._t_start
        lim = f"/{self.wall_limit_s / 60:.1f}" if self.wall_limit_s else ""
        rate = self.control_steps / max(el, 1e-9)
        msg = (f"[{self.cfg['method']['name']} {el / 60:5.1f}{lim} min] {self.control_steps / 1e3:7.1f}k steps "
               f"({rate:4.0f}/s, {self.control_steps * self.cfg['sim']['control_dt'] / 3600:.2f} h driven) | "
               f"{self.episodes} ep | {self._updates} rollouts")
        recent = self._recent[-50:]
        if recent:
            crash = np.mean([max(r.get("collision", 0), r.get("emergency_brake", 0), r.get("off_road", 0))
                             for r in recent])
            msg += (f" | last {len(recent)} ep: return {np.mean([r['return'] for r in recent]):7.1f}, "
                    f"success {np.mean([r['success'] for r in recent]):.2f}, crash {crash:.2f}, "
                    f"{np.mean([r['mean_speed'] for r in recent]) * 3.6:4.1f} km/h")
        print(msg, flush=True)

    def _evaluate(self, final: bool = False) -> dict:
        tag = f"eval_{self.control_steps:010d}" + ("_final" if final else "")
        if self.verbose:
            n = len(self.eval_seeds)
            print(f"[eval {tag}] running {n} episodes on {self.n_eval_workers} workers ...", flush=True)
        # CPU copy: eval workers run on CPU, and CUDA tensors must not be pickled across processes
        state = {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()}
        self.heartbeat()
        summaries = self.evaluator.evaluate(state, self.eval_seeds, self.deterministic, tag=tag)
        self.heartbeat()
        rows = [{**s, "train_control_steps": self.control_steps, "train_decisions": self.decisions,
                 "train_sim_time": self.control_steps * self.cfg["sim"]["control_dt"], "eval_tag": tag}
                for s in summaries]
        self.episodes_csv.write(rows)
        agg = aggregate(summaries)
        for k in LOG_METRICS:
            if k in agg:
                self.logger.record(f"eval/{k}", agg[k])
        key = (agg.get("success", 0.0), agg.get("return", -np.inf))
        if key > self.best_key:
            self.best_key = key
            self.model.save(self.run_dir / "checkpoints" / "best.zip")
            (self.run_dir / "checkpoints" / "best.txt").write_text(
                f"{tag} success={key[0]:.3f} return={key[1]:.2f}\n")
        if self.verbose:
            print(f"[eval {tag}] success={agg.get('success', 0):.2f} collision={agg.get('collision', 0):.2f} "
                  f"return={agg.get('return', 0):.1f}")
        self.logger.dump(self.num_timesteps)
        return agg
