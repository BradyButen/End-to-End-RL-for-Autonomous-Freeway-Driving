"""Evaluation: run a policy (an SB3 policy, or none for B0) on fixed seeds, deterministically.

``ParallelEvaluator`` runs the eval envs in their own subprocesses (one MetaDrive per process). Training uses it
every ``eval.freq_control_steps``, and scripts/warm_start/bc.py uses it to pick the best clone.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np


def run_episodes(env, policy_fn: Callable[[dict], np.ndarray] | None, seeds: list[int],
                 tag: str | None = None) -> list[dict]:
    """One episode per seed -> episode summaries in seed order. ``policy_fn`` None = zero actions (B0)."""
    out = []
    for seed in seeds:
        obs, _ = env.reset(options={"episode_seed": int(seed), "tag": tag})
        while True:
            action = policy_fn(obs) if policy_fn is not None else np.zeros(env.action_space.shape, np.float32)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                out.append(info["episode_summary"])
                break
    return out


def sb3_policy_fn(policy, deterministic: bool = True) -> Callable[[dict], np.ndarray]:
    return lambda obs: policy.predict(obs, deterministic=deterministic)[0]


def load_policy_fn(checkpoint: str | Path | None, deterministic: bool = True, cfg: dict | None = None):
    """SB3 checkpoint -> policy function (None -> None, for B0). With ``cfg``, the checkpoint must have been trained
    with this code's observation and action spaces for that method (checked before any simulator starts)."""
    if checkpoint is None:
        return None
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    if cfg is not None:
        check_spaces(model, cfg, checkpoint)
    return sb3_policy_fn(model.policy, deterministic)


def check_spaces(model, cfg: dict, checkpoint) -> None:
    """Raise a clear error when a checkpoint's observation or action space differs from the current code's for its
    method (it was trained on an older version and cannot drive in the current env)."""
    from handoff.env import method_class, observation_space

    want_obs = observation_space(cfg)
    want_act = method_class(cfg["method"]["name"]).ACTION_SPACE
    bad = []
    if model.observation_space != want_obs:
        bad.append(f"observation {model.observation_space} != current {want_obs}")
    if want_act is not None and model.action_space != want_act:
        bad.append(f"action {model.action_space} != current {want_act}")
    if bad:
        raise SystemExit(f"{checkpoint} was trained with an older version of the code and cannot run in the current "
                         f"env ({'; '.join(bad)}).")


def _const_lr(_progress):   # picklable stand-in for SB3's learning-rate schedule (unused at inference)
    return 0.0


def policy_spec(model) -> tuple[type, dict]:
    """(policy class, constructor kwargs) that can be shipped to a worker process."""
    params = dict(model.policy._get_constructor_parameters())
    params["lr_schedule"] = _const_lr
    return type(model.policy), params


def _worker(conn, cfg: dict, run_dir: str | None, worker_id: int):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import torch

    torch.set_num_threads(1)
    from handoff.env import make_env

    env = policy = None
    try:
        env = make_env(cfg, rank=1000 + worker_id, run_dir=run_dir, is_eval=True)
        conn.send(("ready", None))
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            if cmd == "init_policy":
                cls, kwargs = payload
                policy = cls(**kwargs)
                policy.set_training_mode(False)
                conn.send(("ok", None))
            elif cmd == "eval":
                state_dict, seeds, deterministic, tag = payload
                fn = None
                if state_dict is not None:
                    policy.load_state_dict(state_dict)
                    fn = sb3_policy_fn(policy, deterministic)
                for seed in seeds:        # one message per episode, so the trainer's watchdog sees progress
                    conn.send(("episode", (seed, run_episodes(env, fn, [seed], tag=tag)[0])))
                conn.send(("done", None))
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class ParallelEvaluator:
    def __init__(self, cfg: dict, run_dir: str | Path | None, n_workers: int = 1, policy_cls_kwargs=None):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, str(run_dir) if run_dir else None, i), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:
            self._expect(c, "ready")
        if policy_cls_kwargs is not None:
            for c in self.conns:
                c.send(("init_policy", policy_cls_kwargs))
            for c in self.conns:
                self._expect(c, "ok")

    @staticmethod
    def _expect(conn, kind):
        msg, payload = conn.recv()
        if msg == "error":
            raise RuntimeError(f"evaluation worker failed:\n{payload}")
        if msg != kind:
            raise RuntimeError(f"unexpected worker message {msg}")
        return payload

    def evaluate(self, state_dict, seeds: list[int], deterministic: bool = True, tag: str | None = None,
                 heartbeat=None) -> list[dict]:
        """``state_dict`` None = no policy (B0). ``heartbeat()`` is called after every finished episode. A worker
        that dies raises here (EOFError); one that errors raises RuntimeError with its traceback."""
        from multiprocessing.connection import wait

        for i, c in enumerate(self.conns):
            c.send(("eval", (state_dict, list(seeds[i::len(self.conns)]), deterministic, tag)))
        results, pending = {}, set(self.conns)
        while pending:
            for c in wait(list(pending)):
                msg, payload = c.recv()
                if msg == "error":
                    raise RuntimeError(f"evaluation worker failed:\n{payload}")
                if msg == "episode":
                    results[payload[0]] = payload[1]
                    if heartbeat is not None:
                        heartbeat()
                elif msg == "done":
                    pending.discard(c)
        return [results[s] for s in seeds]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


def aggregate(summaries: list[dict]) -> dict:
    """Mean of every numeric metric over episodes (NaNs skipped)."""
    if not summaries:
        return {}
    keys = [k for k, v in summaries[0].items()
            if isinstance(v, (int, float, np.floating, np.integer)) and k not in ("seed", "rank", "is_eval")]
    agg = {}
    for k in keys:
        vals = np.array([s.get(k, np.nan) for s in summaries], dtype=float)
        vals = vals[np.isfinite(vals)]
        agg[k] = float(np.mean(vals)) if len(vals) else float("nan")
    agg["n_episodes"] = len(summaries)
    return agg
