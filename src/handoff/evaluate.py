"""Evaluation: run a policy (an SB3 policy, or none for B0) on fixed seeds, deterministically.

``ParallelEvaluator`` runs the eval envs in their own subprocesses (one MetaDrive per process), so it can be used
from inside training (every eval.freq_control_steps) and by scripts/evaluate.py.
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


def load_policy_fn(checkpoint: str | Path | None, deterministic: bool = True):
    """SB3 checkpoint -> policy function (None -> None, for B0)."""
    if checkpoint is None:
        return None
    from stable_baselines3 import PPO

    return sb3_policy_fn(PPO.load(str(checkpoint), device="cpu").policy, deterministic)


def _const_lr(_progress):   # picklable stand-in for SB3's lr schedule (unused at inference)
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
                conn.send(("result", run_episodes(env, fn, seeds, tag=tag)))
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

    def evaluate(self, state_dict, seeds: list[int], deterministic: bool = True, tag: str | None = None) -> list[dict]:
        """``state_dict`` None = no policy (B0)."""
        chunks = [list(seeds[i::len(self.conns)]) for i in range(len(self.conns))]
        for c, chunk in zip(self.conns, chunks):
            c.send(("eval", (state_dict, chunk, deterministic, tag)))
        results = {}
        for c, chunk in zip(self.conns, chunks):
            results.update(zip(chunk, self._expect(c, "result")))
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
