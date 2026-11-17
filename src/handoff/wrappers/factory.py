"""Environment factory: config -> fully wrapped method env (method-agnostic entry point)."""

from __future__ import annotations

from pathlib import Path

from handoff.wrappers.core import CoreEnv
from handoff.wrappers.methods import METHOD_WRAPPERS, DecisionWrapper


def make_env(cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
             render: bool = False) -> DecisionWrapper:
    method = cfg["method"]["name"]
    if method not in METHOD_WRAPPERS:
        raise ValueError(f"Unknown method {method}")
    core = CoreEnv(cfg, rank=rank, run_dir=run_dir, is_eval=is_eval, render=render, b0=(method == "b0"))
    return METHOD_WRAPPERS[method](core, cfg)


def env_factory(cfg: dict, *, rank: int = 0, run_dir=None, is_eval: bool = False, monitor: bool = False):
    """Picklable thunk for SubprocVecEnv."""

    def _thunk():
        env = make_env(cfg, rank=rank, run_dir=run_dir, is_eval=is_eval)
        if monitor:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        return env

    return _thunk


def decision_period(cfg: dict) -> int:
    return int(cfg["method"].get("decision_period", 1))


def method_gamma(cfg: dict) -> float:
    """γ_method = γ_ref ^ (T_decision / 0.1 s) — same lookahead in seconds for every method (spec §8)."""
    t_dec = decision_period(cfg) * float(cfg["sim"]["control_dt"])
    return float(cfg["training"]["gamma_ref"]) ** (t_dec / 0.1)
