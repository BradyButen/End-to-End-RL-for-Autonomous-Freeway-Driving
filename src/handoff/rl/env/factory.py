"""Environment factory: config -> fully wrapped method env (method-agnostic entry point)."""

from __future__ import annotations

from pathlib import Path

from handoff.rl.env.core import CoreEnv
from handoff.rl.env.decision import DecisionWrapper
from handoff.rl.methods.b0_main import B0Wrapper
from handoff.rl.methods.m1_main import M1Wrapper
from handoff.rl.methods.m2_main import PIDWrapper
from handoff.rl.methods.m3_main import MPCWrapper

METHOD_WRAPPERS: dict[str, type[DecisionWrapper]] = {
    "b0": B0Wrapper, "m1": M1Wrapper, "m2": PIDWrapper, "m3": MPCWrapper,
}


def make_env(cfg: dict, *, rank: int = 0, run_dir: str | Path | None = None, is_eval: bool = False,
             render: bool = False) -> DecisionWrapper:
    method = cfg["method"]["name"]
    if method not in METHOD_WRAPPERS:
        raise ValueError(f"Unknown method {method}")
    wrapper = METHOD_WRAPPERS[method]
    core = CoreEnv(cfg, rank=rank, run_dir=run_dir, is_eval=is_eval, render=render,
                   agent_policy=wrapper.agent_policy)
    return wrapper(core, cfg)


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


def method_gae_lambda(cfg: dict) -> float:
    """λ_method = λ_ref ^ (T_decision / 0.1 s) — same GAE credit-assignment window in seconds for every method
    (a fixed per-decision λ would give M1 a 10x shorter window than M3). docs/TRAINING.md §3."""
    t_dec = decision_period(cfg) * float(cfg["sim"]["control_dt"])
    return float(cfg["training"]["gae_lambda_ref"]) ** (t_dec / 0.1)
