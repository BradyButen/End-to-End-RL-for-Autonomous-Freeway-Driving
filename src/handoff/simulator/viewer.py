"""Live viewer and replay (spec §13) inside MetaDrive's own window — no separate video pipeline.

* live:   run a policy (checkpoint or B0) with rendering on and overlays every control step.
* replay: deterministic re-simulation of a logged episode from its seed + the frozen checkpoint; overlays
          are redrawn from the logged decision/step arrays (MetaDrive's recorder does not store them), and the
          re-simulated ego pose is checked against the log.
"""

from __future__ import annotations

import numpy as np

from handoff.rl.env.factory import make_env
from handoff.rl.eval.harness import run_episodes, sb3_policy_fn
from handoff.rl.records.recorder import load_episode
from handoff.simulator.overlays import OverlayDrawer, live_frame, replay_frame


def load_policy_fn(checkpoint: str | None, deterministic: bool = True):
    if checkpoint is None:
        return None
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint, device="cpu")
    return sb3_policy_fn(model.policy, deterministic)


def _render(core, frame, topdown: bool):
    if topdown:
        core.md.render(mode="topdown", text=frame.get("text"), screen_size=(900, 900), film_size=(4000, 4000),
                       target_agent_heading_up=False, semantic_map=False)
    else:
        core.md.render(text=frame.get("text"))


def live(cfg: dict, checkpoint: str | None, seeds: list[int], topdown: bool = False, run_dir=None) -> list[dict]:
    env = make_env(cfg, render=not topdown, is_eval=True, run_dir=run_dir)
    core = env.core
    drawer = None

    def hook(core_, decision_ctx, diag):
        nonlocal drawer
        frame = live_frame(core_, decision_ctx, diag)
        if not topdown:
            if drawer is None:
                drawer = OverlayDrawer(core_.md.engine)
            drawer.draw(frame)
        _render(core_, frame, topdown)

    core.overlay_hook = hook
    try:
        return run_episodes(env, load_policy_fn(checkpoint), seeds, tag="view")
    finally:
        env.close()


def replay(cfg: dict, episode_npz: str, checkpoint: str | None, topdown: bool = False,
           tolerance: float = 0.5) -> dict:
    """Re-simulate a logged episode; returns {'max_deviation': m, 'steps': n, 'diverged': bool}."""
    ep = load_episode(episode_npz)
    data, meta = ep["data"], ep["meta"]
    seed = int(meta["seed"])
    env = make_env(cfg, render=not topdown, is_eval=True)
    core = env.core
    drawer = None
    state = {"i": 0, "max_dev": 0.0}

    def hook(core_, decision_ctx, diag):
        nonlocal drawer
        i = state["i"]
        if i < len(data["step_x"]):
            dev = float(np.hypot(core_.ego.x - data["step_x"][i], core_.ego.y - data["step_y"][i]))
            state["max_dev"] = max(state["max_dev"], dev)
            frame = replay_frame(data, i, core_.road)
            frame["text"]["deviation"] = f"{dev:.3f} m"
            if not topdown:
                if drawer is None:
                    drawer = OverlayDrawer(core_.md.engine)
                drawer.draw(frame)
            _render(core_, frame, topdown)
        state["i"] += 1

    core.overlay_hook = hook
    try:
        run_episodes(env, load_policy_fn(checkpoint), [seed], tag="replay")
    finally:
        env.close()
    return {"max_deviation": state["max_dev"], "steps": state["i"], "logged_steps": len(data["step_x"]),
            "diverged": state["max_dev"] > tolerance}
