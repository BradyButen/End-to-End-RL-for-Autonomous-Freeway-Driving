#!/usr/bin/env python
"""Quick CPU timing profile per method: where does the time go in one RL decision, and how long does a PPO
update take? Single env, single process, random-init PPO policy (no learning).

    python tests/profile_timing.py                       # m1 m2 m3, 50 decisions each
    python tests/profile_timing.py --methods m3 --decisions 20 --scenario freeway

Per decision it splits wall time into:
  policy   PPO policy forward pass (CNN on the BEV + MLP)
  sim      MetaDrive env.step (physics, IDM traffic, top-down BEV rendering), summed over control steps
  ctrl     controller (M2: trajectory + PID; M3: safe set + spline + corridor + MPC), summed over control steps
  other    our wrapper overhead (state extraction, reward, logging)
and then times a PPO rollout+update of --n-steps decisions (SB3 defaults otherwise).
"""

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from handoff.config import load_config  # noqa: E402
from handoff.rl.env.factory import make_env  # noqa: E402
from handoff.rl.train import ppo_kwargs  # noqa: E402


class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def wrap(self, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            self.t += time.perf_counter() - t0
            self.n += 1
            return out

        return inner


def profile(method: str, scenario: str, n_dec: int, n_steps: int) -> dict:
    cfg = load_config(method, scenario=scenario,
                      overrides=[f"ppo.n_steps={n_steps}", f"ppo.batch_size={min(64, n_steps)}"])
    env = make_env(cfg)
    sim_t, ctrl_t = Timer(), Timer()
    env.core.md.step = sim_t.wrap(env.core.md.step)
    if method != "m1":
        env.control = ctrl_t.wrap(env.control)
        env.decide = ctrl_t.wrap(env.decide)

    t0 = time.perf_counter()
    model = PPO("MultiInputPolicy", DummyVecEnv([lambda: env]), device="cpu", verbose=0, **ppo_kwargs(cfg))
    t_build = time.perf_counter() - t0

    obs, _ = env.reset(seed=0)
    pol_t, step_t = 0.0, 0.0
    ctrl_steps = 0
    for _ in range(n_dec):
        t0 = time.perf_counter()
        action, _ = model.policy.predict(obs, deterministic=False)
        pol_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        obs, _, te, tr, info = env.step(action)
        step_t += time.perf_counter() - t0
        ctrl_steps += info["decision_control_steps"]
        if te or tr:
            obs, _ = env.reset()

    sim_total, ctrl_total = sim_t.t, ctrl_t.t  # snapshot before learn() (the timers keep counting there)
    # PPO rollout + gradient update of n_steps decisions
    t0 = time.perf_counter()
    model.learn(total_timesteps=n_steps)
    t_learn = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.train()  # one more update on the same buffer: gradient-step cost alone
    t_update = time.perf_counter() - t0
    env.close()

    other = step_t - sim_total - ctrl_total
    return {
        "method": method, "decisions": n_dec, "control_steps": ctrl_steps,
        "ms_per_decision": 1e3 * (pol_t + step_t) / n_dec,
        "policy_ms": 1e3 * pol_t / n_dec, "sim_ms": 1e3 * sim_total / n_dec, "ctrl_ms": 1e3 * ctrl_total / n_dec,
        "other_ms": 1e3 * other / n_dec,
        "ms_per_control_step": 1e3 * (pol_t + step_t) / ctrl_steps,
        "sim_s_per_wall_s": ctrl_steps * cfg["sim"]["control_dt"] / (pol_t + step_t),
        "model_build_s": t_build, "learn_s": t_learn, "update_s": t_update, "n_steps": n_steps,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["m1", "m2", "m3"])
    p.add_argument("--scenario", default="freeway")
    p.add_argument("--decisions", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=256, help="decisions in the timed PPO rollout+update")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = [profile(m, args.scenario, args.decisions, args.n_steps) for m in args.methods]
    print(f"\nscenario={args.scenario}  torch threads={args.threads}  (per decision, single env)")
    print(f"{'method':6} {'ctrl/dec':>8} {'total ms':>9} {'policy':>7} {'sim':>7} {'ctrl':>7} {'other':>7} "
          f"{'ms/ctrl':>8} {'sim-s/s':>8} | {'learn s':>8} {'update s':>8}")
    for r in rows:
        print(f"{r['method']:6} {r['control_steps'] / r['decisions']:8.1f} {r['ms_per_decision']:9.1f} "
              f"{r['policy_ms']:7.1f} {r['sim_ms']:7.1f} {r['ctrl_ms']:7.1f} {r['other_ms']:7.1f} "
              f"{r['ms_per_control_step']:8.2f} {r['sim_s_per_wall_s']:8.1f} | "
              f"{r['learn_s']:8.1f} {r['update_s']:8.1f}")
    print(f"\nlearn s = PPO rollout + update of {args.n_steps} decisions; update s = one gradient update "
          f"(n_epochs=10, batch 64) on that buffer.")


if __name__ == "__main__":
    main()
