#!/usr/bin/env python
"""Throughput benchmark (spec §10, §17.2): control steps/s and decisions/s for M1, M2, M3 (top-down
observation on) at a chosen number of parallel envs, plus the MPC solve-time distribution.

Uses SubprocVecEnv with random actions — the same process layout as training, without learning.

    python scripts/benchmark_throughput.py --n-envs 1 4 8 --decisions 300
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from handoff.config import load_config  # noqa: E402
from handoff.wrappers.factory import env_factory  # noqa: E402


def bench(method: str, n_envs: int, n_decisions: int, overrides) -> dict:
    cfg = load_config(method, overrides=overrides)
    fns = [env_factory(cfg, rank=i) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns, start_method="spawn")
    vec.seed(0)
    vec.reset()
    rng = np.random.default_rng(0)
    # warm-up
    for _ in range(5):
        vec.step(rng.uniform(-1, 1, (n_envs, *vec.action_space.shape)).astype(np.float32))
    control, decisions, solve_times, episodes = 0, 0, [], 0
    t0 = time.perf_counter()
    for _ in range(n_decisions):
        _, _, dones, infos = vec.step(rng.uniform(-1, 1, (n_envs, *vec.action_space.shape)).astype(np.float32))
        for info in infos:
            control += int(info.get("decision_control_steps", 1))
            decisions += 1
            s = info.get("episode_summary")
            if s is not None:
                episodes += 1
                if np.isfinite(s.get("solve_time_mean", np.nan)):
                    solve_times.append((s["solve_time_mean"], s["solve_time_p95"], s["solve_time_max"]))
    wall = time.perf_counter() - t0
    vec.close()
    st = np.array(solve_times) if solve_times else None
    return {
        "method": method, "n_envs": n_envs, "wall_s": wall, "control_steps": control, "decisions": decisions,
        "episodes": episodes,
        "control_steps_per_s": control / wall, "decisions_per_s": decisions / wall,
        "sim_seconds_per_wall_second": control * cfg["sim"]["control_dt"] / wall,
        "solve_time_mean_ms": float(np.mean(st[:, 0]) * 1e3) if st is not None else None,
        "solve_time_p95_ms": float(np.mean(st[:, 1]) * 1e3) if st is not None else None,
        "solve_time_max_ms": float(np.max(st[:, 2]) * 1e3) if st is not None else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["m1", "m2", "m3"])
    p.add_argument("--n-envs", nargs="+", type=int, default=[1, 4])
    p.add_argument("--decisions", type=int, default=300, help="vec-env steps per measurement")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--out", default="runs/benchmark/throughput.json")
    args = p.parse_args()
    results = []
    for m in args.methods:
        for n in args.n_envs:
            r = bench(m, n, args.decisions, args.overrides)
            results.append(r)
            print(f"{m} n_envs={n}: {r['control_steps_per_s']:.0f} ctrl/s  {r['decisions_per_s']:.1f} dec/s  "
                  f"{r['sim_seconds_per_wall_second']:.1f} sim-s/s  "
                  + (f"solve mean {r['solve_time_mean_ms']:.1f} ms p95 {r['solve_time_p95_ms']:.1f} ms"
                     if r["solve_time_mean_ms"] is not None else ""), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cpu_count": os.cpu_count(), "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
