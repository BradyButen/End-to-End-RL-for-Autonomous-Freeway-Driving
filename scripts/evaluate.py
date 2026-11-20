#!/usr/bin/env python
"""Final evaluation of a trained checkpoint, or of B0, on the fixed evaluation seeds (spec §11.1).

    python scripts/evaluate.py --method m3 --checkpoint runs/m3/r3_fast/seed_0/<ts>/checkpoints/best.zip
    python scripts/evaluate.py --method b0 --out runs/eval/b0

The config is taken from the checkpoint's run directory (config.yaml) when available, so the evaluated
env matches training; --set overrides still apply.
"""

import argparse
import json
from pathlib import Path

import yaml

from handoff.config import add_config_args, config_from_args, deep_merge, parse_override
from handoff.rl.eval.harness import ParallelEvaluator, aggregate, policy_spec
from handoff.rl.records.run_dir import CsvAppender


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--checkpoint", default=None, help="SB3 .zip (omit for b0)")
    p.add_argument("--out", default=None, help="output directory (default: <run>/final_eval or runs/eval/<m>)")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the distribution mean")
    p.add_argument("--n-episodes", type=int, default=None)
    args = p.parse_args()

    cfg = config_from_args(args)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is not None:
        run_cfg = ckpt.parent.parent / "config.yaml"
        if run_cfg.exists():
            cfg = yaml.safe_load(run_cfg.read_text())
            for ov in args.overrides:
                cfg = deep_merge(cfg, parse_override(ov))
    if cfg["method"]["name"] != "b0" and ckpt is None:
        p.error("--checkpoint is required for m1/m2/m3")

    out = Path(args.out) if args.out else (ckpt.parent.parent / "final_eval" if ckpt else
                                           Path("runs") / "eval" / cfg["method"]["name"])
    out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["scenario"]["seeds"]["eval"]
    if args.n_episodes:
        seeds = seeds[: args.n_episodes]

    spec, state = None, None
    if ckpt is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(ckpt), device="cpu")
        spec, state = policy_spec(model), model.policy.state_dict()
    ev = ParallelEvaluator(cfg, out, args.n_workers, policy_cls_kwargs=spec)
    try:
        summaries = ev.evaluate(state, seeds, deterministic=not args.stochastic, tag="final")
    finally:
        ev.close()
    for s in summaries:
        s["checkpoint"] = str(ckpt) if ckpt else "b0"
    CsvAppender(out / "episodes.csv").write(summaries)
    agg = aggregate(summaries)
    (out / "summary.json").write_text(json.dumps(agg, indent=2))
    keys = ["success", "collision", "off_road", "timeout", "mean_speed", "time_to_complete", "min_gap", "min_ttc",
            "rms_long_jerk", "solve_time_mean", "solve_time_p95", "fallback_count", "return"]
    for k in keys:
        if k in agg:
            print(f"{k:>20s}: {agg[k]:.4g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
