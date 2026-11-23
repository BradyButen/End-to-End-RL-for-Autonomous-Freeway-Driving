#!/usr/bin/env python
"""Final evaluation of a trained checkpoint, or of B0, on the fixed evaluation seeds.

    python scripts/evaluate.py --checkpoint runs/m3/r5/seed_0/<ts>/checkpoints/best.zip --n-workers 4
    python scripts/evaluate.py --checkpoint runs/m3/r5/seed_0/<ts>  # run folder: final.zip, else best.zip
    python scripts/evaluate.py --method b0 --out runs/eval/b0

With a checkpoint, the config is the run's own config.yaml (--set overrides still apply), so the evaluated env
matches training. Writes <out>/episodes.csv (one row per episode) and <out>/summary.json (means).
"""

import argparse
import json
from pathlib import Path

from handoff.config import METHODS, checkpoint_path, load_config, load_run_config, run_dir_of
from handoff.evaluate import ParallelEvaluator, aggregate, policy_spec
from handoff.logs import CsvAppender


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="SB3 .zip or run folder (omit for b0)")
    p.add_argument("--method", choices=METHODS, default=None, help="only needed without --checkpoint (b0)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="output directory (default: <run>/final_eval or runs/eval/<method>)")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--n-episodes", type=int, default=None, help="first N eval seeds (default: all)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the distribution mean")
    args = p.parse_args()

    ckpt = checkpoint_path(args.checkpoint) if args.checkpoint else None
    if ckpt is not None:
        cfg = load_run_config(run_dir_of(ckpt), args.overrides)
    elif args.method == "b0":
        cfg = load_config("b0", overrides=args.overrides)
    else:
        p.error("--checkpoint is required for m1/m2/m3 (or --method b0)")
    out = Path(args.out) if args.out else (run_dir_of(ckpt) / "final_eval" if ckpt else Path("runs/eval/b0"))
    out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["scenario"]["seeds"]["eval"][: args.n_episodes]

    spec = state = None
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
    for k in ("success", "collision", "emergency_brake", "infeasible_count", "off_road", "timeout", "mean_speed",
              "time_to_complete",
              "min_gap", "min_ttc", "rms_long_jerk", "solve_time_mean", "solve_time_p95", "fallback_count", "return"):
        if k in agg:
            print(f"{k:>20s}: {agg[k]:.4g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
