#!/usr/bin/env python
"""TEMP: replay the best and worst final-eval episode of each method from a trial folder in the 3D viewer.

    python scripts/tmp_best_worst.py                                   # latest runs/laptop_trial/*
    python scripts/tmp_best_worst.py runs/laptop_trial/20261119-182257 --methods m3
    python scripts/tmp_best_worst.py --dry-run                         # just print the picks + commands
    python scripts/tmp_best_worst.py --tag start                       # step-0 eval instead (untrained policy)
    python scripts/tmp_best_worst.py --tag best                        # eval round that produced best.zip (works for crashed runs)
    python scripts/tmp_best_worst.py --show best                       # only the best (or --show worst)

Best / worst = highest / lowest return among the chosen eval round's 40 deterministic episodes. Each is replayed by
re-simulation with the policy that produced it (final eval -> checkpoints/final.zip; step-0 eval -> the run's
untrained policy is not saved, so --tag start uses best.zip only if best.txt says it came from step 0).
Windows open one after another: close the window (or let the episode end) to go to the next.
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd


def pick(run: Path, tag: str):
    ep = pd.read_csv(run / "episodes.csv", low_memory=False)
    ev = ep[ep["eval_tag"].notna()]
    if ev.empty:
        return None, None, "no eval episodes"
    tags = sorted(ev["eval_tag"].unique())
    best_txt = run / "checkpoints" / "best.txt"
    if tag == "best":
        if not best_txt.exists():
            return None, None, "no best.txt"
        want = [t for t in tags if t == best_txt.read_text().split()[0]]
    else:
        want = [t for t in tags if t.endswith("_final")] if tag == "final" else [tags[0]]
    if not want:
        return None, None, f"no {tag} eval (crashed run? try --tag best)"
    g = ev[ev["eval_tag"] == want[0]].sort_values("return")
    if tag == "final":
        ckpt = run / "checkpoints" / "final.zip"
    elif tag == "best":
        ckpt = run / "checkpoints" / "best.zip"
    else:
        if not best_txt.exists() or not best_txt.read_text().startswith(want[0]):
            return None, None, "step-0 policy not saved (best.zip is from a later eval)"
        ckpt = run / "checkpoints" / "best.zip"
    return g, ckpt, want[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=None)
    p.add_argument("--methods", nargs="+", default=["m1", "m2", "m3"])
    p.add_argument("--tag", choices=["final", "best", "start"], default="final")
    p.add_argument("--show", choices=["both", "best", "worst"], default="both", help="which episodes to replay")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--topdown", action="store_true", help="2D top-down window instead of 3D")
    args = p.parse_args()
    folder = Path(args.folder) if args.folder else Path(sorted(glob.glob("runs/laptop_trial/*/"))[-1])
    print(f"trial: {folder}")

    cmds = []
    for m in args.methods:
        run = folder / m
        if not (run / "episodes.csv").exists():
            print(f"{m}: no run")
            continue
        g, ckpt, info = pick(run, args.tag)
        if g is None:
            print(f"{m}: {info}")
            continue
        print(f"\n{m} ({info}, {len(g)} episodes, checkpoint {ckpt.name}):")
        picks = {"worst": [("WORST", g.iloc[0])], "best": [("BEST", g.iloc[-1])],
                 "both": [("WORST", g.iloc[0]), ("BEST", g.iloc[-1])]}[args.show]
        for label, row in picks:
            print(f"  {label:5s} seed {int(row['seed'])}: {row['termination']:15s} return {row['return']:7.1f}  "
                  f"{row['mean_speed'] * 3.6:5.1f} km/h  {row['sim_time']:5.1f} s  lane changes "
                  f"{int(row['lane_changes'])}")
            cmd = [sys.executable, "scripts/view.py", "replay", "--episode", str(run / "episodes" / Path(str(row["log_file"])).name),
                   "--checkpoint", str(ckpt)] + (["--topdown"] if args.topdown else [])
            cmds.append((f"{m} {label} seed {int(row['seed'])}", cmd))

    print()
    for name, cmd in cmds:
        print(f"# {name}\n{' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
