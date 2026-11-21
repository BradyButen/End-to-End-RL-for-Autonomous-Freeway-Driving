#!/usr/bin/env python
"""Analysis plots.

    python scripts/plot.py curves runs/m*/r4_straight/seed_*/* --metric success --x control_steps --out curves.png
    python scripts/plot.py table runs/*/r4_straight/seed_*/*/final_eval/episodes.csv runs/eval/b0/episodes.csv
    python scripts/plot.py episode runs/.../episodes/<id>.npz      # <id>_overview.png + <id>_steps.csv
    python scripts/plot.py reward runs/.../episodes/<id>.npz       # how the return built up (<id>_reward.png)
    python scripts/plot.py reward <id>.npz --reward r4             # re-scored with configs/r4.yaml
"""

import argparse
from pathlib import Path

from handoff.viz import comparison_table, episode_report, learning_curves, reward_report, reward_weights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--metric", default="success")
    c.add_argument("--x", default="control_steps", choices=["control_steps", "decisions", "sim_time"])
    c.add_argument("--out", default="curves.png")
    t = sub.add_parser("table")
    t.add_argument("csvs", nargs="+")
    t.add_argument("--out", default=None, help="optional .csv / .md output")
    e = sub.add_parser("episode")
    e.add_argument("npz")
    e.add_argument("--out", default=None, help="output folder (default: next to the .npz)")
    rw = sub.add_parser("reward")
    rw.add_argument("npz")
    rw.add_argument("--reward", default=None, help="re-score with configs/<name>.yaml (default: as logged)")
    rw.add_argument("--out", default=None, help="output folder (default: next to the .npz)")
    args = p.parse_args()
    if args.cmd == "curves":
        print(learning_curves(args.run_dirs, args.metric, args.x, args.out))
    elif args.cmd == "table":
        tab = comparison_table(args.csvs)
        print(tab.to_string())
        if args.out:
            (tab.to_markdown if args.out.endswith(".md") else tab.to_csv)(args.out)
    elif args.cmd == "episode":
        print(episode_report(args.npz, args.out or Path(args.npz).parent))
    else:
        path, totals = reward_report(args.npz, args.out, reward_weights(args.reward) if args.reward else None,
                                     label=args.reward)
        for k, v in totals.items():
            print(f"{k:>12s}: {v:+9.2f}")
        print(path)


if __name__ == "__main__":
    main()
