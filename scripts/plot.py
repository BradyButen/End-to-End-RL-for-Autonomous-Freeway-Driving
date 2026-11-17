#!/usr/bin/env python
"""Analysis plots.

    python scripts/plot.py curves runs/m1/r0/seed_*/* runs/m3/r0/seed_*/* --metric success --x control_steps
    python scripts/plot.py table runs/*/r0/seed_*/*/final_eval/episodes.csv runs/eval/b0/episodes.csv
    python scripts/plot.py episode runs/.../episodes/<id>.npz
"""

import argparse

from handoff.viz.plots import comparison_table, episode_timeseries, learning_curves


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
    t.add_argument("--out", default=None, help="optional .csv/.md output")
    e = sub.add_parser("episode")
    e.add_argument("npz")
    e.add_argument("--out", default="episode.png")
    args = p.parse_args()
    if args.cmd == "curves":
        print(learning_curves(args.run_dirs, args.metric, args.x, args.out))
    elif args.cmd == "table":
        tab = comparison_table(args.csvs)
        print(tab.to_string())
        if args.out:
            (tab.to_markdown if args.out.endswith(".md") else tab.to_csv)(args.out)
    else:
        print(episode_timeseries(args.npz, args.out))


if __name__ == "__main__":
    main()
