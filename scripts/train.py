#!/usr/bin/env python
"""Train PPO for one (method, seed).

    python scripts/train.py --method m3 --seed 0 --set training.n_envs=8
"""

import argparse

from handoff.config import add_config_args, config_from_args
from handoff.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--run-dir", default=None, help="explicit run directory (default: runs/<m>/<r>/seed_<n>/<ts>)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_dir = train(config_from_args(args), run_dir=args.run_dir, verbose=0 if args.quiet else 1)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
