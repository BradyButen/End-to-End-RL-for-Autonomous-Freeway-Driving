#!/usr/bin/env python
"""Train PPO for one (method, reward variant, seed).

    python scripts/train.py --method m3 --reward r3_fast --seed 0 --set training.n_envs=8
"""

import argparse

from handoff.config import add_config_args, config_from_args
from handoff.rl.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--run-dir", default=None, help="explicit run directory (default: runs/<m>/<r>/seed_<n>/<ts>)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    run_dir = train(cfg, run_dir=args.run_dir, verbose=0 if args.quiet else 1)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
