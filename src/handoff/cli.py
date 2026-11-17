"""Shared argparse helpers for scripts/."""

from __future__ import annotations

import argparse

from handoff.config import load_config


def add_config_args(p: argparse.ArgumentParser, method_required: bool = True) -> None:
    p.add_argument("--method", required=method_required, choices=["m1", "m2", "m3", "b0"])
    p.add_argument("--reward", default="r0", help="configs/reward/<name>.yaml")
    p.add_argument("--scenario", default="freeway_pass", help="configs/scenario/<name>.yaml")
    p.add_argument("--seed", type=int, default=None, help="experiment.seed override")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (e.g. --set training.n_envs=8)")


def config_from_args(args) -> dict:
    overrides = list(args.overrides)
    if getattr(args, "seed", None) is not None:
        overrides.append(f"experiment.seed={args.seed}")
    return load_config(args.method, args.reward, args.scenario, overrides)
