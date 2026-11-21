"""Configuration: plain YAML files in configs/, deep-merged in a fixed order.

    configs/base.yaml -> configs/<scenario>.yaml -> configs/<method>.yaml -> configs/<reward>.yaml -> --set overrides

Overrides use dotted keys parsed as YAML: ``--set method.mpc.weights.lateral=4.0``, ``--set sim.map.blocks=SSSS``.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
METHODS = ("m1", "m2", "m3", "b0")


def kmh(v_km_h: float) -> float:
    """km/h -> m/s."""
    return float(v_km_h) / 3.6


def deep_merge(base: dict, override: dict) -> dict:
    """New dict: ``override`` merged into ``base`` recursively (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_override(text: str) -> dict:
    """``"a.b.c=1"`` -> ``{"a": {"b": {"c": 1}}}``."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    key, raw = text.split("=", 1)
    value: Any = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 leaves "1e-4" as a string
        try:
            value = float(value)
        except ValueError:
            pass
    node: dict = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for ov in overrides or []:
        cfg = deep_merge(cfg, parse_override(ov))
    return cfg


def load_config(method: str, reward: str = "r4_straight", scenario: str = "freeway",
                overrides: list[str] | None = None) -> dict:
    """The fully merged config for one run."""
    cfg = load_yaml(CONFIG_DIR / "base.yaml")
    for name in (scenario, method, reward):
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No config {path}")
        cfg = deep_merge(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides)


def load_run_config(run_dir: str | Path, overrides: list[str] | None = None) -> dict:
    """Config a training run used (``<run_dir>/config.yaml``), on top of today's defaults for that method so keys
    added since the run have a value. The reward is the run's own, whole (no new terms mixed into an old reward), so
    runs trained under a reward whose file has since been removed still evaluate as they were trained."""
    saved = load_yaml(Path(run_dir) / "config.yaml")
    exp = saved["experiment"]
    cfg = deep_merge(load_config(exp["method"], scenario=exp.get("scenario", "freeway")), saved)
    cfg["reward"] = saved["reward"]
    return apply_overrides(cfg, overrides)


def checkpoint_path(path: str | Path) -> Path:
    """A checkpoint .zip, or a run folder (-> its checkpoints/final.zip, else best.zip)."""
    p = Path(path)
    if p.is_dir():
        final = p / "checkpoints" / "final.zip"
        return final if final.exists() else p / "checkpoints" / "best.zip"
    return p


def run_dir_of(checkpoint: str | Path) -> Path:
    """Run folder of ``<run>/checkpoints/<name>.zip``."""
    return Path(checkpoint).resolve().parent.parent


# --------------------------------------------------------------------------------------- script arguments
def add_config_args(p: argparse.ArgumentParser, method_required: bool = True) -> None:
    p.add_argument("--method", required=method_required, choices=METHODS)
    p.add_argument("--reward", default="r4_straight", help="configs/<name>.yaml")
    p.add_argument("--scenario", default="freeway", help="configs/<name>.yaml")
    p.add_argument("--seed", type=int, default=None, help="experiment.seed override")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (e.g. --set training.n_envs=8)")


def config_from_args(args) -> dict:
    overrides = list(args.overrides)
    if getattr(args, "seed", None) is not None:
        overrides.append(f"experiment.seed={args.seed}")
    return load_config(args.method, args.reward, args.scenario, overrides)
