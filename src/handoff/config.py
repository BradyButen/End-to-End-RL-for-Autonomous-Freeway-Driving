"""Plain-YAML configuration with an in-house deep merge (spec §14; no Hydra).

Merge order: base.yaml -> scenario/<s>.yaml -> method/<m>.yaml -> reward/<r>.yaml -> CLI overrides.
CLI overrides use dotted keys: ``method.decision_period=10`` or ``ppo.learning_rate=1e-4``;
values are parsed as YAML so lists/numbers/bools work (``sim.map.straight_block_lengths=[100,100]``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged into ``base`` recursively (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_override(text: str) -> dict:
    """``"a.b.c=1"`` -> ``{"a": {"b": {"c": 1}}}``."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 leaves "1e-4" as a string
        try:
            value = float(value)
        except ValueError:
            pass
    node: dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def load_config(
    method: str,
    reward: str = "r0",
    scenario: str = "freeway_pass",
    overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """Build the fully merged run config."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = load_yaml(cdir / "base.yaml")
    for sub, name in (("scenario", scenario), ("method", method), ("reward", reward)):
        path = cdir / sub / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No {sub} config {path}")
        cfg = deep_merge(cfg, load_yaml(path))
    for ov in overrides or []:
        cfg = deep_merge(cfg, parse_override(ov))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def kmh(v_km_h: float) -> float:
    """km/h -> m/s."""
    return float(v_km_h) / 3.6
