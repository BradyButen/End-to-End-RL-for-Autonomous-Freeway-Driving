"""Run directory management (spec §12.3).

runs/<method>/<reward_variant>/seed_<n>/<timestamp>/
    config.yaml  git_commit.txt  train_progress.csv  episodes.csv  episodes/<id>.npz  checkpoints/
"""

from __future__ import annotations

import csv
import datetime as _dt
import subprocess
from pathlib import Path

from handoff.config import REPO_ROOT, save_config


def git_description() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - no git / no commits yet
        return "unknown"


def make_run_dir(cfg: dict, root: str | Path | None = None, name: str | None = None) -> Path:
    exp = cfg["experiment"]
    root = Path(root or exp.get("runs_root", "runs"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    stamp = name or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return init_run_dir(root / exp["method"] / exp["reward_variant"] / f"seed_{exp['seed']}" / stamp, cfg)


def init_run_dir(run: str | Path, cfg: dict) -> Path:
    """Create the run layout at an explicit path and snapshot config + code version."""
    run = Path(run)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "episodes").mkdir(exist_ok=True)
    save_config(cfg, run / "config.yaml")
    (run / "git_commit.txt").write_text(git_description() + "\n")
    return run


class CsvAppender:
    """Append dict rows to a CSV; header fixed by the first row (later unknown keys are dropped)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fields: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path) as f:
                self.fields = next(csv.reader(f))

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.fields is None:
            self.fields = list(rows[0].keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, self.fields, restval="", extrasaction="ignore")
            for r in rows:
                w.writerow(r)
