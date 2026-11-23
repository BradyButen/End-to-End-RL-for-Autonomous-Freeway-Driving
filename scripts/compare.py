#!/usr/bin/env python
"""Method comparison table for the report: one row per method, mean ± std across seeds.

    python scripts/compare.py runs/eval/b0 runs/m*/r3_fast/seed_*/*                # markdown to stdout
    python scripts/compare.py runs/eval/b0 runs/m*/r3_fast/seed_*/* --out results/  # + table.md, table.csv, per_run.csv
    python scripts/compare.py 11_19_data/m*/r0/seed_0/* --tag final                 # a specific eval round

Each PATH is either
  * an evaluation folder (scripts/evaluate.py --out, or <run>/final_eval/): every row of its episodes.csv, or
  * a training run folder: <run>/final_eval/ if it exists (skip with --no-final-eval), otherwise one eval round
    from the run's episodes.csv: --tag (default: the last round logged, so a still-running job gives its latest eval).

Each folder is scored on its own first (per-episode means), then folders of the same method are averaged; ± is the
std across folders (seeds) and is left out when a method has a single folder. Crash = collision + emergency_brake +
off_road (the M3 e-brake ends the episode as a predicted crash, see docs/DESIGN.md §4). time_to_complete is over
successful episodes only; min gap / min TTC are per-episode minima averaged over episodes (inf = no car ahead, ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# (column, header, scale, format) — rates are shown in %, solve times in ms
METRICS = [
    ("success", "success %", 100, ".0f"),
    ("crash", "crash %", 100, ".0f"),
    ("collision", "collision %", 100, ".0f"),
    ("emergency_brake", "e-brake %", 100, ".0f"),
    ("off_road", "off-road %", 100, ".0f"),
    ("timeout", "timeout %", 100, ".0f"),
    ("mean_speed", "speed m/s", 1, ".1f"),
    ("time_to_complete", "finish s", 1, ".1f"),
    ("min_gap", "min gap m", 1, ".2f"),
    ("min_ttc", "min TTC s", 1, ".2f"),
    ("rms_long_jerk", "jerk m/s³", 1, ".2f"),
    ("rms_lat_accel", "lat acc m/s²", 1, ".2f"),
    ("lane_changes", "lane chg", 1, ".2f"),
    ("solve_time_p95", "solve p95 ms", 1000, ".1f"),
    ("fallback_count", "fallbacks", 1, ".2f"),
    ("return", "return", 1, ".1f"),
]
RATE_COLS = ("success", "collision", "emergency_brake", "off_road", "timeout")
METHOD_ORDER = ["b0", "m1", "m2", "m3"]


def episodes_for(path: Path, tag: str | None, use_final_eval: bool) -> tuple[pd.DataFrame, str]:
    """Rows to score for one PATH, and a short description of where they came from."""
    if use_final_eval and (path / "final_eval" / "episodes.csv").exists():
        path = path / "final_eval"
    csv = path / "episodes.csv"
    if not csv.exists():
        raise FileNotFoundError(f"no episodes.csv in {path}")
    df = pd.read_csv(csv, low_memory=False)
    if "eval_tag" not in df.columns or df["eval_tag"].isna().all():   # evaluate.py output: all rows are the eval
        return df, "eval"
    ev = df[df["eval_tag"].notna()]
    tags = list(dict.fromkeys(ev["eval_tag"]))                       # order of appearance = training order
    if tag is None:
        pick = tags[-1]
    elif tag == "start":                                             # step-0 eval (untrained policy)
        pick = tags[0]
    else:                                                            # exact tag, or a suffix like "final"
        hits = [t for t in tags if t == tag or t.endswith("_" + tag)]
        if not hits:
            raise ValueError(f"{path}: eval tag {tag!r} not found (have {tags[:3]}...{tags[-3:]})")
        pick = hits[-1]
    return ev[ev["eval_tag"] == pick], f"tag {pick}"


def score(df: pd.DataFrame) -> dict:
    """Per-episode means for one folder."""
    df = df.copy()
    for c in RATE_COLS:
        df[c] = df[c].astype(float) if c in df.columns else 0.0
    df["crash"] = df["collision"] + df["emergency_brake"] + df["off_road"]
    out = {"n_episodes": len(df)}
    for col, *_ in METRICS:
        if col not in df.columns:
            out[col] = np.nan
            continue
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col] = float(x.mean()) if x.notna().any() else np.nan
    return out


def fmt(mean: float, std: float, scale: float, f: str) -> str:
    if np.isnan(mean):
        return "–"
    s = format(mean * scale, f)
    return s if np.isnan(std) else f"{s} ± {format(std * scale, f)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--tag", default=None, help="eval round from training runs: 'final', 'start', or an exact tag "
                                               "(default: the last one logged)")
    p.add_argument("--no-final-eval", action="store_true", help="ignore <run>/final_eval/, use the training evals")
    p.add_argument("--metrics", default=None, help="comma-separated subset of columns (default: all)")
    p.add_argument("--out", type=Path, default=None, help="folder for table.md, table.csv, per_run.csv")
    args = p.parse_args()

    rows = []
    for path in args.paths:
        if not path.is_dir():
            continue
        try:
            df, src = episodes_for(path, args.tag, not args.no_final_eval)
        except (FileNotFoundError, ValueError) as e:
            print(f"skip: {e}", file=sys.stderr)
            continue
        method = str(df["method"].iloc[0])
        rows.append({"method": method, "path": str(path), "source": src, **score(df)})
    if not rows:
        sys.exit("no usable folders")
    per_run = pd.DataFrame(rows)

    metrics = METRICS
    if args.metrics:
        keep = set(args.metrics.split(","))
        metrics = [m for m in METRICS if m[0] in keep]
    # drop columns no method has (e.g. solve time when there is no M3)
    metrics = [m for m in metrics if per_run[m[0]].notna().any()]

    methods = sorted(per_run["method"].unique(), key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))
    table, csv_rows = [], []
    for m in methods:
        g = per_run[per_run["method"] == m]
        line = {"method": m.upper(), "seeds": str(len(g)), "episodes": str(int(g["n_episodes"].sum()))}
        csv_row = {"method": m, "seeds": len(g), "episodes": int(g["n_episodes"].sum())}
        for col, head, scale, f in metrics:
            mean = g[col].mean()
            std = g[col].std(ddof=1) if len(g) > 1 else np.nan
            line[head] = fmt(mean, std, scale, f)
            csv_row[col] = mean
            csv_row[col + "_std"] = std
        table.append(line)
        csv_rows.append(csv_row)

    heads = list(table[0].keys())
    md = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    md += ["| " + " | ".join(r[h] for h in heads) + " |" for r in table]
    md_text = "\n".join(md)
    print(md_text)
    print("\nsources: " + "; ".join(f"{r['method']} {r['path']} ({r['source']}, n={r['n_episodes']})"
                                    for r in rows), file=sys.stderr)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "table.md").write_text(md_text + "\n")
        pd.DataFrame(csv_rows).to_csv(args.out / "table.csv", index=False)
        per_run.to_csv(args.out / "per_run.csv", index=False)
        print(f"wrote {args.out}/table.md, table.csv, per_run.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
