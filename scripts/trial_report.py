#!/usr/bin/env python
"""Report for a trial folder of training runs (scripts/laptop_trial.sh, or any folder of run dirs).

    python scripts/trial_report.py runs/laptop_trial/<timestamp>

Reads <folder>/<method>/{episodes.csv, train_progress.csv} and writes <folder>/report/:
  training_curves.png   rolling mean over training episodes (stochastic policy) vs simulated driving time:
                        return, success, crash rate (collision + emergency_brake + off_road), mean speed,
                        episode length, lane changes
  eval_curves.png       deterministic eval rounds (step 0, every eval.freq_control_steps, final)
  ppo_diagnostics.png   approx_kl, clip fraction, explained variance, value loss, action std
  summary.md / .csv     per method: totals, first vs last 20 % of training episodes, first vs last eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CRASH_COLS = ("collision", "emergency_brake", "off_road")
METRICS = [("return", "return"), ("success", "success rate"),
           ("crash", "crash rate (coll + e-brake + off-road)"), ("mean_speed_kmh", "mean speed (km/h)"),
           ("sim_time", "episode length (s)"), ("lane_changes", "lane changes")]
PPO = [("train/approx_kl", "approx KL"), ("train/clip_fraction", "clip fraction"),
       ("train/explained_variance", "explained variance"), ("train/value_loss", "value loss"),
       ("train/std", "action std")]


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep = pd.read_csv(run / "episodes.csv", low_memory=False)
    for c in CRASH_COLS:
        if c not in ep.columns:
            ep[c] = 0.0
    ep["crash"] = ep[list(CRASH_COLS)].sum(axis=1).clip(upper=1)
    ep["mean_speed_kmh"] = ep["mean_speed"] * 3.6
    ep["hours"] = ep["train_sim_time"] / 3600.0
    is_eval = ep["eval_tag"].notna() if "eval_tag" in ep.columns else ep["is_eval"] == 1
    tr = ep[~is_eval].sort_values("train_control_steps")
    ev = ep[is_eval]
    prog_f = run / "train_progress.csv"
    prog = pd.read_csv(prog_f) if prog_f.exists() else pd.DataFrame()
    return tr, ev, prog


def rolling(tr: pd.DataFrame, col: str) -> pd.Series:
    w = max(10, len(tr) // 20)
    return tr[col].rolling(w, min_periods=max(3, w // 3)).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    args = p.parse_args()
    root = Path(args.folder)
    runs = {d.name: d for d in sorted(root.iterdir()) if (d / "episodes.csv").exists()}
    if not runs:
        sys.exit(f"no run dirs with episodes.csv under {root}")
    out = root / "report"
    out.mkdir(exist_ok=True)
    data = {m: load(d) for m, d in runs.items()}

    # training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for m, (tr, _, _) in data.items():
        for ax, (col, label) in zip(axes.flat, METRICS):
            if col in tr.columns and len(tr):
                ax.plot(tr["hours"], rolling(tr, col), label=f"{m} ({len(tr)} ep)")
                ax.set_title(label)
    for ax in axes.flat:
        ax.set_xlabel("simulated driving (h)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f"training episodes (rolling mean, stochastic policy) — {root.name}")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)

    # eval curves
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for m, (_, ev, _) in data.items():
        if ev.empty:
            continue
        g = ev.groupby("train_control_steps")
        for ax, (col, label) in zip(axes, METRICS[:4]):
            s = g[col].mean()
            ax.plot(g["hours"].first(), s.values, "o-", label=m)
            ax.set_title(f"eval: {label}")
    for ax in axes:
        ax.set_xlabel("simulated driving (h)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "eval_curves.png", dpi=130)
    plt.close(fig)

    # PPO diagnostics
    fig, axes = plt.subplots(1, len(PPO), figsize=(4 * len(PPO), 3.6))
    for m, (_, _, prog) in data.items():
        if prog.empty or "counters/control_steps" not in prog.columns:
            continue
        for ax, (col, label) in zip(axes, PPO):
            if col in prog.columns:
                d = prog[["counters/control_steps", col]].dropna()
                ax.plot(d["counters/control_steps"] / 1e3, d[col], ".-", label=m)
                ax.set_title(label)
    for ax in axes:
        ax.set_xlabel("control steps (k)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "ppo_diagnostics.png", dpi=130)
    plt.close(fig)

    # summary
    rows = []
    for m, (tr, ev, prog) in data.items():
        n = len(tr)
        k = max(1, n // 5)
        first, last = tr.head(k), tr.tail(k)
        row = {"method": m, "train_episodes": n,
               "control_steps": int(tr["train_control_steps"].max()) if n else 0,
               "driving_h": round(float(tr["hours"].max()), 2) if n else 0.0,
               "ppo_updates": int(prog["train/approx_kl"].notna().sum()) if "train/approx_kl" in prog else 0}
        for col in ("return", "success", "crash", "mean_speed_kmh"):
            row[f"{col}_first20%"] = round(float(first[col].mean()), 3) if n else np.nan
            row[f"{col}_last20%"] = round(float(last[col].mean()), 3) if n else np.nan
        if not ev.empty:
            tags = ev.groupby("train_control_steps")
            e0, e1 = tags.get_group(min(tags.groups)), tags.get_group(max(tags.groups))
            for col in ("return", "success", "crash", "mean_speed_kmh"):
                row[f"eval_{col}_start"] = round(float(e0[col].mean()), 3)
                row[f"eval_{col}_end"] = round(float(e1[col].mean()), 3)
            row["eval_rounds"] = len(tags)
        rows.append(row)
    summ = pd.DataFrame(rows).set_index("method")
    summ.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(f"# Trial {root.name}\n\n```\n{summ.T.to_string()}\n```\n")
    print(summ.T.to_string())
    print(f"\nreport: {out}/ (training_curves.png, eval_curves.png, ppo_diagnostics.png, summary.md)")


if __name__ == "__main__":
    main()
