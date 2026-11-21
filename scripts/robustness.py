#!/usr/bin/env python
"""Robustness sweep: evaluate frozen policies under conditions they were not trained on (configs/robustness.yaml).

    python scripts/robustness.py --b0 --runs runs/m*/r3_fast/seed_*/*            # full sweep, then the report
    python scripts/robustness.py --b0 --runs ... --sweeps density --n-episodes 8 # quick partial check
    python scripts/robustness.py --b0 --runs ... --dry-run                       # list the evaluations only
    python scripts/robustness.py --report runs/robustness/<timestamp>            # rebuild tables/plots only

For every policy (B0 and/or each training run's checkpoint) x every condition, scripts/evaluate.py runs the eval
seeds with that condition's --set overrides on top of the run's own config.yaml. No training happens. A condition
equal to the nominal one is evaluated once and shared by every sweep. Re-running with the same --out skips
evaluations whose summary.json already exists, so an interrupted sweep resumes where it stopped.

Output (<out> = runs/robustness/<timestamp> by default):
  <out>/<condition>/<policy>/   episodes.csv, summary.json, eval.log (one evaluate.py run)
  <out>/manifest.json           the jobs and their status
  <out>/results.csv             one row per (sweep, value, policy) with the scripts/compare.py metrics
  <out>/report.md               per sweep: success / crash / speed tables, method x value (mean ± std over seeds)
  <out>/<sweep>.png             success, crash and speed against the swept value, one line per method
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from compare import METHOD_ORDER, episodes_for, score  # noqa: E402

# Fixed hue per method (validated categorical palette, slots 1-4; color follows the method, never its rank) and a
# marker per method as the second encoding.
STYLE = {"b0": ("#2a78d6", "o"), "m1": ("#eb6834", "s"), "m2": ("#1baf7a", "^"), "m3": ("#eda100", "D")}
PLOT_METRICS = [("success", "success rate (%)", 100), ("crash", "crash rate (%)", 100),
                ("mean_speed", "mean speed (m/s)", 1)]
TABLE_METRICS = [("success", "success %", 100, ".0f"), ("crash", "crash %", 100, ".0f"),
                 ("mean_speed", "speed m/s", 1, ".1f")]


# ------------------------------------------------------------------------------------------------ conditions
def value_str(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def conditions(spec: dict, only: list[str] | None) -> tuple[dict, dict]:
    """-> ({condition id: {key: value}}, {sweep: [(value, condition id), ...]}). Nominal values map to 'nominal'."""
    nominal = spec["nominal"]
    conds, sweeps = {"nominal": {}}, {}
    for name, sw in spec["sweeps"].items():
        if only and name not in only:
            continue
        key = sw["key"]
        if key not in nominal:
            sys.exit(f"configs/robustness.yaml: sweep {name!r} key {key!r} has no nominal value")
        sweeps[name] = []
        for v in sw["values"]:
            if v == nominal[key]:
                cid = "nominal"
            else:
                cid = f"{key.split('.')[-1]}={value_str(v)}"
                conds[cid] = {key: v}
            sweeps[name].append((v, cid))
    return conds, sweeps


# -------------------------------------------------------------------------------------------------- policies
def get_key(cfg: dict, dotted: str):
    for k in dotted.split("."):
        cfg = cfg.get(k, {}) if isinstance(cfg, dict) else {}
    return cfg if cfg != {} else None


def policies(args, nominal: dict) -> list[dict]:
    out = []
    if args.b0:
        out.append({"id": "b0", "method": "b0", "checkpoint": None, "run": None})
    for run in args.runs or []:
        run = Path(run)
        ckpt = run / "checkpoints" / f"{args.checkpoint}.zip"
        cfg_file = run / "config.yaml"
        if not ckpt.exists() or not cfg_file.exists():
            print(f"skip {run}: no {ckpt.name} or config.yaml", file=sys.stderr)
            continue
        cfg = yaml.safe_load(cfg_file.read_text())
        method, seed = cfg["method"]["name"], cfg["experiment"].get("seed", "x")
        diff = {k: get_key(cfg, k) for k, v in nominal.items() if get_key(cfg, k) != v}
        if diff:
            print(f"warning: {run} was trained with {diff}, not the nominal values; its 'nominal' column is "
                  "not its training conditions", file=sys.stderr)
        out.append({"id": f"{method}_s{seed}_{run.name}", "method": method, "checkpoint": str(ckpt), "run": str(run)})
    ids = [p["id"] for p in out]
    if len(ids) != len(set(ids)):
        sys.exit(f"two runs map to the same policy id: {ids}")
    return out


# ------------------------------------------------------------------------------------------------------- run
def run_one(pol: dict, cid: str, overrides: dict, nominal: dict, out: Path, args) -> str:
    d = out / cid / pol["id"]
    if (d / "summary.json").exists():
        return "cached"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPTS / "evaluate.py"), "--method", pol["method"], "--out", str(d),
           "--n-workers", str(args.n_workers)]
    if pol["checkpoint"]:
        cmd += ["--checkpoint", pol["checkpoint"]]
    if args.n_episodes:
        cmd += ["--n-episodes", str(args.n_episodes)]
    sets = {**nominal, **overrides}          # always explicit: a run trained elsewhere still gets the same world
    if not args.full_logs:
        sets["logging.full_log_eval"] = False
    for k, v in sets.items():
        cmd += ["--set", f"{k}={v}"]
    t0 = time.time()
    with open(d / "eval.log", "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    status = "ok" if p.returncode == 0 and (d / "summary.json").exists() else f"failed (exit {p.returncode})"
    print(f"  {status:<8s} {time.time() - t0:6.0f} s", flush=True)
    return status


def sweep(args) -> Path:
    spec = yaml.safe_load(Path(args.config).read_text())
    conds, sweeps = conditions(spec, args.sweeps.split(",") if args.sweeps else None)
    pols = policies(args, spec["nominal"])
    if not pols:
        sys.exit("no policies: pass --b0 and/or --runs <run dirs>")
    out = Path(args.out) if args.out else REPO / "runs" / "robustness" / time.strftime("%Y%m%d-%H%M%S")
    jobs = [(p, cid) for cid in conds for p in pols]
    print(f"{len(pols)} policies x {len(conds)} conditions = {len(jobs)} evaluations -> {out}")
    if args.dry_run:
        for p, cid in jobs:
            print(f"  {cid:<28s} {p['id']:<40s} {conds[cid] or '(trained conditions)'}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": spec, "sweeps": {k: [[v, c] for v, c in vs] for k, vs in sweeps.items()},
                "conditions": conds, "policies": pols, "n_episodes": args.n_episodes, "jobs": {}}
    for i, (p, cid) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {cid} / {p['id']}", flush=True)
        manifest["jobs"][f"{cid}/{p['id']}"] = run_one(p, cid, conds[cid], spec["nominal"], out, args)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    failed = [k for k, s in manifest["jobs"].items() if s.startswith("failed")]
    if failed:
        print(f"{len(failed)} failed (see <out>/<condition>/<policy>/eval.log): {failed}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------------------------------- report
def collect(out: Path) -> pd.DataFrame:
    manifest = json.loads((out / "manifest.json").read_text())
    pols = {p["id"]: p for p in manifest["policies"]}
    rows = []
    for name, entries in manifest["sweeps"].items():
        for pos, (value, cid) in enumerate(entries):
            for pid, pol in pols.items():
                d = out / cid / pid
                if not (d / "summary.json").exists():
                    continue
                df, _ = episodes_for(d, None, use_final_eval=False)
                rows.append({"sweep": name, "value": value_str(value), "pos": pos, "condition": cid,
                             "method": pol["method"], "policy": pid, **score(df)})
    return pd.DataFrame(rows)


def method_key(m: str):
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m)


def plot_sweep(res: pd.DataFrame, name: str, key: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = res.drop_duplicates("pos").sort_values("pos")["value"].tolist()
    x = np.arange(len(values))
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(4.2 * len(PLOT_METRICS), 3.6))
    methods = sorted(res["method"].unique(), key=method_key)
    dodge = {m: (i - (len(methods) - 1) / 2) * 0.05 for i, m in enumerate(methods)}   # ties stay visible
    for ax, (col, label, scale) in zip(axes, PLOT_METRICS):
        ends = []
        for m in methods:
            g = res[res["method"] == m].groupby("pos")[col]
            mean = g.mean().reindex(range(len(values))) * scale
            std = g.std(ddof=1).reindex(range(len(values))) * scale
            color, marker = STYLE.get(m, ("#6250d6", "v"))
            xm = x + dodge[m]
            ax.plot(xm, mean, color=color, marker=marker, markersize=7, linewidth=2, label=m.upper(),
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            if std.notna().any():
                ax.fill_between(xm, mean - std.fillna(0), mean + std.fillna(0), color=color, alpha=0.15, lw=0)
            last = mean.last_valid_index()
            if last is not None:
                ends.append((float(mean[last]), m.upper()))
        # direct labels at the right edge; labels closer than 4 % of the axis span merge into one ("B0, M2")
        lo, hi = ax.get_ylim() if scale != 100 else (-5, 105)
        groups = []
        for y, lab in sorted(ends):
            if groups and y - groups[-1][0][-1] < 0.04 * (hi - lo):
                groups[-1][0].append(y)
                groups[-1][1].append(lab)
            else:
                groups.append(([y], [lab]))
        for ys, names in groups:
            ax.annotate(", ".join(names), (len(values) - 1 + max(dodge.values()), float(np.mean(ys))),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color="#3d3d3a")
        ax.set_xticks(x, values)
        ax.set_xlim(-0.3, len(values) - 0.3)
        ax.set_xlabel(key)
        ax.set_title(label, fontsize=10, color="#1f1f1e", loc="left")
        if scale == 100:
            ax.set_ylim(-5, 105)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ac")
        ax.tick_params(colors="#5f5e5a", labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Robustness: {name}  (band = ±1 std over seeds)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(out: Path):
    res = collect(out)
    if res.empty:
        sys.exit(f"no finished evaluations in {out}")
    res.to_csv(out / "results.csv", index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    keys = {name: sw["key"] for name, sw in manifest["config"]["sweeps"].items()}
    nominal = manifest["config"]["nominal"]
    lines = [f"# Robustness sweep — {out.name}", "",
             "Frozen policies evaluated under changed conditions; one setting varied per sweep, the rest nominal "
             f"({', '.join(f'{k}={v}' for k, v in nominal.items())}). Cells are mean ± std over seeds "
             "(± omitted with one seed). `*` marks the trained (nominal) value. Crash = collision + e-brake + "
             "off-road. Episodes per cell: " + str(int(res["n_episodes"].max())) + ".", ""]
    for name in manifest["sweeps"]:
        r = res[res["sweep"] == name]
        if r.empty:
            continue
        key = keys[name]
        vals = r.drop_duplicates("pos").sort_values("pos")
        heads = [v + ("*" if c == "nominal" else "") for v, c in zip(vals["value"], vals["condition"])]
        lines += [f"## {name} (`{key}`)", ""]
        for col, title, scale, f in TABLE_METRICS:
            lines += [f"**{title}**", "", "| method | " + " | ".join(heads) + " |",
                      "|---|" + "---|" * len(heads)]
            for m in sorted(r["method"].unique(), key=method_key):
                cells = []
                for pos in vals["pos"]:
                    x = r[(r["method"] == m) & (r["pos"] == pos)][col] * scale
                    if x.notna().sum() == 0:
                        cells.append("–")
                    elif x.notna().sum() == 1:
                        cells.append(format(x.mean(), f))
                    else:
                        cells.append(f"{format(x.mean(), f)} ± {format(x.std(ddof=1), f)}")
                lines.append(f"| {m.upper()} | " + " | ".join(cells) + " |")
            lines.append("")
        plot_sweep(r, name, key, out / f"{name}.png")
        lines += [f"![{name}]({name}.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out}/report.md, results.csv and {len(manifest['sweeps'])} plots")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="training run folders (each with checkpoints/, config.yaml)")
    p.add_argument("--b0", action="store_true", help="include the rule-based B0 baseline")
    p.add_argument("--checkpoint", default="best", choices=["best", "final"], help="which checkpoint of each run")
    p.add_argument("--config", default=str(REPO / "configs" / "robustness.yaml"))
    p.add_argument("--sweeps", default=None, help="comma-separated subset of sweeps (default: all)")
    p.add_argument("--n-episodes", type=int, default=None, help="first N eval seeds only (default: all)")
    p.add_argument("--n-workers", type=int, default=4, help="parallel MetaDrive workers per evaluation")
    p.add_argument("--full-logs", action="store_true", help="keep per-step .npz logs (needed for replay; large)")
    p.add_argument("--out", default=None, help="output folder (reuse one to resume)")
    p.add_argument("--dry-run", action="store_true", help="list the evaluations and exit")
    p.add_argument("--report", type=Path, default=None, help="only rebuild report.md / plots for this folder")
    args = p.parse_args()

    if args.report:
        report(args.report)
        return
    out = sweep(args)
    if not args.dry_run:
        report(out)


if __name__ == "__main__":
    main()
