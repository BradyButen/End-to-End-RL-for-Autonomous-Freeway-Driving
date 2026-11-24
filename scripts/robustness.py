#!/usr/bin/env python
"""Robustness sweep: evaluate frozen policies under conditions they were not trained on.

    python scripts/robustness.py --config configs/robust_v2.yaml --b0 --runs runs/m*/r5/seed_*/* --n-workers 16
    python scripts/robustness.py --b0 --runs ... --sweeps density --n-episodes 8 # quick partial check (v1 suite)
    python scripts/robustness.py --config configs/robust_v2.yaml --b0 --runs ... --dry-run   # list the evaluations
    python scripts/robustness.py --report runs/robustness/<name>                # rebuild tables/plots only
    sbatch --export=ALL,POLICY=<run dir or b0> slurm/robustness.sbatch          # Great Lakes: one job per policy

The suite (configs/robustness.yaml = v1, configs/robust_v2.yaml = v2) lists nominal values, one-key sweeps, optional
two-key grids (every combination), optional `fixed` settings for every condition, `eval_seeds` [lo, hi) and
`spawn_at_traffic_speed` (ego spawns at traffic speed + mean lane offset). For every policy x condition,
scripts/evaluate.py runs with those --set overrides on top of the run's own config.yaml. No training happens. A
condition shared by sweeps / grids (same non-nominal values) is evaluated once. Evaluations whose summary.json exists
are skipped, so a rerun with the same --out resumes, and several jobs (e.g. one per policy) can share one --out.

Output (<out> = runs/robustness/<timestamp> by default):
  <out>/spec.yaml               the suite
  <out>/<condition>/<policy>/   episodes.csv, summary.json, eval.log (one evaluate.py run)
  <out>/all_episodes.csv        EVERY episode (one row each): condition, policy, world settings + all episode metrics
  <out>/results.csv             one row per (sweep, value, policy) with the scripts/compare.py metrics (+ n_cars)
  <out>/grid_results.csv        one row per (grid cell, policy)
  <out>/report.md               per sweep: cars on route, success / crash / timeout / speed tables (mean ± std over
                                seeds); per grid: heatmaps and tables per method
  <out>/<sweep>.png             success, crash and speed against the swept value, one line per method
  <out>/<grid>_<metric>.png     heatmap per method (success, crash, speed)
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
                ("mean_speed", "mean speed (km/h)", 3.6)]
TABLE_METRICS = [("success", "success %", 100, ".0f"), ("crash", "crash %", 100, ".0f"),
                 ("timeout", "timeout %", 100, ".0f"), ("mean_speed", "speed km/h", 3.6, ".1f")]
# Grid heatmaps: one-hue sequential ramp (blue 100 -> 700 of the validated palette), light = low.
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
GRID_METRICS = [("success", "success %", 100, (0, 100)), ("crash", "crash %", 100, (0, 100)),
                ("mean_speed", "speed km/h", 3.6, None)]


# ------------------------------------------------------------------------------------------------ conditions
def value_str(v) -> str:
    if isinstance(v, (list, tuple)):
        return "_".join(value_str(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def cond_id(over: dict) -> str:
    """Folder name of a condition: its non-nominal values ('nominal' when there are none). The same values give the
    same id whichever sweep or grid asks for them, so shared conditions run once."""
    return ",".join(f"{k.split('.')[-1]}={value_str(v)}" for k, v in sorted(over.items())) or "nominal"


def conditions(spec: dict, only: list[str] | None) -> tuple[dict, dict, dict]:
    """-> ({condition id: {key: value}}, {sweep: [(value, cid), ...]}, {grid: [(x, y, cid), ...]})."""
    nominal = spec["nominal"]
    conds: dict = {"nominal": {}}

    def add(pairs) -> str:
        for k, _ in pairs:
            if k not in nominal:
                sys.exit(f"robustness config: key {k!r} has no nominal value")
        over = {k: v for k, v in pairs if v != nominal[k]}
        conds[cond_id(over)] = over
        return cond_id(over)

    sweeps = {name: [(v, add([(sw["key"], v)])) for v in sw["values"]]
              for name, sw in (spec.get("sweeps") or {}).items() if not only or name in only}
    grids = {name: [(x, y, add([(g["keys"][0], x), (g["keys"][1], y)])) for x in g["values"][0] for y in g["values"][1]]
             for name, g in (spec.get("grids") or {}).items() if not only or name in only}
    return conds, sweeps, grids


def condition_sets(spec: dict, overrides: dict) -> dict:
    """Every --set of one evaluation: nominal, then the suite's fixed settings, then the condition; the eval seeds;
    the ego's spawn speed at the average road speed when spawn_at_traffic_speed."""
    sets = {**spec["nominal"], **(spec.get("fixed") or {}), **overrides}
    if spec.get("eval_seeds"):
        sets["scenario.seeds.eval"] = list(range(*spec["eval_seeds"]))
    if spec.get("spawn_at_traffic_speed"):
        offsets = sets.get("scenario.traffic.lane_speed_offset_km_h") or [0]
        sets["sim.ego.spawn_speed_km_h"] = float(sets["scenario.traffic.speed_km_h"]) + float(np.mean(offsets))
    return sets



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
def run_one(pol: dict, cid: str, overrides: dict, spec: dict, out: Path, args) -> str:
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
    sets = condition_sets(spec, overrides)   # always explicit: a run trained elsewhere still gets the same world
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
    conds, _, _ = conditions(spec, args.sweeps.split(",") if args.sweeps else None)
    pols = policies(args, spec["nominal"])
    if not pols:
        sys.exit("no policies: pass --b0 and/or --runs <run dirs>")
    out = Path(args.out) if args.out else REPO / "runs" / "robustness" / time.strftime("%Y%m%d-%H%M%S")
    jobs = [(p, cid) for cid in conds for p in pols]
    print(f"{len(pols)} policies x {len(conds)} conditions = {len(jobs)} evaluations -> {out}")
    if args.dry_run:
        for p, cid in jobs:
            print(f"  {cid:<40s} {p['id']:<40s} {conds[cid] or '(trained conditions)'}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    spec_file = out / "spec.yaml"            # the report reads the suite from here; jobs sharing <out> must agree
    if spec_file.exists() and yaml.safe_load(spec_file.read_text()) != spec:
        sys.exit(f"{out} holds a different suite ({spec_file}); use another --out")
    spec_file.write_text(yaml.safe_dump(spec, sort_keys=False))
    failed = []
    for i, (p, cid) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {cid} / {p['id']}", flush=True)
        if run_one(p, cid, conds[cid], spec, out, args).startswith("failed"):
            failed.append(f"{cid}/{p['id']}")
    if failed:
        print(f"{len(failed)} failed (see <out>/<condition>/<policy>/eval.log): {failed}", file=sys.stderr)
    return out



# ---------------------------------------------------------------------------------------------------- report
def load_spec(out: Path) -> dict:
    if (out / "spec.yaml").exists():
        return yaml.safe_load((out / "spec.yaml").read_text())
    return json.loads((out / "manifest.json").read_text())["config"]     # suites run before 2026-11-24


def collect(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per sweep x value x policy, one row per grid cell x policy) from every finished evaluation in
    <out>/<condition>/<policy>/ (policy id = <method>_s<seed>_<run> or b0)."""
    _, sweeps, grids = conditions(load_spec(out), None)

    def rows_for(cid: str, base: dict) -> list[dict]:
        rows = []
        for d in sorted((out / cid).glob("*/summary.json")) if (out / cid).is_dir() else []:
            df, _ = episodes_for(d.parent, None, use_final_eval=False)
            extra = {"n_cars": float(df["n_cars"].mean())} if "n_cars" in df else {}
            rows.append({**base, "condition": cid, "method": d.parent.name.split("_")[0], "policy": d.parent.name,
                         **score(df), **extra})
        return rows

    res = [r for name, entries in sweeps.items() for pos, (v, cid) in enumerate(entries)
           for r in rows_for(cid, {"sweep": name, "value": value_str(v), "pos": pos})]
    gres = [r for name, cells in grids.items() for x, y, cid in cells
            for r in rows_for(cid, {"grid": name, "x": x, "y": y})]
    return pd.DataFrame(res), pd.DataFrame(gres)



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


def plot_grid(g: pd.DataFrame, name: str, keys: list[str], out: Path) -> list[str]:
    """Per metric: one heatmap per method (mean over its seeds) on a shared colour scale, values written in."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    xs, ys = sorted(g["x"].unique()), sorted(g["y"].unique())
    methods = sorted(g["method"].unique(), key=method_key)
    files = []
    for col, label, scale, lim in GRID_METRICS:
        vmin, vmax = lim or (float(g[col].min() * scale), float(g[col].max() * scale))
        fig, axes = plt.subplots(1, len(methods), figsize=(2.9 * len(methods) + 1.2, 3.4), squeeze=False,
                                 sharey=True, layout="constrained")
        for ax, m in zip(axes[0], methods):
            t = (g[g["method"] == m].groupby(["x", "y"])[col].mean().unstack("y")
                 .reindex(index=xs, columns=ys) * scale).to_numpy(dtype=float)
            im = ax.imshow(t, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(xs)):
                for j in range(len(ys)):
                    if np.isfinite(t[i, j]):
                        dark = (t[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55
                        ax.text(j, i, f"{t[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color="white" if dark else "#1f1f1e")
            ax.set_xticks(range(len(ys)), [value_str(y) for y in ys])
            ax.set_yticks(range(len(xs)), [value_str(x) for x in xs])
            ax.set_title(m.upper(), loc="left", fontsize=10, color="#1f1f1e")
            ax.set_xlabel(keys[1].split(".")[-1], color="#5f5e5a")
            ax.tick_params(colors="#5f5e5a", labelsize=8, length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0][0].set_ylabel(keys[0].split(".")[-1], color="#5f5e5a")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label=label)
        fig.suptitle(f"Robustness: {name}, {label}", fontsize=11, x=0.01, ha="left")
        f = f"{name}_{col}.png"
        fig.savefig(out / f, dpi=150)
        plt.close(fig)
        files.append(f)
    return files


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *body, ""]


def cell(x: pd.Series, f: str) -> str:
    x = x.dropna()
    if x.empty:
        return "–"
    return format(x.mean(), f) if len(x) == 1 else f"{format(x.mean(), f)} ± {format(x.std(ddof=1), f)}"


def all_episodes(out: Path, spec: dict) -> pd.DataFrame:
    """Every episode of every evaluation in one table, each row tagged with its condition (the values that differ
    from nominal), the policy and the full world it ran in (density, traffic speed, lane offsets, spawn speed)."""
    conds, _, _ = conditions(spec, None)
    frames = []
    for cid, over in conds.items():
        sets = condition_sets(spec, over)
        world = {"density": sets.get("scenario.traffic.density"),
                 "traffic_speed_km_h": sets.get("scenario.traffic.speed_km_h"),
                 "lane_offsets_km_h": value_str(sets.get("scenario.traffic.lane_speed_offset_km_h") or ""),
                 "spawn_speed_km_h": sets.get("sim.ego.spawn_speed_km_h"),
                 **{k.split(".")[-1]: v for k, v in over.items() if k not in (
                     "scenario.traffic.density", "scenario.traffic.speed_km_h",
                     "scenario.traffic.lane_speed_offset_km_h")}}
        for f in sorted((out / cid).glob("*/episodes.csv")) if (out / cid).is_dir() else []:
            pid = f.parent.name
            df = pd.read_csv(f, low_memory=False)
            tags = {"condition": cid, "policy": pid, "method": pid.split("_")[0],
                    "run_seed": pid.split("_")[1][1:] if pid.count("_") >= 2 else "", **world}
            df = df.drop(columns=[c for c in tags if c in df.columns]).reset_index(drop=True)
            frames.append(pd.concat([pd.DataFrame([tags] * len(df)), df], axis=1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(out: Path):
    spec = load_spec(out)
    res, gres = collect(out)
    if res.empty and gres.empty:
        sys.exit(f"no finished evaluations in {out}")
    eps = all_episodes(out, spec)
    eps.to_csv(out / "all_episodes.csv", index=False)
    print(f"all_episodes.csv: {len(eps)} episodes x {eps.shape[1]} columns")
    res.to_csv(out / "results.csv", index=False)
    if not gres.empty:
        gres.to_csv(out / "grid_results.csv", index=False)
    nominal = spec["nominal"]
    n_eps = int(pd.concat([res, gres])["n_episodes"].max())
    lines = [f"# Robustness sweep — {out.name}", "",
             "Frozen policies evaluated under changed conditions; one setting varied per sweep, the rest nominal "
             f"({', '.join(f'{k}={value_str(v)}' for k, v in nominal.items())}). "
             + (f"Every condition also: {', '.join(f'{k}={v}' for k, v in spec['fixed'].items())}. "
                if spec.get("fixed") else "")
             + ("The ego spawns at the average road speed. " if spec.get("spawn_at_traffic_speed") else "")
             + "Cells are mean ± std over seeds (± omitted with one seed). `*` marks the trained (nominal) value. "
             f"Crash = collision + e-brake + off-road. Episodes per cell: {n_eps}. Per policy: results.csv"
             + (", grid_results.csv" if not gres.empty else "") + ".", ""]
    for name, sw in (spec.get("sweeps") or {}).items():
        r = res[res["sweep"] == name] if not res.empty else res
        if r.empty:
            continue
        vals = r.drop_duplicates("pos").sort_values("pos")
        heads = [v + ("*" if c == "nominal" else "") for v, c in zip(vals["value"], vals["condition"])]
        lines += [f"## {name} (`{sw['key']}`)", ""]
        if "n_cars" in r:
            lines += ["**cars on route** (traffic cars placed per episode)", ""]
            lines += md_table(["", *heads], [["cars", *[cell(r[r["pos"] == p].groupby("policy")["n_cars"].mean()
                                                            .head(1), ".0f") for p in vals["pos"]]]])
        for col, title, scale, f in TABLE_METRICS:
            rows = [[m.upper(), *[cell(r[(r["method"] == m) & (r["pos"] == p)][col] * scale, f) for p in vals["pos"]]]
                    for m in sorted(r["method"].unique(), key=method_key)]
            lines += [f"**{title}**", "", *md_table(["method", *heads], rows)]
        plot_sweep(r, name, sw["key"], out / f"{name}.png")
        lines += [f"![{name}]({name}.png)", ""]
    for name, g in (spec.get("grids") or {}).items():
        r = gres[gres["grid"] == name] if not gres.empty else gres
        if r.empty:
            continue
        kx, ky = g["keys"]
        lines += [f"## {name} (rows `{kx}`, columns `{ky}`)", ""]
        for f in plot_grid(r, name, g["keys"], out):
            lines += [f"![{f}]({f})", ""]
        xs, ys = sorted(r["x"].unique()), sorted(r["y"].unique())
        for col, title, scale, f in TABLE_METRICS:
            for m in sorted(r["method"].unique(), key=method_key):
                rm = r[r["method"] == m]
                rows = [[value_str(x), *[cell(rm[(rm["x"] == x) & (rm["y"] == y)][col] * scale, f) for y in ys]]
                        for x in xs]
                lines += [f"**{m.upper()}: {title}**", "",
                          *md_table([f"{kx.split('.')[-1]} \\ {ky.split('.')[-1]}", *map(value_str, ys)], rows)]
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out}/report.md and results.csv" + (", grid_results.csv" if not gres.empty else ""))



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
