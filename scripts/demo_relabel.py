#!/usr/bin/env python
"""Demo of the human-drive relabelling (docs/human-data-pipeline.md, step 5): one human drive, then the same drive
as M2 and as M3 would fly it, from the labels scripts/build_demos.py makes for their behaviour cloning.

    python scripts/demo_relabel.py                                   # seed 428 demo: human, M2, M3 (3D window)
    python scripts/demo_relabel.py data/human_demos/drive_eval_r0_e000000_s11.npz --fps 0
    python scripts/demo_relabel.py --methods m3 --no-human --no-render              # plot + numbers only
    python scripts/demo_relabel.py --set method.ebrake.ends_episode=false           # M3 brakes instead of ending

1. human: the logged [steering, throttle_brake] replayed on the same seed (exactly the recorded drive).
2. M2 / M3: at every decision the label is looked up at the point of the human's path nearest the car (by distance
   along the road, so a car that falls behind the human still follows the human's route): M2 (d*, v*) = the human's
   lateral offset and speed 2 s further on, flown by the quintic + PID; M3 = 5 waypoints on the human's path at
   speed x 2 s spacing (smoothed where M3's lateral-accel check would reject them) + v*, flown by the spline + MPC.
   Traffic is the seed's (same cars as the human met), so a controller that lags the human can meet them differently.
Output: runs/demo_relabel/<ts>/ : episode logs, path.png (lateral offset and speed vs distance, all three).
Keys (3D window): X = end the episode, M = overlays on / off.
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT = "data/human_demos/drive_eval_r0_e000005_s428.npz"


def load_build_demos():
    spec = importlib.util.spec_from_file_location("build_demos", Path(__file__).with_name("build_demos.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", nargs="?", default=DEFAULT, help="a human drive log (data/human_demos/*.npz)")
    p.add_argument("--methods", nargs="+", default=["m2", "m3"], choices=["m2", "m3"])
    p.add_argument("--no-human", action="store_true", help="skip showing the human drive (it is still replayed)")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--fps", type=float, default=10.0, help="10 = real time, 0 = as fast as possible")
    p.add_argument("--m3-spacing-s", type=float, default=2.0)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config overrides for the M2 / M3 runs")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    render = not args.no_render
    if render and os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ.pop("SDL_VIDEODRIVER")
    elif not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from handoff.config import load_config
    from handoff.env import make_env, method_class
    from handoff.logs import load_episode
    from handoff.viz import Viewer, drive_episode

    bd = load_build_demos()
    out = Path(args.out or f"runs/demo_relabel/{time.strftime('%Y%m%d-%H%M%S')}")
    log = load_episode(args.episode)
    seed = int(log["meta"]["seed"])
    raw = np.stack([log["data"]["step_steering"], log["data"]["step_throttle_brake"]], axis=1).astype(np.float32)
    runs = {}

    # ---------------------------------------------------------------- 1. the human drive (replayed, recorded)
    show_h = render and not args.no_human
    env = make_env(load_config("m1", overrides=["method.accel_range=null"]), render=show_h, is_eval=True,
                   run_dir=out / "human")
    state, k = [], [0]

    def human(_obs):
        state.append((env.ego.s, env.ego.d, env.ego.v))
        a = raw[min(k[0], len(raw) - 1)]
        k[0] += 1
        return a

    print(f"human drive, seed {seed}: {args.episode}")
    runs["human"] = drive_episode(env, human, seed, "human", Viewer(fps=args.fps, show_return=True) if show_h else None)
    state.append((env.ego.s, env.ego.d, env.ego.v))
    road = SimpleNamespace(d_min=env.road.d_min, d_max=env.road.d_max)
    env.close()
    st = np.asarray(state, np.float64)
    S = np.maximum.accumulate(st[:, 0])
    r = {"state": st, "raw": raw[: len(st) - 1]}

    # ---------------------------------------------------------------- 2. the same drive as M2 / M3
    for m in args.methods:
        cfg = load_config(m, overrides=args.overrides)
        stub = SimpleNamespace(dt=float(cfg["sim"]["control_dt"]), v_cap=cfg["sim"]["v_cap_km_h"] / 3.6,
                               max_steering_rad=np.deg2rad(cfg["sim"]["ego"]["max_steering_deg"]), road=road, ego=None)
        labeller = method_class(m)(cfg, stub)
        labeller.period = 1                      # a label for every human step; looked up by position below
        bd.labels_m3.smoothed = 0
        idx, act, _, _, dropped = bd.LABELS[m](r, cfg, labeller, SimpleNamespace(m3_spacing_s=args.m3_spacing_s,
                                                                                  m3_infeasible="smooth",
                                                                                  m3_speed_ahead_s=3.0))
        n_smooth = getattr(bd.labels_m3, "smoothed", 0) if m == "m3" else 0
        env = make_env(cfg, render=render, is_eval=True, run_dir=out / m)

        def plan(_obs, idx=idx, act=act, env=env, m=m):
            j = int(np.clip(np.searchsorted(S[idx], env.ego.s), 0, len(idx) - 1))   # human's point at this s
            a = act[j]
            if m != "m3":
                return a
            # the label was smoothed for the HUMAN's state; re-check it from the car's actual state (as a policy
            # would choose it) and re-smooth if M3's feasibility check would reject it
            from handoff.m3 import plan_lat_accel

            meth, ego = env.method, env.ego
            slots, k, v = meth.decode(a)
            wps = slots[:k]
            if plan_lat_accel(ego, wps, v, meth.corridor.brake) <= meth.lat_accel_limit:
                return a
            d = bd.smooth_waypoints(ego, wps[:, 0], wps[:, 1], v, meth)
            return meth.encode(np.diff(np.concatenate([[ego.s], wps[:, 0]])).tolist(), d.tolist(), v)

        print(f"{m}: {len(idx)} labels ({n_smooth} M3 plans smoothed, {dropped} dropped); driving ...")
        runs[m] = drive_episode(env, plan, seed, m, Viewer(fps=args.fps, show_return=True) if render else None)
        env.close()

    # ---------------------------------------------------------------- summary + plot
    print(f"\n{'':6s} {'end':>12s} {'time s':>7s} {'km/h':>6s} {'dist m':>7s} {'lane ch':>7s} {'return':>8s}")
    for name, s in runs.items():
        print(f"{name:6s} {str(s['termination']):>12s} {s['sim_time']:7.1f} {s['mean_speed'] * 3.6:6.1f} "
              f"{s['distance']:7.0f} {int(s['lane_changes']):7d} {s['return']:8.1f}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"human": "tab:blue", "m2": "tab:orange", "m3": "tab:green"}
    for name, s in runs.items():
        d = load_episode(s["log_file"])["data"]
        c = colors.get(name)
        ax[0].plot(d["step_s"], d["step_d"], color=c, lw=1.8, label=name, zorder=3)
        ax[1].plot(d["step_s"], d["step_v"] * 3.6, color=c, label=name)
        start = d["dec_start_step"].astype(int)
        s0, d0 = d["step_s"][np.maximum(start - 1, 0)], d["step_d"][np.maximum(start - 1, 0)]   # ego at decision
        if name == "m3":      # every decision's waypoints (dots); every 3rd plan drawn from the ego (lines)
            wps = d["dec_waypoints"]                       # (n_dec, 5, 2), NaN = slot not used
            ax[0].plot(wps[:, :, 0].ravel(), wps[:, :, 1].ravel(), ".", color=c, ms=3, alpha=0.35, zorder=2,
                       label="m3 waypoints (every decision)")
            for j in range(0, len(wps), 3):
                ok = ~np.isnan(wps[j, :, 0])
                ax[0].plot(np.r_[s0[j], wps[j, ok, 0]], np.r_[d0[j], wps[j, ok, 1]], "-o", color=c, ms=3.5,
                           lw=0.8, alpha=0.6, zorder=2, label="m3 plan (every 3rd decision)" if j == 0 else None)
        elif name == "m2":    # d* targets, drawn where they are reached: 2 s ahead at the ego's speed
            dec = d["dec_decoded"]
            v_dec = d["step_v"][np.maximum(start - 1, 0)]
            ax[0].plot(s0 + 2.0 * v_dec, dec[:, 0], "x", color=c, ms=4, alpha=0.6, zorder=2,
                       label="m2 d* targets (2 s ahead)")
    for c in (0.0, -3.5, -7.0):
        ax[0].axhline(c, color="0.85", lw=0.8, zorder=0)
    ax[0].set_ylabel("lateral offset d (m)")
    ax[1].set_ylabel("speed (km/h)")
    ax[1].set_xlabel("distance along the road s (m)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title(f"seed {seed}: human drive vs. M2 / M3 flying its relabelled actions")
    fig.tight_layout()
    fig.savefig(out / "path.png", dpi=120)
    print(f"\nplot: {out / 'path.png'}   logs: {out}/<run>/episodes/")


if __name__ == "__main__":
    main()
