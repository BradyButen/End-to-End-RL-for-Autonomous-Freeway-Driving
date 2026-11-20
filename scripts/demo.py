#!/usr/bin/env python
"""Demo: a trained policy drives eval seeds in the MetaDrive 3D window with everything the RL and the controller
are doing drawn on top (m3_harness-style), then an overview plot per episode.

    python scripts/demo.py runs/gl_round2/m3 --seeds 1003 1027          # final.zip (or best.zip if no final)
    python scripts/demo.py runs/gl_round2/m2 --checkpoint best --seeds 1038
    python scripts/demo.py runs/gl_round2/m1 --seeds 1012 --fps 0       # as fast as it renders
    python scripts/demo.py runs/gl_round2/m3 --seeds 1003 --no-render   # headless: CSVs + plot only

The run's own config.yaml is used and the env is the eval env (deterministic policy, same seeds as the eval),
so an episode here is the same episode the eval scored.

Overlays (M = toggle overlays, X = end this episode and save, avoid R / Esc):
  all     white line     ego trail (last 20 s)
  M3      white balls    RL waypoints in use (k of 5)        grey balls   slots the RL switched off with k
          yellow line    spline reference                    cyan line    MPC predicted path (5 s)
          blue lines     corridor over the MPC horizon (left/right bound per stage, cross lines = car faces)
          magenta boxes  car keep-out zones                  orange line  emergency-gate check (ego -> next
                                                                          waypoint; red while the gate fires)
  M2      yellow line    quintic reference (d* over 2 s + speed ramp)
          white ball     target d* at the end of the reference   magenta ball  PID's tracking point (ref at ego s)
  M1      nothing to plan: text panel shows the raw steering / throttle the policy outputs.
Text panel: speed, RL command (v*, k, waypoints / d*, v* / steering, throttle), controller state, lead car gap and
closing speed, reward terms.

Writes <out>/<method>_seed<seed>_<time>/{steps.csv, decisions.csv, overview.png} (default out: runs/demo).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WHITE, GREY = (1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 1.0)
YELLOW, CYAN = (1.0, 0.85, 0.1, 1.0), (0.1, 0.9, 1.0, 1.0)
BLUE, MAGENTA = (0.3, 0.5, 1.0, 1.0), (1.0, 0.3, 1.0, 1.0)
ORANGE, RED, TRAIL = (1.0, 0.6, 0.2, 1.0), (1.0, 0.15, 0.15, 1.0), (0.85, 0.85, 0.85, 1.0)
Z = 0.6


def lead_car(ego, objects):
    """Nearest car ahead overlapping the ego laterally: (gap bumper-to-bumper, closing speed, car) or None."""
    best = None
    for o in objects:
        if o.s > ego.s and abs(o.d - ego.d) < 0.5 * (o.width + ego.width) + 0.3:
            gap = o.s - ego.s - 0.5 * (o.length + ego.length)
            if best is None or gap < best[0]:
                best = (gap, ego.v - o.v, o)
    return best


class Drawer:
    """Thin wrapper over MetaDrive's line / point drawers (same setup as simulator.overlays.OverlayDrawer)."""

    def __init__(self, engine):
        self.lines = engine.make_line_drawer(thickness=3.0)
        self.lines.setShaderOff(1)        # line geometry has no normals: draw unlit or it is invisible
        self.lines.setLightOff(1)
        self.points = engine.make_point_drawer(scale=4.0)

    def draw(self, lines, points):
        self.lines.reset()
        self.points.reset()
        ls, cs = [], []
        for xy, col in lines:
            xy = np.asarray(xy, float).reshape(-1, 2)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) >= 2:
                ls.append([(float(x), float(y), Z) for x, y in xy])
                cs.append([col] * (len(xy) - 1))
        if ls:
            self.lines.draw_lines(ls, cs)
        pts = [((float(x), float(y), Z + 0.3), col) for (x, y), col in points if np.isfinite(x) and np.isfinite(y)]
        if pts:
            self.points.draw_points([p for p, _ in pts], [c for _, c in pts])

    def clear(self):
        self.lines.reset()
        self.points.reset()


def box_xy(road, s_lo, s_hi, d_lo, d_hi):
    x, y = road.to_world(np.array([s_lo, s_hi, s_hi, s_lo, s_lo]), np.array([d_lo, d_lo, d_hi, d_hi, d_lo]))
    return np.stack([x, y], axis=1)


def corridor_lines(mpc, road):
    """The road corridor the MPC is solving with (as in m3_harness): per stage the left/right bound on the ego
    centre at that stage's planned position, joined into two ribbons; car-made front/back faces as cross lines."""
    from handoff.rl.methods.m3_corridor import RoadCorridor

    tube = mpc.tube
    if tube is None or getattr(tube, "extents", None) is None:
        return []
    st = min(max(mpc._tube_age - 1, 0), len(tube.traj) - 1)
    big = 0.5 * RoadCorridor.BIG
    idx = np.minimum(np.arange(mpc.N + 1) + st, len(tube.traj) - 1)
    out, left, right, cuts = [], [], [], set()

    def flush():
        for side in (left, right):
            if len(side) >= 2:
                out.append(np.array(side))
            side.clear()

    for k in idx:
        if tube.blocked[k]:
            flush()
            continue
        sk, dk = float(tube.traj[k, 0]), float(tube.traj[k, 1])
        um, up, wm, wp = tube.extents[k]
        left.append(road.to_world(sk, dk + float(wp)))
        right.append(road.to_world(sk, dk + float(wm)))
        for ext in (um, up):
            key = round(sk + float(ext), 1)
            if abs(ext) < big and key not in cuts:
                cuts.add(key)
                x, y = road.to_world(np.full(2, sk + float(ext)), np.array([dk + float(wm), dk + float(wp)]))
                out.append(np.stack([x, y], axis=1))
    flush()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory (has config.yaml and checkpoints/)")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: first 3 eval seeds")
    p.add_argument("--checkpoint", default=None, help="final | best | path to .zip (default final, else best)")
    p.add_argument("--fps", type=float, default=10.0, help="10 = real time, 0 = as fast as possible")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--stochastic", action="store_true", help="sample actions (training behaviour) instead of mean")
    p.add_argument("--out", default="runs/demo")
    args = p.parse_args()

    import yaml

    from handoff.rl.env.factory import make_env
    from handoff.simulator.viewer import load_policy_fn

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    method = cfg["method"]["name"]
    ck = args.checkpoint or ("final" if (run / "checkpoints" / "final.zip").exists() else "best")
    ckpt = run / "checkpoints" / f"{ck}.zip" if ck in ("final", "best") else Path(ck)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    seeds = args.seeds or cfg["scenario"]["seeds"]["eval"][:3]
    render = not args.no_render
    print(f"{method}: {ckpt} ({'stochastic' if args.stochastic else 'deterministic'}), seeds {seeds}")

    env = make_env(cfg, render=render, is_eval=True)
    core = env.core
    policy = load_policy_fn(str(ckpt), deterministic=not args.stochastic)
    state = {"drawer": None, "show": True, "finish": False, "keys": False}

    for seed in seeds:
        steps, decisions, trail = [], [], []
        dec_info: dict = {}
        state["finish"] = False
        t_last = [time.time()]

        def hook(core_, ctx, diag, steps=steps, decisions=decisions, trail=trail, dec_info=dec_info):
            ego, road, objs = core_.ego, core_.road, core_.objects
            t = core_.episode_control_steps * core_.dt
            dec = (ctx or {}).get("decision") or {}
            if (ctx or {}).get("first_step"):
                dec_info.clear()
                row = {"t": t, "s": ego.s, "d": ego.d, "v": ego.v}
                if method == "m3":
                    wps = np.asarray(dec["waypoints_raw"]).reshape(-1, 2)
                    vec = np.asarray(dec["vector"], float)
                    dec_info.update(k=int(dec["k"]), v_star=float(dec["v_target"]), wps=wps,
                                    unused=vec[: 2 * env.n_slots].reshape(-1, 2)[int(dec["k"]):])
                    row.update(k=int(dec["k"]), v_star=float(dec["v_target"]),
                               waypoints=json.dumps(np.round(wps, 2).tolist()))
                elif method == "m2":
                    dec_info.update(d_star=float(dec["d_target"]), v_star=float(dec["v_target"]))
                    row.update(d_star=float(dec["d_target"]), v_star=float(dec["v_target"]))
                decisions.append(row)

            lead = lead_car(ego, objs)
            gated = bool(diag.get("gated_brake"))
            terms = {n: v[-1] for n, v in core_.recorder.step_terms.items()
                     if v and core_.reward.weights.get(n, 0) != 0}
            steps.append({
                "t": t, "x": ego.x, "y": ego.y, "s": ego.s, "d": ego.d, "psi": ego.psi, "v": ego.v,
                "steering": ego.steering, "throttle_brake": ego.throttle_brake,
                "v_star": dec_info.get("v_star", np.nan), "d_star": dec_info.get("d_star", np.nan),
                "k": dec_info.get("k", np.nan),
                "ref_d": diag.get("ref_d", np.nan), "ref_v": diag.get("ref_v", np.nan),
                "tracking_error": diag.get("tracking_error", np.nan),
                "heading_error": diag.get("heading_error", np.nan), "speed_error": diag.get("speed_error", np.nan),
                "solve_ms": 1000 * float(diag.get("solve_time", 0.0) or 0.0),
                "cpu_ms": 1000 * float(diag.get("cpu_time", 0.0) or 0.0),
                "status": str(diag.get("status", "")), "fallback": int(bool(diag.get("fallback"))),
                "gated_brake": int(gated), "tube_blocked": diag.get("tube_blocked", np.nan),
                "lead_gap": lead[0] if lead else np.nan, "lead_closing": lead[1] if lead else np.nan,
                "lead_v": lead[2].v if lead else np.nan, "n_objects": len(objs),
                "objects": json.dumps([[o.id, round(o.s, 1), round(o.d, 2)] for o in objs]),
                **{f"r_{n}": v for n, v in terms.items()},
            })
            trail.append((ego.x, ego.y))
            if not render:
                return

            if not state["keys"]:
                eng = core_.md.engine
                eng.accept("x", lambda: state.__setitem__("finish", True))
                eng.accept("m", lambda: state.__setitem__("show", not state["show"]))
                state["keys"] = True
            if state["drawer"] is None:
                state["drawer"] = Drawer(core_.md.engine)
            drawer = state["drawer"]

            text = {"method": f"{method}  seed {seed}", "t": f"{t:5.1f} s",
                    "speed": f"{ego.v * 3.6:5.1f} km/h", "s / d": f"{ego.s:6.1f} / {ego.d:5.2f} m"}
            lines, points = [], []
            if len(trail) > 1:
                lines.append((np.array(trail[-200:]), TRAIL))
            ref = env.reference
            if method == "m3":
                text["RL v* (max speed)"] = f"{dec_info.get('v_star', 0) * 3.6:5.1f} km/h"
                text["RL waypoints k"] = f"{dec_info.get('k', 0)} of {env.n_slots}"
                for j, (s, d) in enumerate(dec_info.get("wps", [])):
                    text[f"  wp{j + 1} ahead / d"] = f"{s - ego.s:5.1f} / {d:5.2f} m"
                text["MPC"] = f"{str(diag.get('status', ''))[:22]}  {1000 * float(diag.get('solve_time', 0)):4.0f} ms"
                if diag.get("fallback"):
                    text["FALLBACK"] = f"{diag.get('fallback_reason', '')} / {diag.get('fallback_mode', '')}"
                text["corridor car faces"] = diag.get("tube_blocked", "-")
                text["EMERGENCY BRAKE"] = "GATE FIRED" if gated else "-"
                if ref is not None:
                    m = (ref.s >= ego.s - 5) & (ref.s <= ego.s + 150)
                    lines.append((np.stack([ref.x[m], ref.y[m]], axis=1), YELLOW))
                if diag.get("pred_xy") is not None:
                    lines.append((diag["pred_xy"], CYAN))
                lines += [(seg, BLUE) for seg in corridor_lines(env.mpc, road)]
                if hasattr(env.mpc.corridor, "zone_half"):
                    for o in objs:
                        zl, zw = env.mpc.corridor.zone_half(o)
                        lines.append((box_xy(road, o.s - zl, o.s + zl, o.d - zw, o.d + zw), MAGENTA))
                gw = np.asarray(env._gate_wps).reshape(-1, 2)
                ahead = gw[gw[:, 0] > ego.s]
                if len(ahead):
                    x1, y1 = road.to_world(float(ahead[0, 0]), float(ahead[0, 1]))
                    lines.append((np.array([[ego.x, ego.y], [x1, y1]]), RED if gated else ORANGE))
                for s, d in dec_info.get("wps", []):
                    points.append((road.to_world(float(s), float(d)), WHITE))
                for s, d in dec_info.get("unused", []):
                    points.append((road.to_world(float(s), float(d)), GREY))
            elif method == "m2":
                text["RL d* / v*"] = f"{dec_info.get('d_star', 0):5.2f} m / {dec_info.get('v_star', 0) * 3.6:5.1f} km/h"
                text["ref d / v"] = f"{diag.get('ref_d', 0):5.2f} m / {diag.get('ref_v', 0) * 3.6:5.1f} km/h"
                text["cross-track / heading err"] = (f"{diag.get('cross_track', 0):+5.2f} m / "
                                                     f"{np.rad2deg(diag.get('heading_error', 0)):+5.1f} deg")
                text["speed err"] = f"{diag.get('speed_error', 0) * 3.6:+5.1f} km/h"
                if ref is not None:
                    lines.append((np.stack([ref.x, ref.y], axis=1), YELLOW))
                    points.append(((ref.x[-1], ref.y[-1]), WHITE))
                    r = ref.at_s(ego.s)
                    points.append((road.to_world(float(ego.s), float(r["d"])), MAGENTA))
            text["steering / throttle"] = f"{ego.steering:+5.2f} / {ego.throttle_brake:+5.2f}"
            if lead:
                text["lead car gap / closing"] = f"{lead[0]:5.1f} m / {lead[1] * 3.6:+5.1f} km/h"
            for n, v in terms.items():
                text[f"r.{n}"] = f"{v:+.3f}"
            text["keys"] = "M overlays, X end episode"
            if state["show"]:
                drawer.draw(lines, points)
                core_.md.render(text=text)
            else:
                drawer.clear()
                core_.md.render()
            if args.fps > 0:
                wait = 1.0 / args.fps - (time.time() - t_last[0])
                if wait > 0:
                    time.sleep(wait)
            t_last[0] = time.time()

        core.overlay_hook = hook
        obs, _ = env.reset(options={"episode_seed": int(seed), "tag": "demo"})
        summary = {}
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                summary = info.get("episode_summary", {})
                break
            if state["finish"]:
                summary = {"termination": "user_finish"}
                break
        save(args, cfg, method, seed, steps, decisions, summary, core.road)

    env.close()


def save(args, cfg, method, seed, steps, decisions, summary, road):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out) / f"{method}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("steps.csv", steps), ("decisions.csv", decisions)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    if not steps:
        return
    c = {k: np.array([r.get(k, np.nan) for r in steps]) for k in steps[0] if k not in ("status", "objects")}
    t = c["t"]
    term = summary.get("termination", "?")
    print(f"seed {seed}: {term}, {t[-1]:.1f} s, mean {np.mean(c['v']) * 3.6:.1f} km/h, "
          f"return {summary.get('return', float('nan')):.1f} -> {out}/")

    n_rows = 6 if method != "m1" else 4
    fig, ax = plt.subplots(n_rows, 1, figsize=(14, 3 + 2.3 * n_rows),
                           gridspec_kw={"height_ratios": [2.4] + [1] * (n_rows - 1)})
    # 1) road map in (s, d): lanes, traffic tracks, ego path, reference, RL commands
    a = ax[0]
    a.axhspan(road.d_min, road.d_max, color="0.94")
    a.axhline(road.d_max, color="goldenrod", lw=2)
    a.axhline(road.d_min, color="0.3", lw=2)
    for lc in road.lane_centers[:-1]:
        a.axhline(lc - 1.75, color="0.6", lw=0.8, ls="--")
    tracks: dict = {}
    for r in steps:
        for oid, s, d in json.loads(r["objects"]):
            tracks.setdefault(oid, []).append((s, d))
    for tr in tracks.values():
        tr = np.array(tr)
        a.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=0.6, alpha=0.5)
    a.plot([], [], color="tab:blue", lw=0.6, label="traffic (while within 50 m)")
    if method == "m3":
        for dr in decisions:
            w = np.array(json.loads(dr["waypoints"])).reshape(-1, 2)
            a.plot(w[:, 0], w[:, 1], ".", color="crimson", ms=3, alpha=0.5)
        a.plot([], [], ".", color="crimson", label="RL waypoints (every decision)")
    if method in ("m2", "m3"):
        a.plot(c["s"], c["ref_d"], "--", color="goldenrod", lw=1.2, label="reference d at ego s")
    if method == "m2":
        a.step(c["s"], c["d_star"], where="post", color="crimson", lw=1, label="RL d*")
    a.plot(c["s"], c["d"], color="k", lw=1.6, label="ego")
    bad = c["gated_brake"] + c["fallback"] > 0
    if bad.any():
        a.plot(c["s"][bad], c["d"][bad], "x", color="red", label="gate / fallback")
    a.set_xlim(c["s"][0] - 5, c["s"][-1] + 40)
    a.set_ylabel("d (m)")
    a.legend(loc="upper right", fontsize=8, ncol=3)
    a.set_title(f"{method} seed {seed}: {term}, {t[-1]:.1f} s, mean {np.mean(c['v']) * 3.6:.1f} km/h, "
                f"return {summary.get('return', float('nan')):.1f}  (x axis: s)")
    # 2) speed
    a = ax[1]
    a.plot(t, c["v"] * 3.6, "k", label="ego v")
    if method != "m1":
        a.plot(t, c["v_star"] * 3.6, color="crimson", lw=1, label="RL v*")
        a.plot(t, c["ref_v"] * 3.6, "--", color="goldenrod", lw=1, label="controller ref v")
    a.plot(t, c["lead_v"] * 3.6, color="tab:blue", lw=1, label="lead car v")
    a.set_ylabel("km/h")
    a.legend(fontsize=8, ncol=4)
    # 3) lateral / commands
    a = ax[2]
    a.plot(t, c["d"], "k", label="ego d")
    if method != "m1":
        a.plot(t, c["ref_d"], "--", color="goldenrod", label="ref d")
    if method == "m2":
        a.step(t, c["d_star"], where="post", color="crimson", lw=1, label="RL d*")
    a.set_ylabel("d (m)")
    a.legend(fontsize=8, ncol=3)
    a = ax[3]
    a.plot(t, c["steering"], label="steering")
    a.plot(t, c["throttle_brake"], label="throttle_brake")
    a.set_ylabel("command" + (" (= RL action)" if method == "m1" else ""))
    a.legend(fontsize=8, ncol=2)
    if method != "m1":
        a = ax[4]
        a.plot(t, c["lead_gap"], color="tab:blue", label="lead gap (m)")
        a.set_ylabel("gap (m)")
        a2 = a.twinx()
        a2.plot(t, c["lead_closing"] * 3.6, color="tab:red", lw=0.8, label="closing (km/h)")
        a2.set_ylabel("closing km/h")
        a.legend(fontsize=8, loc="upper left")
        a2.legend(fontsize=8, loc="upper right")
        a = ax[5]
        if method == "m3":
            a.plot(t, c["solve_ms"], label="MPC solve (ms)")
            a.plot(t, c["cpu_ms"], lw=0.8, label="cpu (ms)")
            a.axhline(1000 * float(cfg["method"]["mpc"]["solver"]["max_cpu_time"]), color="tab:red", ls=":")
            a2 = a.twinx()
            a2.step(t, c["k"], where="post", color="tab:green", lw=1, label="RL k")
            a2.set_ylabel("k")
            a2.legend(fontsize=8, loc="upper right")
            a.set_ylabel("ms")
        else:
            a.plot(t, c["tracking_error"], label="|cross-track| (m)")
            a.plot(t, np.rad2deg(c["heading_error"]) / 10, label="heading err (deg/10)")
            a.plot(t, c["speed_error"], label="speed err (m/s)")
            a.set_ylabel("PID errors")
        a.legend(fontsize=8, loc="upper left")
    for a in ax[1:]:
        a.set_xlim(t[0], t[-1])
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v))
                                                  for k, v in summary.items()}, indent=1) + "\n")


if __name__ == "__main__":
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ.pop("SDL_VIDEODRIVER")
    main()
