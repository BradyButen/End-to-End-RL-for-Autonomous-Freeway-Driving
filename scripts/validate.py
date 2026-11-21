#!/usr/bin/env python
"""Validation: the real M2 / M3 envs (the same code PPO trains) driven by scripted actions instead of a policy.

1. Scenarios (M3, hand-placed waypoints + scripted cars, configs/validation/*.json):
       python scripts/validate.py run configs/validation/overtake.json             # MetaDrive window, real time
       python scripts/validate.py run <scenario> --no-render                      # headless: logs + plots only
       python scripts/validate.py run <scenario> --set method.mpc.weights.lateral=4.0
       python scripts/validate.py run <scenario> --no-constraints                 # MPC without the corridor (K)
       python scripts/validate.py run <scenario> --demo                           # road + cars only (M toggles)
   Every decision (1 s) the next <= 5 placed waypoints more than ds_min ahead go into the RL action exactly as a
   policy would output them (re-spaced to ds in [5, 40] m, d clipped to the action range), with v* = the scenario's
   target speed. The run ends on a crash / off-road, when the ego has stood still for 3 s, or 30 m past the last
   waypoint. An emergency brake does not end the run here (method.ebrake.ends_episode=false) so you see the stop.
   Writes runs/validation/<scenario>/<timestamp>/: episodes/<id>.npz, <id>_overview.png, <id>_steps.csv, result.json.

   Scenario JSON: {"map_seed": 0, "ego_lane": 1, "ego_speed_kmh": 60, "target_speed_kmh": 70,
                   "waypoints": [[s, d], ...], "cars": [[s, d] or [s, d, speed_kmh], ...], "overrides": ["k=v"]}
   Cars are static unless given a speed (then held at that speed along the road).

2. Editor (place waypoints and static cars on the road in (s, d); Run saves and starts `run` in a new process):
       python scripts/validate.py edit --out configs/validation/my_test.json
       python scripts/validate.py edit --scenario configs/validation/overtake.json
   Left click places (mode buttons: waypoint / car; Shift+click or 'c' always places a car, snapped to the lane
   centre); right click deletes the nearest. Turn pan/zoom off before placing. Segments the RL cannot produce
   (spacing outside 5-40 m) are orange; segments needing more lateral acceleration than the MPC allows are red.

3. Checks (pass / fail; each in its own process, report in runs/validation/checks/):
       python scripts/validate.py check                 # all
       python scripts/validate.py check m3 scenarios    # some
   env        ego dynamics pinned, spawn speed / lane, RoadFrame vs MetaDrive, forced off-road ends the episode,
              determinism (same seed + actions -> identical trajectory; replay relies on it)
   m2         PID: lane 1 -> 0 -> 2 -> 1 with speed changes, scripted (d*, v*); cross-track / speed tolerances
   m3         passes a 35 km/h lead; stops for a stopped car in its lane; fallback chain (forced solver failure ->
              replayed plan -> emergency brake) and recovery
   smoke      M1 / M2 / M3 complete random-action episodes; logs consistent (terms, returns, decision counts)
   b0         B0 completes every evaluation seed through the evaluator
   train      PPO entry point end to end for a few hundred control steps per method (plumbing, not learning)
   scenarios  every configs/validation/*.json headless; fails on a collision or off-road
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from handoff.config import REPO_ROOT, kmh, load_config

SCENARIO_DIR = REPO_ROOT / "configs" / "validation"
OUT = REPO_ROOT / "runs" / "validation"
CAR_LENGTH, CAR_WIDTH = 4.515, 1.852     # TrafficDefaultVehicle (same body as the ego)


# ======================================================================================= scenarios
def load_scenario(path) -> dict:
    sc = {"map_seed": 0, "ego_lane": 1, "ego_speed_kmh": 60.0, "target_speed_kmh": 70.0, "waypoints": [],
          "cars": [], "overrides": []}
    if path:
        sc.update(json.loads(Path(path).read_text()))
    return sc


def scenario_config(sc: dict, overrides=()) -> dict:
    """M3 config for a scenario: no traffic, fixed spawn, no time limit, solid lines do not end the run."""
    return load_config("m3", overrides=[
        "scenario.traffic.density=0", f"sim.ego.spawn_lane={int(sc['ego_lane'])}",
        f"sim.ego.spawn_speed_km_h={float(sc['ego_speed_kmh'])}",
        f"sim.target_speed_km_h={float(sc['target_speed_kmh'])}",
        "sim.on_continuous_line_done=false", "sim.horizon=100000", "method.ebrake.ends_episode=false",
        *sc.get("overrides", []), *overrides])


def controlled_env(method: str, ego_lane: int, ego_kmh: float, overrides=(), run_dir=None, render=False):
    """A method env on map seed 0 without MetaDrive traffic, the ego spawned in ``ego_lane`` at ``ego_kmh``
    (add cars after reset with ``env.add_scripted_car``)."""
    from handoff.env import make_env

    cfg = load_config(method, overrides=["scenario.traffic.density=0", f"sim.ego.spawn_lane={ego_lane}",
                                         f"sim.ego.spawn_speed_km_h={float(ego_kmh)}", *overrides])
    return make_env(cfg, is_eval=True, run_dir=run_dir, render=render)


def scripted_action(method, ego_s: float, placed: np.ndarray, v: float) -> np.ndarray:
    """What the RL could output now: the next <= n_slots placed waypoints ahead, re-spaced like ``decode``."""
    lo, hi = method.ds_range
    ahead = placed[placed[:, 0] > ego_s + lo]
    if len(ahead) == 0:   # past the last one: keep straight on at its offset
        ahead = np.array([[ego_s + hi, placed[-1, 1]]])
    ds, prev = [], ego_s
    for s, _ in ahead[: method.n_slots]:
        ds.append(float(np.clip(s - prev, lo, hi)))
        prev += ds[-1]
    return method.encode(ds, list(ahead[: method.n_slots, 1]), v)


def cmd_run(args) -> dict:
    from handoff.env import make_env
    from handoff.viz import Viewer, drive_episode, episode_report

    sc = load_scenario(args.scenario)
    placed = np.array(sorted(sc["waypoints"]), dtype=float).reshape(-1, 2)
    if len(placed) == 0:
        raise SystemExit("scenario has no waypoints; place some with `validate.py edit` first")
    render = not args.no_render
    if render and os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ.pop("SDL_VIDEODRIVER")
    cfg = scenario_config(sc, args.overrides + (["sim.render_clean=true"] if args.demo else []))
    out = Path(args.out) if args.out else OUT / Path(args.scenario).stem / time.strftime("%Y%m%d-%H%M%S")
    env = make_env(cfg, render=render, is_eval=True, run_dir=out)
    method = env.method
    method.mpc.constraints_on = not args.no_constraints
    v_target = kmh(sc["target_speed_kmh"])
    s_stop = None
    viewer = None
    if render:
        viewer = Viewer(fps=args.fps, allow_constraint_toggle=True)
        viewer.show = not args.demo

    def act(_obs):
        return scripted_action(method, env.ego.s, placed, v_target)

    def stop():
        v = env.recorder.steps["v"][-30:]
        if len(v) == 30 and max(v) < 0.1:     # standing still for 3 s: the run cannot progress any further
            return "stopped_emergency_brake" if env.recorder.steps["gated_brake"][-1] > 0 else "stopped"
        return "end_of_waypoints" if env.ego.s >= s_stop else None

    def setup_and_act(obs):    # first decision: the map exists now -> spawn the cars, draw the placed waypoints
        nonlocal s_stop
        if s_stop is None:
            road = env.road
            s_stop = min(float(placed[-1, 0]) + 30.0, road.length - 5.0)
            for car in sc["cars"]:
                env.add_scripted_car(car[0], car[1], car[2] if len(car) > 2 else 0.0)
            if viewer is not None:
                viewer.extra_points = list(zip(*road.to_world(placed[:, 0], placed[:, 1])))
        return act(obs)

    print(f"scenario {args.scenario}: {len(placed)} waypoints, {len(sc['cars'])} cars, target "
          f"{sc['target_speed_kmh']:g} km/h, corridor constraints {'on' if method.mpc.constraints_on else 'off'}")
    try:
        summ = drive_episode(env, setup_and_act, int(sc["map_seed"]), "validate", viewer, stop)
        episode_report(summ, out, env.road, placed=placed, cars=[c[:2] for c in sc["cars"]],
                       solver_limit_s=float(cfg["method"]["mpc"]["solver"]["max_cpu_time"]))
    finally:
        env.close()
    keys = ("termination", "sim_time", "distance", "mean_speed", "min_gap", "fallback_count", "gated_brake_count",
            "solve_time_mean", "solve_time_p95", "solve_time_max", "tracking_error_rms", "tracking_error_max")
    result = {"scenario": str(args.scenario), "overrides": args.overrides, **{k: summ.get(k) for k in keys}}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out}/")
    return result


# ======================================================================================= editor
def cmd_edit(args):
    import matplotlib

    matplotlib.use("TkAgg")
    import shlex

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

    from handoff.env import make_env

    sc = load_scenario(args.scenario)
    out_path = Path(args.out or args.scenario or SCENARIO_DIR / "scenario.json")
    print("building the map (headless MetaDrive) ...")
    cfg = scenario_config(sc)
    env = make_env(cfg)
    env.reset(options={"episode_seed": int(sc["map_seed"])})
    road, ego0 = env.road, env.ego
    env.close()
    m = cfg["method"]
    ds_lo, ds_hi = m["ds_range"]
    a_lat_max = float(m["mpc"]["max_lat_accel"])

    wps = [tuple(w) for w in sc["waypoints"]]
    cars = [tuple(c) for c in sc["cars"]]
    history = []
    ui = {"mode": "waypoint", "proc": None}
    plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "c"]
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.05, right=0.99)
    view = {"xlim": (max(ego0.s - 10, 0), ego0.s + 250), "ylim": (road.d_min - 1.5, road.d_max + 1.5)}

    def redraw():
        if ax.lines or ax.patches:
            view["xlim"], view["ylim"] = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax.axhspan(road.d_min, road.d_max, color="0.93", zorder=0)
        ax.axhline(road.d_max, color="goldenrod", lw=2.5, zorder=1)
        ax.axhline(road.d_min, color="0.3", lw=2.5, zorder=1)
        for c in road.lane_centers[:-1]:
            ax.axhline(c - road.lane_width / 2, color="0.55", lw=1, ls=(0, (8, 8)), zorder=1)
        ax.axvline(road.length, color="k", lw=1)
        ax.text(road.length, road.d_max + 0.4, " end of route", fontsize=8)
        ax.add_patch(Rectangle((ego0.s - CAR_LENGTH / 2, ego0.d - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
                               color="tab:green", zorder=3))
        ax.text(ego0.s, ego0.d + 1.3, "ego", ha="center", fontsize=8, color="tab:green")
        for car in cars:
            ax.add_patch(Rectangle((car[0] - CAR_LENGTH / 2, car[1] - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
                                   color="tab:blue", zorder=3))
        pts = sorted(wps)
        if pts:
            v = float(sc["target_speed_kmh"]) / 3.6
            chain = [(ego0.s, ego0.d)] + pts
            for (s0, d0), (s1, d1) in zip(chain[:-1], chain[1:]):
                ds = s1 - s0
                ay = 5.77 * abs(d1 - d0) * v * v / max(ds, 1e-6) ** 2    # peak lateral accel of the quintic
                col = "tab:red" if ay > a_lat_max else ("darkorange" if not ds_lo <= ds <= ds_hi else "0.35")
                ax.plot([s0, s1], [d0, d1], color=col, lw=1.6, zorder=2)
                if ay > a_lat_max:
                    ax.text(0.5 * (s0 + s1), 0.5 * (d0 + d1) + 0.5, f"a_y≈{ay:.1f}", color="tab:red", fontsize=8,
                            ha="center")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color="crimson", ms=7, zorder=4)
            for i, (s, d) in enumerate(pts):
                ax.text(s, d - 0.9, str(i + 1), ha="center", fontsize=8, color="crimson")
        ax.set_xlim(view["xlim"])
        ax.set_ylim(view["ylim"])
        ax.set_xlabel("s: distance along route (m)")
        ax.set_ylabel("d: lateral (m, left +)")
        ax.set_title(f"{out_path}  |  {len(pts)} waypoints, {len(cars)} cars  |  ego lane {sc['ego_lane']} at "
                     f"{sc['ego_speed_kmh']:g} km/h, target {sc['target_speed_kmh']:g} km/h\n"
                     f"mode: {ui['mode'].upper()}  |  left click: place {ui['mode']}   shift+click / 'c': car   "
                     "right click: delete nearest   (turn off pan/zoom before placing)", fontsize=9)
        fig.canvas.draw_idle()

    def busy():
        tb = getattr(fig.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "") != ""

    def add(kind, item):
        (wps if kind == "wp" else cars).append(item)
        history.append((kind, item))
        redraw()

    def add_car(s, d):
        add("car", (float(s), road.lane_centers[road.nearest_lane(d)]))

    def on_click(ev):
        if ev.inaxes is not ax or busy() or ev.xdata is None:
            return
        s, d = float(ev.xdata), float(ev.ydata)
        if ev.button == 1 and (ev.key == "shift" or ui["mode"] == "car"):
            add_car(s, d)
        elif ev.button == 1:
            add("wp", (s, float(np.clip(d, road.d_min, road.d_max))))
        elif ev.button == 3:
            items = [("wp", p) for p in wps] + [("car", c) for c in cars]
            if items:
                dist = [abs(p[0] - s) / 5.0 + abs(p[1] - d) / 1.5 for _, p in items]
                i = int(np.argmin(dist))
                if dist[i] <= 2.0:
                    kind, p = items[i]
                    (wps if kind == "wp" else cars).remove(p)
                    history.append(("del_" + kind, p))
                    redraw()

    def on_key(ev):
        if ev.key == "c" and ev.inaxes is ax and ev.xdata is not None:
            add_car(ev.xdata, ev.ydata)

    def undo(_ev=None):
        if history:
            kind, p = history.pop()
            target = wps if kind.endswith("wp") else cars
            (target.append if kind.startswith("del_") else target.remove)(p)
            redraw()

    def clear(_ev=None):
        wps.clear()
        cars.clear()
        history.clear()
        redraw()

    def save(_ev=None):
        out = {k: v for k, v in sc.items() if k not in ("waypoints", "cars")}
        out["waypoints"] = [[round(s, 2), round(d, 3)] for s, d in sorted(wps)]
        out["cars"] = [[round(c[0], 2), round(c[1], 3), *c[2:]] for c in sorted(cars)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"saved {out_path}  ({len(wps)} waypoints, {len(cars)} cars)")
        ax.set_title(ax.get_title().split("\n")[0] + f"\nsaved {time.strftime('%H:%M:%S')}", fontsize=9)
        fig.canvas.draw_idle()

    def set_mode(label):
        ui["mode"] = label
        redraw()

    def run(_ev=None):
        if ui["proc"] is not None and ui["proc"].poll() is None:
            print("a run is still open; close its MetaDrive window (or press X in it) first")
            return
        save()
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", str(out_path)]
        constraints_on, demo = opt_box.get_status()
        cmd += ([] if constraints_on else ["--no-constraints"]) + (["--demo"] if demo else [])
        for ov in shlex.split(set_box.text.strip()):
            cmd += ["--set", ov]
        print("running:", " ".join(shlex.quote(c) for c in cmd))
        ui["proc"] = subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    def set_target(text):
        try:
            sc["target_speed_kmh"] = float(text)
        except ValueError:
            print(f"target speed must be a number (km/h), got {text!r}")
            speed_box.set_val(f"{sc['target_speed_kmh']:g}")
            return
        redraw()

    widgets = []
    mode_ax = fig.add_axes([0.05, 0.03, 0.09, 0.12])
    mode_ax.set_title("click places", fontsize=8)
    widgets.append(RadioButtons(mode_ax, ("waypoint", "car")))
    widgets[-1].on_clicked(set_mode)
    for i, (label, fn) in enumerate((("Save", save), ("Undo", undo), ("Clear", clear))):
        widgets.append(Button(fig.add_axes([0.16 + 0.075 * i, 0.06, 0.065, 0.06]), label))
        widgets[-1].on_clicked(fn)
    opt_box = CheckButtons(fig.add_axes([0.39, 0.03, 0.16, 0.11]), ["corridor constraints", "demo mode"],
                           [True, False])
    set_box = TextBox(fig.add_axes([0.58, 0.06, 0.27, 0.06]), "--set ", initial="")
    speed_box = TextBox(fig.add_axes([0.31, 0.155, 0.06, 0.05]), "target speed (km/h) ",
                        initial=f"{sc['target_speed_kmh']:g}")
    speed_box.on_submit(set_target)
    run_btn = Button(fig.add_axes([0.88, 0.05, 0.09, 0.08]), "Run", color="palegreen", hovercolor="limegreen")
    run_btn.on_clicked(run)
    widgets += [opt_box, set_box, speed_box, run_btn]
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


# ======================================================================================= checks
PID_TOL_LANE_CHANGE = 0.30   # m, max cross-track error during a lane change
PID_TOL_STEADY = 0.10        # m, cross-track error once settled
PID_TOL_SPEED = 0.5          # m/s, speed error once settled


def check_env() -> dict:
    from handoff.env import make_env

    cfg = load_config("m1")
    env = make_env(cfg, is_eval=True)
    res, ok = {"spawn": []}, True
    want = {"max_engine_force": 800, "max_brake_force": 150, "wheel_friction": 0.9, "max_steering": 40}
    v0 = kmh(cfg["sim"]["ego"]["spawn_speed_km_h"])
    for seed in cfg["scenario"]["seeds"]["eval"][:6]:
        env.reset(options={"episode_seed": seed})
        dyn = env.md.agent.get_dynamics_parameters()
        e, road = env.ego, env.road
        err = [abs(e.d - road.lane_centers[road.nearest_lane(e.d)]), abs(e.v - v0)]
        good = all(abs(float(dyn[k]) - v) < 1e-6 for k, v in want.items()) and max(err) < 0.05
        res["spawn"].append({"seed": seed, "err_d_v": [round(x, 4) for x in err],
                             "dynamics": {k: float(v) for k, v in dyn.items()}, "ok": good})
        ok &= good

    road, net = env.road, env.md.current_map.road_network   # RoadFrame vs MetaDrive lane geometry
    errs = []
    for lanes in (net.graph[">>>"]["1S0_0_"], net.graph["3S0_0_"]["4S0_0_"]):
        for i, ln in enumerate(lanes):
            for lon in np.linspace(1, ln.length - 1, 9):
                x, y = ln.position(lon, 0.0)
                s, d = road.to_frenet(x, y)
                x2, y2 = road.to_world(s, d)
                errs.append(max(abs(x2 - x), abs(y2 - y), abs(d - road.lane_centers[i])))
    res["roadframe"] = {"max_err": float(max(errs)), "lane_centers": road.lane_centers, "ok": max(errs) < 1e-6}
    ok &= res["roadframe"]["ok"]

    env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][0]})   # steer right until off-road
    reason = None
    for _ in range(300):
        _, _, te, tr, info = env.step(np.array([-0.6, 0.3]))
        if te or tr:
            reason = info["termination"]
            break
    res["forced_off_road_termination"] = reason
    ok &= reason == "off_road"
    env.close()

    cfg = load_config("m3")              # determinism: same seed + actions -> identical trajectory
    env = make_env(cfg, is_eval=True)
    actions = np.random.default_rng(1).uniform(-1, 1, (25, env.action_space.shape[0]))
    trajs = []
    for _ in range(2):
        env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][3]})
        xs = []
        for a in actions:
            _, _, te, tr, _ = env.step(a)
            xs.append((env.ego.x, env.ego.y, env.ego.v))
            if te or tr:
                break
        trajs.append(np.array(xs))
    env.close()
    same = len(trajs[0]) == len(trajs[1])
    dev = float(np.max(np.abs(trajs[0] - trajs[1]))) if same else float("inf")
    res["determinism"] = {"max_abs_diff": dev, "steps": len(trajs[0]), "ok": bool(same and dev == 0.0)}
    res["ok"] = bool(ok and res["determinism"]["ok"])
    return res


def check_m2() -> dict:
    env = controlled_env("m2", ego_lane=1, ego_kmh=54.0)
    env.reset(options={"episode_seed": 0})
    L = env.road.lane_centers
    plan = [(L[1], kmh(60), 6), (L[0], kmh(60), 10), (L[2], kmh(70), 14), (L[1], kmh(50), 10)]
    rows = []
    for d_star, v_star, n_dec in plan:
        for j in range(n_dec):
            _, _, te, tr, _ = env.step(env.method.encode(d_star, v_star))
            cte = env.recorder.steps["tracking_error"][-env.method.period:]
            rows.append((j, d_star, v_star, env.ego.d, env.ego.v, cte))
            if te or tr:
                break
    gains = {k: env.cfg["method"]["pid"][k] for k in ("heading", "lateral", "speed")}
    env.close()
    max_cte = max(float(np.nanmax(r[5])) for r in rows)
    steady_cte = max(abs(r[3] - r[1]) for r in rows if r[0] >= (6 if r[1] != L[1] else 3))
    speed_err = [abs(r[4] - r[2]) for r in rows if r[0] >= 6]   # last decisions of each segment
    res = {"gains": gains, "max_cross_track_m": max_cte, "steady_cross_track_m": steady_cte,
           "max_settled_speed_err_ms": max(speed_err) if speed_err else None,
           "tolerances": {"lane_change": PID_TOL_LANE_CHANGE, "steady": PID_TOL_STEADY, "speed": PID_TOL_SPEED},
           "terminated_early": len(rows) < sum(p[2] for p in plan)}
    res["ok"] = bool(max_cte < PID_TOL_LANE_CHANGE and steady_cte < PID_TOL_STEADY and speed_err
                     and max(speed_err) < PID_TOL_SPEED and not res["terminated_early"])
    return res


def check_m3() -> dict:
    out = {}
    # (a) pass a 35 km/h lead: lane 1 -> lane 0 while passing -> back to lane 1
    env = controlled_env("m3", ego_lane=1, ego_kmh=60.0)
    env.reset(options={"episode_seed": 0})
    L = env.road.lane_centers
    lead = env.add_scripted_car(130.0, L[1], 35.0)
    lead_s = lambda: env.road.to_frenet(*lead.position)[0]  # noqa: E731
    passed, reason = False, None
    for _ in range(80):
        d_tgt = L[0] if lead_s() - env.ego.s > -15 else L[1]
        _, _, te, tr, info = env.step(env.method.encode([40.0, 40.0], [d_tgt, d_tgt], kmh(70)))
        passed |= env.ego.s > lead_s() + 20
        if te or tr:
            reason = info["termination"]
            break
    st = env.recorder.steps
    solve = np.array(st["solve_time"])
    solve = solve[solve > 0]
    out["pass_lead"] = {"passed": passed, "termination": reason, "min_gap": float(np.min(st["min_gap"])),
                        "final_d": env.ego.d, "solve_ms_mean": float(solve.mean() * 1e3) if len(solve) else None,
                        "solve_ms_p95": float(np.percentile(solve, 95) * 1e3) if len(solve) else None,
                        "fallbacks": int(np.sum(st["fallback"])), "gated_brake_steps": int(np.sum(st["gated_brake"]))}
    out["pass_lead"]["ok"] = bool(passed and reason in (None, "success") and out["pass_lead"]["min_gap"] > 0.3)
    env.close()

    # (b) stopped car in the ego's lane, waypoints straight through it: the gate / MPC must stop the ego
    env = controlled_env("m3", ego_lane=1, ego_kmh=50.0, overrides=["method.ebrake.ends_episode=false"])
    env.reset(options={"episode_seed": 0})
    L = env.road.lane_centers
    env.add_scripted_car(170.0, L[1])
    reason = None
    for _ in range(40):
        _, _, te, tr, info = env.step(env.method.encode([25.0, 25.0], [L[1], L[1]], kmh(50)))
        if te or tr:
            reason = info["termination"]
            break
    st = env.recorder.steps
    out["stopped_car"] = {"termination": reason, "min_gap": float(np.min(st["min_gap"])), "final_s": env.ego.s,
                          "final_v": env.ego.v, "gated_brake_steps": int(np.sum(st["gated_brake"]))}
    out["stopped_car"]["ok"] = bool(reason != "collision" and out["stopped_car"]["min_gap"] > 0.2 and env.ego.v < 0.5)
    env.close()

    # (c) fallback chain: solver raises -> replayed plan (if replay_steps.timeout > 0) -> emergency brake; recovery
    env = controlled_env("m3", ego_lane=1, ego_kmh=60.0)
    env.reset(options={"episode_seed": 0})
    L = env.road.lane_centers
    act = env.method.encode([30.0], [L[1]], kmh(60))
    env.step(act)
    healthy = int(np.sum(env.recorder.steps["fallback"]))
    mpc = env.method.mpc
    real = mpc.solver

    def broken(**_kw):
        raise RuntimeError("forced solver failure (validation)")

    mpc.solver = broken
    for _ in range(3):
        _, _, te, tr, _ = env.step(act)
        if te or tr:
            break
    modes = {m.split(":")[-1] for m in env.recorder.status[-30:] if m.startswith("fallback")}
    mpc.solver = real
    env.step(act)
    recovered = not env.recorder.steps["fallback"][-1]
    expected = {"emergency_brake"} | ({"shifted_previous"} if mpc.replay["timeout"] > 0 else set())
    out["fallback"] = {"healthy_fallbacks": healthy, "modes_seen": sorted(modes), "modes_expected": sorted(expected),
                       "recovered": bool(recovered), "speed_after": env.ego.v}
    out["fallback"]["ok"] = bool(healthy == 0 and expected <= modes and recovered)
    env.close()
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def check_smoke(n_episodes: int = 2) -> dict:
    from handoff.env import make_env
    from handoff.logs import load_episode
    from handoff.reward import TERMS

    out = {}
    for method in ("m1", "m2", "m3"):
        cfg = load_config(method)
        run_dir = OUT / "checks" / "smoke" / method
        shutil.rmtree(run_dir, ignore_errors=True)
        env = make_env(cfg, run_dir=run_dir, is_eval=True)
        rng = np.random.default_rng(0)
        sums, t0 = [], time.time()
        for ep in range(n_episodes):
            env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][ep]})
            while True:
                _, r, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
                assert np.isfinite(r)
                if te or tr:
                    sums.append(info["episode_summary"])
                    break
        env.close()
        problems = []
        for s in sums:
            for k in ("return", "mean_speed", "sim_time"):
                if not np.isfinite(float(s[k])):
                    problems.append(f"{k} not finite")
            d = load_episode(s["log_file"])["data"]
            for n in TERMS:
                if len(d[f"term_{n}"]) != int(s["control_steps"]):
                    problems.append(f"term {n} length")
            if len(d["dec_k"]) != int(s["decisions"]):
                problems.append("decision count mismatch")
            if abs(float(np.sum(d["step_reward"])) - float(s["return"])) > 1e-6:
                problems.append("return != sum of step rewards")
        out[method] = {"episodes": [(s["termination"], int(s["control_steps"]), round(float(s["return"]), 2))
                                    for s in sums], "wall_s": round(time.time() - t0, 1), "problems": problems,
                       "ok": not problems}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def check_b0() -> dict:
    from handoff.evaluate import ParallelEvaluator, aggregate

    cfg = load_config("b0")
    seeds = cfg["scenario"]["seeds"]["eval"]
    ev = ParallelEvaluator(cfg, None, n_workers=4)
    try:
        summaries = ev.evaluate(None, seeds, tag="validate")
    finally:
        ev.close()
    agg = aggregate(summaries)
    res = {k: agg.get(k) for k in ("n_episodes", "success", "collision", "off_road", "timeout", "mean_speed",
                                   "time_to_complete", "min_gap")}
    res["ok"] = agg.get("n_episodes") == len(seeds)
    return res


def check_train() -> dict:
    from handoff.train import train

    out = {}
    for method in ("m1", "m2", "m3"):
        cfg = load_config(method, overrides=[
            "training.total_control_steps=300", "training.n_envs=2", "eval.n_episodes=2",
            "eval.freq_control_steps=200", "training.checkpoint_freq_control_steps=200", "ppo.n_steps=32",
            "ppo.batch_size=32", "ppo.n_epochs=1"])
        run_dir = OUT / "checks" / "train" / method
        shutil.rmtree(run_dir, ignore_errors=True)
        t0 = time.time()
        try:
            train(cfg, run_dir=run_dir, verbose=0)
            import pandas as pd

            files = {f: (run_dir / f).exists() for f in ("train_progress.csv", "episodes.csv", "config.yaml",
                                                         "checkpoints/final.zip", "checkpoints/best.zip")}
            ev = pd.read_csv(run_dir / "episodes.csv")
            out[method] = {"files": files, "eval_rows": int((ev["is_eval"] == 1).sum()),
                           "train_rows": int((ev["is_eval"] == 0).sum()), "wall_s": round(time.time() - t0, 1)}
            out[method]["ok"] = all(files.values()) and out[method]["eval_rows"] >= 4
        except Exception:  # noqa: BLE001
            out[method] = {"error": traceback.format_exc()[-2000:], "ok": False}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def check_scenarios() -> dict:
    out = {}
    for sc in sorted(SCENARIO_DIR.glob("*.json")):
        run_out = OUT / "checks" / "scenarios" / sc.stem
        shutil.rmtree(run_out, ignore_errors=True)
        p = subprocess.run([sys.executable, __file__, "run", str(sc), "--no-render", "--out", str(run_out)],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        res_file = run_out / "result.json"
        if p.returncode != 0 or not res_file.exists():
            out[sc.stem] = {"ok": False, "error": p.stderr[-1500:]}
            continue
        res = json.loads(res_file.read_text())
        res["ok"] = res["termination"] not in ("collision", "off_road")
        out[sc.stem] = res
    out["ok"] = all(v["ok"] for v in out.values())
    return out


CHECKS = {"env": check_env, "m2": check_m2, "m3": check_m3, "smoke": check_smoke, "b0": check_b0,
          "train": check_train, "scenarios": check_scenarios}


def run_check(name: str) -> dict:
    t0 = time.time()
    try:
        res = CHECKS[name]()
    except Exception:  # noqa: BLE001
        res = {"ok": False, "error": traceback.format_exc()}
    res["wall_s"] = round(time.time() - t0, 1)
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    (OUT / "checks" / f"{name}.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def cmd_check(args):
    """Each check in its own process (one MetaDrive per process)."""
    names = args.names or list(CHECKS)
    summary = {}
    for name in names:
        subprocess.run([sys.executable, __file__, "_one", name], cwd=REPO_ROOT)
        rep = OUT / "checks" / f"{name}.json"
        res = json.loads(rep.read_text()) if rep.exists() else {"ok": False}
        summary[name] = bool(res.get("ok"))
        print(f"{'PASS' if summary[name] else 'FAIL'}  {name}  ({res.get('wall_s', '?')} s)", flush=True)
        if not summary[name] and "error" in res:
            print(res["error"][-1500:])
    (OUT / "checks" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"details: {OUT / 'checks'}/<name>.json")
    sys.exit(0 if all(summary.values()) else 1)


# ======================================================================================= cli
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a scenario through the M3 env")
    r.add_argument("scenario")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--fps", type=float, default=10.0, help="10 = real time, 0 = as fast as possible")
    r.add_argument("--no-render", action="store_true", help="headless (logs + plots only)")
    r.add_argument("--no-constraints", action="store_true", help="start with the corridor constraints off (K)")
    r.add_argument("--demo", action="store_true", help="only the road and the cars; M toggles the overlays")
    r.add_argument("--out", default=None, help="default runs/validation/<scenario>/<timestamp>")
    e = sub.add_parser("edit", help="place waypoints and cars")
    e.add_argument("--scenario", default=None, help="existing scenario JSON to edit")
    e.add_argument("--out", default=None,
                   help=f"where Save writes (default: --scenario or {SCENARIO_DIR}/scenario.json)")
    c = sub.add_parser("check", help="pass / fail checks")
    c.add_argument("names", nargs="*", metavar="NAME", help=f"any of {list(CHECKS)} (default: all)")
    one = sub.add_parser("_one")
    one.add_argument("name", choices=list(CHECKS))
    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "edit":
        cmd_edit(args)
    elif args.cmd == "check":
        unknown = set(args.names) - set(CHECKS)
        if unknown:
            p.error(f"unknown checks {sorted(unknown)}; choose from {list(CHECKS)}")
        cmd_check(args)
    else:
        sys.exit(0 if run_check(args.name)["ok"] else 1)


if __name__ == "__main__":
    main()
