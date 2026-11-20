#!/usr/bin/env python
"""M3 test harness: the M3 pipeline minus the safe-set projection (spline -> corridor -> MPC) driven by
hand-placed waypoints instead of the RL policy, in MetaDrive in real time with static cars.

1. Place waypoints and static cars in a small editor (top-down road in (s, d), lane lines drawn):
       python scripts/m3_harness.py edit --out configs/m3_harness/my_test.json
       python scripts/m3_harness.py edit --scenario configs/m3_harness/overtake.json   # edit an existing one
   Pick "waypoint" or "car" mode on the left, then left click to place (cars snap to the nearest lane
   centre; Shift+click or 'c' places a car in either mode). Right click deletes the nearest item.
   "target speed (km/h)" sets the desired trajectory speed (spline reference + MPC target; press Enter).
   Buttons: Save / Undo / Clear, and Run, which saves and starts the MetaDrive run in a new process with
   the "corridor constraints" / "road-aligned corridor" checkboxes and the --set box (space-separated
   KEY=VALUE overrides) applied.
   Pan/zoom with the toolbar (click the tool again to turn it off before placing). Segments the RL could
   not produce (spacing outside 5-40 m) are orange; segments whose quintic needs more lateral accel than
   the safe set allows are red.

2. Run it from the editor, or directly (MetaDrive 3D window, 10 Hz real time):
       python scripts/m3_harness.py run configs/m3_harness/overtake.json
       python scripts/m3_harness.py run <scenario> --no-corridor-constraints        # MPC without the tube
       python scripts/m3_harness.py run <scenario> --corridor stock                 # M3's own corridor builder
       python scripts/m3_harness.py run <scenario> --set method.mpc.corridor.keepout_s=1.5
   The default corridor here is the harness's road-aligned one (RoadCorridor below): fixed band between the
   exterior lane centres, cut by the cars' keep-out zones; --corridor stock uses M3's CorridorBuilder.
       python scripts/m3_harness.py run <scenario> --set method.mpc.weights.lateral=4.0
       python scripts/m3_harness.py run <scenario> --no-render                      # headless, logs only
   Overlays:
       white balls   all waypoints placed in the scenario
       yellow line   reference spline;  cyan line  MPC predicted path
       blue lines    corridor over the MPC horizon: each stage's left/right bound on the ego centre at that
                     stage's planned position (road corridor; --corridor stock draws the current box only)
       magenta boxes car keep-out zones (car + keepout_s fore/aft, + margin_d sideways; road corridor only)
   Demo mode (editor checkbox, or --demo): only the cars and the road; M toggles the overlays + text live.
   K = toggle the corridor constraints live (the blue box is hidden while they are off; the text panel
   shows the state), X = finish and save; avoid R (MetaDrive reset) and Esc (exits, no save).
   Writes runs/m3_harness/<scenario>/<timestamp>/{steps.csv, decisions.csv, summary.json, overview.png}.

Waypoints reach the MPC the way the RL's do: every decision period (10 control steps) the next <= 5 placed
waypoints more than ds_min ahead of the ego are taken, re-spaced to the RL's per-slot range
(Δs ∈ ds_range, cumulative from the ego's s) and laterally clipped to the RL's d range, then splined and
handed to the MPC (no safe-set projection). Ego: vehicle_model static_default, no traffic, gains from
configs/method/m3.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

SCENARIO_DIR = Path("configs/m3_harness")
CAR_LENGTH, CAR_WIDTH = 4.515, 1.852          # TrafficDefaultVehicle (same body as the ego)


# =============================================================================== shared setup
def default_scenario() -> dict:
    return {"map_seed": 0, "ego_lane": 1, "ego_speed_kmh": 60.0, "target_speed_kmh": 70.0,
            "waypoints": [], "cars": []}


def load_scenario(path) -> dict:
    sc = default_scenario()
    sc.update(json.loads(Path(path).read_text()))
    return sc


def load_cfg(sc: dict, overrides: list[str]) -> dict:
    from handoff.config import load_config

    ov = ["scenario.traffic.density=0",
          "sim.ego.random_spawn_lane=false",
          f"sim.ego.spawn_speed_km_h={float(sc['ego_speed_kmh'])}",
          f"sim.target_speed_km_h={float(sc['target_speed_kmh'])}",
          "sim.on_continuous_line_done=false",     # keep running if the car touches a solid line
          "sim.horizon=100000"]
    return load_config("m3", overrides=ov + list(overrides))


def make_env(sc: dict, cfg: dict, render: bool, fps: float | None = None, demo: bool = False):
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.envs.top_down_env import TopDownMetaDrive

    from handoff.simulator.metadrive_env import make_metadrive_config

    md = make_metadrive_config(cfg, render=render)
    md["vehicle_config"]["vehicle_model"] = "static_default"   # fixed dynamics (engine 800, brake 150)
    md["num_scenarios"] = 1
    md["start_seed"] = int(sc["map_seed"])
    md["agent_configs"] = {"default_agent": {
        "use_special_color": True,
        "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, int(sc["ego_lane"]))}}
    if fps:
        md["force_render_fps"] = float(fps)
    if demo:   # just the cars and the road: no MetaDrive navigation marks, arrow, dashboard or FPS counter
        md["vehicle_config"]["show_navi_mark"] = False
        md["vehicle_config"]["show_navigation_arrow"] = False
        md["show_interface"] = False
        md["show_fps"] = False
    env = TopDownMetaDrive(md)
    env.reset(seed=int(sc["map_seed"]))
    return env


def ego_state(env, road, t: float):
    from handoff.simulator.state import make_ego_state

    a = env.agent
    x, y = a.position
    return make_ego_state(t, x, y, a.heading_theta, a.speed, float(a.steering), float(a.throttle_brake),
                          a.LENGTH, a.WIDTH, road)


def rl_waypoints(ego_s: float, placed: np.ndarray, road, m: dict) -> np.ndarray:
    """What the RL could output now: the next <= n_slots placed waypoints ahead, re-spaced like decode()."""
    lo, hi = m["ds_range"]
    margin = m["lateral_margin"]
    ahead = placed[placed[:, 0] > ego_s + lo] if len(placed) else placed
    if len(ahead) == 0:   # nothing left: keep going straight at the last placed (or current lane) offset
        d_last = placed[-1, 1] if len(placed) else road.lane_centers[0]
        ahead = np.array([[ego_s + hi, d_last]])
    ahead = ahead[: int(m["n_slots"])]
    out, prev = [], ego_s
    for s, d in ahead:
        s_i = prev + float(np.clip(s - prev, lo, hi))
        out.append((s_i, float(np.clip(d, road.d_min - margin, road.d_max + margin))))
        prev = s_i
    return np.array(out)


# =============================================================================== editor
def cmd_edit(args):
    import matplotlib

    matplotlib.use("TkAgg")
    import shlex
    import subprocess
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

    from handoff.simulator.metadrive_env import build_road_frame

    sc = load_scenario(args.scenario) if args.scenario else default_scenario()
    for key in ("map_seed", "ego_lane", "ego_speed_kmh", "target_speed_kmh"):
        if getattr(args, key) is not None:
            sc[key] = getattr(args, key)
    out_path = Path(args.out or args.scenario or SCENARIO_DIR / "scenario.json")

    print("building the map (headless MetaDrive) ...")
    cfg = load_cfg(sc, [])
    env = make_env(sc, cfg, render=False)
    road = build_road_frame(env)
    ego0 = ego_state(env, road, 0.0)
    env.close()
    m = cfg["method"]
    ds_lo, ds_hi = m["ds_range"]
    a_lat_max = float(m["safe_set"]["max_lat_accel"])

    wps = [tuple(w) for w in sc["waypoints"]]
    cars = [tuple(c) for c in sc["cars"]]
    history = []
    ui = {"mode": "waypoint", "proc": None}

    plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "c"]
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.05, right=0.99)
    view = {"xlim": (max(ego0.s - 10, 0), ego0.s + 250), "ylim": (road.d_min - 1.5, road.d_max + 1.5)}

    def lane_snap(d):
        return road.lane_centers[road.nearest_lane(d)]

    def redraw():
        if ax.lines or ax.patches:
            view["xlim"], view["ylim"] = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        w = road.lane_width if hasattr(road, "lane_width") else 3.5
        ax.axhspan(road.d_min, road.d_max, color="0.93", zorder=0)
        ax.axhline(road.d_max, color="goldenrod", lw=2.5, zorder=1)        # centre line (solid)
        ax.axhline(road.d_min, color="0.3", lw=2.5, zorder=1)              # right road edge
        for c in road.lane_centers[:-1]:
            ax.axhline(c - w / 2, color="0.55", lw=1, ls=(0, (8, 8)), zorder=1)
        ax.axvline(road.length, color="k", lw=1)
        ax.text(road.length, road.d_max + 0.4, " end of route", fontsize=8)
        ax.add_patch(Rectangle((ego0.s - CAR_LENGTH / 2, ego0.d - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
                               color="tab:green", zorder=3))
        ax.text(ego0.s, ego0.d + 1.3, "ego", ha="center", fontsize=8, color="tab:green")
        for s, d in cars:
            ax.add_patch(Rectangle((s - CAR_LENGTH / 2, d - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
                                   color="tab:blue", zorder=3))
        pts = sorted(wps)
        if pts:
            v = float(sc["target_speed_kmh"]) / 3.6
            chain = [(ego0.s, ego0.d)] + pts
            for (s0, d0), (s1, d1) in zip(chain[:-1], chain[1:]):
                ds = s1 - s0
                ay = 5.77 * abs(d1 - d0) * v * v / max(ds, 1e-6) ** 2
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

    def add_car(s, d):
        car = (float(s), float(lane_snap(d)))
        cars.append(car)
        history.append(("car", car))
        redraw()

    def on_click(ev):
        if ev.inaxes is not ax or busy() or ev.xdata is None:
            return
        s, d = float(ev.xdata), float(ev.ydata)
        if ev.button == 1 and (ev.key == "shift" or ui["mode"] == "car"):
            add_car(s, d)
        elif ev.button == 1:
            wp = (s, float(np.clip(d, road.d_min, road.d_max)))
            wps.append(wp)
            history.append(("wp", wp))
            redraw()
        elif ev.button == 3:
            items = [("wp", p) for p in wps] + [("car", c) for c in cars]
            if not items:
                return
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
        if not history:
            return
        kind, p = history.pop()
        if kind == "wp":
            wps.remove(p)
        elif kind == "car":
            cars.remove(p)
        elif kind == "del_wp":
            wps.append(p)
        elif kind == "del_car":
            cars.append(p)
        redraw()

    def clear(_ev=None):
        wps.clear()
        cars.clear()
        history.clear()
        redraw()

    def save(_ev=None):
        out = dict(sc)
        out["waypoints"] = [[round(s, 2), round(d, 3)] for s, d in sorted(wps)]
        out["cars"] = [[round(s, 2), round(d, 3)] for s, d in sorted(cars)]
        out["road_length"] = round(float(road.length), 1)
        out["ego_start"] = [round(float(ego0.s), 2), round(float(ego0.d), 3)]
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
        corridor_on, road_corridor, demo = opt_box.get_status()
        if demo:
            cmd.append("--demo")
        if not corridor_on:
            cmd.append("--no-corridor-constraints")
        cmd += ["--corridor", "road" if road_corridor else "stock"]
        for ov in shlex.split(set_box.text.strip()):
            cmd += ["--set", ov]
        print("running:", " ".join(shlex.quote(c) for c in cmd))
        ui["proc"] = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent))

    widgets = []
    mode_ax = fig.add_axes([0.05, 0.03, 0.09, 0.12])
    mode_ax.set_title("click places", fontsize=8)
    mode_radio = RadioButtons(mode_ax, ("waypoint", "car"))
    mode_radio.on_clicked(set_mode)
    widgets.append(mode_radio)
    for i, (label, fn) in enumerate((("Save", save), ("Undo", undo), ("Clear", clear))):
        b = Button(fig.add_axes([0.16 + 0.075 * i, 0.06, 0.065, 0.06]), label)
        b.on_clicked(fn)
        widgets.append(b)
    opt_box = CheckButtons(fig.add_axes([0.39, 0.03, 0.16, 0.11]),
                           ["corridor constraints", "road-aligned corridor", "demo mode"], [True, True, False])
    widgets.append(opt_box)
    set_box = TextBox(fig.add_axes([0.58, 0.06, 0.27, 0.06]), "--set ", initial="")
    widgets.append(set_box)

    def set_target(text):
        try:
            v = float(text)
        except ValueError:
            print(f"target speed must be a number (km/h), got {text!r}")
            speed_box.set_val(f"{sc['target_speed_kmh']:g}")
            return
        if not 0.0 < v <= 80.0:
            print(f"target speed {v:g} km/h outside (0, 80]; MetaDrive cuts the engine above 80 km/h")
        sc["target_speed_kmh"] = v
        redraw()

    speed_box = TextBox(fig.add_axes([0.31, 0.155, 0.06, 0.05]), "target speed (km/h) ",
                        initial=f"{sc['target_speed_kmh']:g}")
    speed_box.on_submit(set_target)
    widgets.append(speed_box)
    run_btn = Button(fig.add_axes([0.88, 0.05, 0.09, 0.08]), "Run", color="palegreen", hovercolor="limegreen")
    run_btn.on_clicked(run)
    widgets.append(run_btn)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


# =============================================================================== runner
def spawn_static_cars(env, road, cars):
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

    spawned = []
    for s, d in cars:
        x, y = road.to_world(float(s), float(d))
        v = env.engine.spawn_object(TrafficDefaultVehicle, vehicle_config={"spawn_velocity": None},
                                    position=[x, y], heading=float(road.heading_at(float(s))))
        v.set_velocity([0.0, 0.0])
        v.set_static(True)
        spawned.append(v)
    return spawned


def tracked_objects(cars_v, ego, road, R):
    from handoff.simulator.state import make_tracked_object

    near, everyone = [], []
    for i, v in enumerate(cars_v):
        x, y = v.position
        o = make_tracked_object(f"static{i}", x, y, v.heading_theta, 0.0, 0.0, v.LENGTH, v.WIDTH, "vehicle", road)
        everyone.append(o)
        dist = float(np.hypot(x - ego.x, y - ego.y))
        if dist <= R:          # same perception radius as CoreEnv
            near.append((dist, o))
    near.sort(key=lambda p: p[0])
    return [o for _, o in near], everyone


def cmd_run(args):

    from handoff.rl.methods.common import ActuationMap
    from handoff.rl.methods.m3_mpc import MPCController
    from handoff.rl.methods.m3_spline import build_m3_reference
    from handoff.simulator.metadrive_env import build_road_frame
    from handoff.simulator.overlays import OverlayDrawer, frenet_box_polygon
    from handoff.simulator.state import frenet_box_gap

    class HarnessMPC(MPCController):
        """M3's MPC (barrier/yaw-rate costs, e-brake law and road corridor all live in MPCController now) with a
        switch for the corridor constraints. When off, every stage's 4 half-spaces are replaced by
        0·x + 0·y <= 1e6 (always true); the tube is still built because the cost tracks its initial trajectory.
        The NLP structure is unchanged, so it can be toggled between solves."""

        corridor_on = True

        def _params(self, ego):
            p, ref, idx = super()._params(ego)
            if not self.corridor_on:
                start = self.NX + 2 + 4 * (self.N + 1)
                tube = p[start:].reshape(self.N + 1, 4, 3)    # per stage: 4 x (a_x, a_y, b)
                tube[:, :, :2] = 0.0
                tube[:, :, 2] = 1e6
                p = p.copy()
                p[start:] = tube.ravel()
            return p, ref, idx

    from handoff.rl.methods.m3_corridor import RoadCorridor, path_blocked

    sc = load_scenario(args.scenario)
    if args.target_speed is not None:
        sc["target_speed_kmh"] = float(args.target_speed)
    placed = np.array(sorted(sc["waypoints"]), dtype=float).reshape(-1, 2)
    if len(placed) == 0:
        raise SystemExit("scenario has no waypoints; place some with the editor first")
    cfg = load_cfg(sc, args.overrides)
    m = cfg["method"]
    sim = cfg["sim"]
    dt = float(sim["control_dt"])
    period = int(m["decision_period"])
    v_target = float(sim["target_speed_km_h"]) / 3.6
    v_cap = float(sim["v_cap_km_h"]) / 3.6
    R = float(sim["perception_range"])
    render = not args.no_render

    env = make_env(sc, cfg, render=render, fps=None if args.no_render else args.fps, demo=args.demo)
    road = build_road_frame(env)
    cars_v = spawn_static_cars(env, road, sc["cars"])
    ego_cfg = sim["ego"]
    m["mpc"]["corridor"]["type"] = args.corridor
    if args.corridor_inset is not None:
        m["mpc"]["corridor"]["inset"] = args.corridor_inset
    actuation = ActuationMap(cfg["actuation"]["table"])
    mpc = HarnessMPC(m["mpc"], road, actuation, lf=ego_cfg["front_wheelbase"], lr=ego_cfg["rear_wheelbase"],
                        ego_length=ego_cfg["length"], ego_width=ego_cfg["width"],
                        max_steering_rad=float(np.deg2rad(ego_cfg["max_steering_deg"])), v_cap=v_cap,
                        v_target=v_target)
    mpc.set_road(road)
    mpc.reset()
    mpc.corridor_on = not args.no_corridor_constraints

    print(f"vehicle {type(env.agent).__name__}: "
          + ", ".join(f"{k} {float(v):g}" for k, v in env.agent.get_dynamics_parameters().items()))
    print(f"scenario {args.scenario}: {len(placed)} waypoints, {len(cars_v)} static cars, "
          f"target {v_target:.1f} m/s, "
          f"corridor constraints {'on' if mpc.corridor_on else 'off'} ({args.corridor} corridor)")
    if render:
        print("keys: K = toggle corridor constraints, X = finish and save "
              "(avoid R = MetaDrive reset, Esc = exit without saving)")

    state = {"finish": False, "demo": bool(args.demo)}

    def toggle_demo():
        state["demo"] = not state["demo"]
        print(f"demo mode {'ON' if state['demo'] else 'OFF'}")

    def toggle_corridor():
        mpc.corridor_on = not mpc.corridor_on
        mpc._prev_lam_g = mpc._prev_lam_x = None      # old multipliers belong to the other constraint set
        print(f"corridor constraints {'ON' if mpc.corridor_on else 'OFF'}")

    if render:
        env.engine.accept("x", lambda: state.__setitem__("finish", True))
        env.engine.accept("k", toggle_corridor)
        env.engine.accept("m", toggle_demo)
    drawer = None
    if render:
        drawer = OverlayDrawer(env.engine)
    placed_pts = [(float(x), float(y), 0.9) for x, y in zip(*road.to_world(placed[:, 0], placed[:, 1]))]

    def road_corridor_lines(ego, st):
        """The corridor the MPC is solving with: for each of its N+1 stages, the left/right bound on the ego
        centre drawn at that stage's planned position, joined into two ribbons along the path (they step in
        where a keep-out box narrows a stage, ahead of the car). Car-made front/back faces are drawn as
        cross lines. Blocked stages break the ribbons. (Polylines; the drawer closes them by retracing.)"""
        tube = mpc.tube
        big = 0.5 * RoadCorridor.BIG
        idx = np.minimum(np.arange(mpc.N + 1) + st, len(tube) - 1)
        lines, left, right, cuts = [], [], [], set()

        def flush():
            for side in (left, right):
                if len(side) >= 2:
                    lines.append(np.array(side))
                side.clear()

        for k in idx:
            if tube.blocked[k]:
                flush()
                continue
            sk, dk = tube.traj[k, 0], tube.traj[k, 1]
            um, up, wm, wp = tube.extents[k]
            left.append(road.to_world(float(sk), float(dk + wp)))
            right.append(road.to_world(float(sk), float(dk + wm)))
            for ext in (um, up):
                key = round(float(sk + ext), 1)
                if abs(ext) < big and key not in cuts:     # a car-made front/back face
                    cuts.add(key)
                    x, y = road.to_world(np.full(2, sk + ext), np.array([dk + wm, dk + wp]))
                    lines.append(np.stack([x, y], axis=1))
        flush()
        return lines

    def draw(ego, ref, dec, diag, text, st):
        if not render:
            return
        if state["demo"]:                     # demo mode: only the cars and the road
            drawer.lines.reset()
            drawer.points.reset()
            env.render()
            return
        frame = {}
        mk = (ref.s >= ego.s - 5.0) & (ref.s <= ego.s + 150.0)
        frame["ref_xy"] = np.stack([ref.x[mk], ref.y[mk]], axis=1)
        if diag.get("pred_xy") is not None:
            frame["pred_xy"] = diag["pred_xy"]
        if args.corridor == "road":     # keep-out zones (car + keepout_s fore/aft, + margin_d sideways)
            polys = []
            for o in objects:
                zl, zw = mpc.corridor.zone_half(o)
                polys.append(frenet_box_polygon(road, (o.s - zl, o.s + zl, o.d - zw, o.d + zw)))
            frame["boxes"] = polys
        line = check_line(ego)                # emergency-brake check line (orange)
        if line is not None:
            frame["obj_pred"] = [np.stack(road.to_world(line[0][[0, -1]], line[1][[0, -1]]), axis=1)]
        if mpc.tube is None:
            pass
        elif not args.hide_corridor and mpc.corridor_on and args.corridor == "road":
            frame["tube"] = road_corridor_lines(ego, st)
        elif diag.get("tube_corners") is not None and not args.hide_corridor and mpc.corridor_on:
            frame["tube"] = list(diag["tube_corners"][:1])   # current stage only (the box the MPC is in now)
        drawer.draw(frame)
        drawer.points.draw_points(placed_pts, [(1.0, 1.0, 1.0, 1.0)] * len(placed_pts))
        env.render(text=text)

    corr_cfg = m["mpc"]["corridor"]
    keep_s, keep_d = float(corr_cfg.get("keepout_s", 1.0)), float(corr_cfg["margin_d"])

    def check_line(ego):
        """Straight line (s, d) from the ego centre to the next placed waypoint ahead, sampled every 0.25 m."""
        ahead = placed[placed[:, 0] > ego.s]
        if len(ahead) == 0:
            return None
        s1, d1 = ahead[0]
        n = max(int(np.hypot(s1 - ego.s, d1 - ego.d) / 0.25), 2)
        return np.linspace(ego.s, s1, n), np.linspace(ego.d, d1, n)

    def reference_blocked(ego, ref, objects):
        """Shared M3 gate (m3_corridor.path_blocked) against the placed waypoints: the magenta keep-out zones."""
        return path_blocked(ego, placed, objects, keep_s, keep_d, road=road,
                            max_decel=float(m.get("ebrake_gate_decel", 6.0)), ego_length=float(ego_cfg["length"]))

    n_ebrake = 0
    stopped_steps = 0
    steps, decisions = [], []
    reason = "end_of_waypoints"
    s_stop = min(float(placed[-1, 0]) + 30.0, road.length - 5.0)
    dec, ref = None, None
    n_fb = 0
    k = 0
    while True:
        t = k * dt
        ego = ego_state(env, road, t)
        objects, all_objects = tracked_objects(cars_v, ego, road, R)
        if k % period == 0:
            # RL-equivalent waypoints go straight into the spline (no safe-set projection in this harness)
            raw = rl_waypoints(ego.s, placed, road, m)
            ref = build_m3_reference(ego, road, raw, v_target, ds=m["spline"]["ds"], extend=m["spline"]["extend"])
            dec = {"raw": raw}
            decisions.append({"t": t, "ego_s": ego.s, "k": len(raw), "waypoints": json.dumps(np.round(raw, 2).tolist())})
        # Emergency brake: the straight line from the ego to the next placed waypoint passes through a car's
        # keep-out zone. While it is active the MPC is not solved (no stale cyan plan), its warm start is
        # cleared, and the emergency-brake law (HarnessMPC.emergency_brake) drives.
        blocker = reference_blocked(ego, ref, objects)
        if blocker is not None:
            u, diag = mpc.gated_brake(ego, ref)
            n_ebrake += 1
        else:
            u, diag = mpc.step(ego, ref, objects)
        n_fb += bool(diag["fallback"])
        # current corridor stage as lateral bounds on the ego centre (absolute d), for checking the geometry
        if mpc.tube is not None:
            st = min(max(mpc._tube_age - 1, 0), len(mpc.tube) - 1)
            corr_d_lo = float(mpc.tube.traj[st, 1] + mpc.tube.extents[st, 2])
            corr_d_hi = float(mpc.tube.traj[st, 1] + mpc.tube.extents[st, 3])
            corr_blocked_now = int(bool(mpc.tube.blocked[st]))
        else:                                    # e-brake from the very first step: no corridor built yet
            st, corr_d_lo, corr_d_hi, corr_blocked_now = 0, float("nan"), float("nan"), 0
        gaps = [frenet_box_gap(ego, o) for o in all_objects]
        min_gap = min(gaps) if gaps else np.inf
        steps.append({
            "t": t, "x": ego.x, "y": ego.y, "s": ego.s, "d": ego.d, "psi": ego.psi, "v": ego.v,
            "u_s": float(u[0]), "u_tb": float(u[1]), "solve_ms": 1000 * diag["solve_time"],
            "status": diag["status"], "fallback": int(bool(diag["fallback"])),
            "fallback_reason": diag.get("fallback_reason", ""), "fallback_mode": diag.get("fallback_mode", ""),
            "tube_rebuilt": int(bool(diag.get("tube_rebuilt"))), "tube_retry": int(bool(diag.get("tube_retry"))),
            "tube_blocked": int(diag.get("tube_blocked", 0)), "tracking_error": diag["tracking_error"],
            "ref_d": diag["ref_d"], "ref_v": diag["ref_v"], "min_gap": min_gap, "n_objects": len(objects),
            "corridor_on": int(mpc.corridor_on), "corr_d_lo": corr_d_lo, "corr_d_hi": corr_d_hi,
            "ebrake_ref_blocked": int(blocker is not None),
            "corr_blocked_now": corr_blocked_now})
        text = {
            "t": f"{t:.1f} s",
            "speed": f"{ego.v:.1f} / target {v_target:.1f} m/s",
            "MPC": f"{diag['status'][:24]}  {1000 * diag['solve_time']:.0f} ms",
            "fallbacks": f"{n_fb}" + (f"  (now: {diag.get('fallback_reason')}/{diag.get('fallback_mode')})"
                                      if diag["fallback"] else ""),
            "tracking err": f"{diag['tracking_error']:.2f} m",
            "min gap": f"{min_gap:.2f} m" if np.isfinite(min_gap) else "-",
            "tube blocked stages": f"{diag.get('tube_blocked', 0)}",
            "corridor constraints": "ON" if mpc.corridor_on else "OFF",
            "EMERGENCY BRAKE": f"car on the reference ({blocker})" if blocker is not None else "-",
            "keys": "K corridor on/off, X finish+save",
        }
        draw(ego, ref, dec, diag, text, st)
        _, _, term, trunc, info = env.step(u)
        k += 1
        if term or trunc:
            reason = next((key for key in ("crash_vehicle", "crash_object", "out_of_road", "arrive_dest", "max_step")
                           if info.get(key)), "terminated" if term else "truncated")
            break
        if state["finish"]:
            reason = "user_finish"
            break
        stopped_steps = stopped_steps + 1 if ego.v < 0.1 else 0
        if stopped_steps * dt >= 3.0:        # at a standstill for 3 s: the run cannot progress any further
            reason = "stopped_emergency_brake" if blocker is not None else "stopped"
            break
        if ego.s >= s_stop:
            break
    env.close()
    save_run(args, sc, cfg, placed, steps, decisions, reason, road)


def save_run(args, sc, cfg, placed, steps, decisions, reason, road):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    name = Path(args.scenario).stem
    out = Path(args.out) if args.out else Path("runs/m3_harness") / name / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("steps.csv", steps), ("decisions.csv", decisions)):
        if rows:
            with open(out / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    col = {key: np.array([r[key] for r in steps]) for key in steps[0]}
    fb = col["fallback"].astype(bool)
    reasons = {}
    for r, mode in zip(col["fallback_reason"][fb], col["fallback_mode"][fb]):
        reasons[f"{r}/{mode}"] = reasons.get(f"{r}/{mode}", 0) + 1
    summary = {
        "termination": reason, "steps": len(steps), "duration_s": float(col["t"][-1]),
        "distance_m": float(col["s"][-1] - col["s"][0]),
        "keepout_s": float(cfg["method"]["mpc"]["corridor"].get("keepout_s", 1.0)),
        "corridor_on_pct": float(100 * np.mean(col["corridor_on"])), "corridor_type": args.corridor,
        "ebrake_ref_blocked_steps": int(col["ebrake_ref_blocked"].sum()),
        "overrides": list(args.overrides),
        "fallbacks": int(fb.sum()), "fallback_breakdown": reasons,
        "tube_retries": int(col["tube_retry"].sum()),
        "steps_with_blocked_tube": int((col["tube_blocked"] > 0).sum()),
        "solve_ms_p50": float(np.percentile(col["solve_ms"], 50)),
        "solve_ms_p95": float(np.percentile(col["solve_ms"], 95)),
        "solve_ms_max": float(col["solve_ms"].max()),
        "min_gap_m": float(np.min(col["min_gap"])) if np.isfinite(col["min_gap"]).any() else None,
        "tracking_err_rms_m": float(np.sqrt(np.mean(col["tracking_error"] ** 2))),
        "tracking_err_max_m": float(np.max(col["tracking_error"])),
        "mean_speed": float(np.mean(col["v"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(4, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [2.2, 1, 1, 1]})
    a = ax[0]
    a.axhspan(road.d_min, road.d_max, color="0.93")
    a.axhline(road.d_max, color="goldenrod", lw=2)
    a.axhline(road.d_min, color="0.3", lw=2)
    for c in road.lane_centers[:-1]:
        a.axhline(c - 1.75, color="0.6", lw=0.8, ls="--")
    for s, d in sc["cars"]:
        a.add_patch(Rectangle((s - CAR_LENGTH / 2, d - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH, color="tab:blue"))
    a.plot(placed[:, 0], placed[:, 1], "o", color="crimson", label="placed waypoints")
    a.plot(col["s"], col["d"], color="k", lw=1.5, label="ego path")
    a.plot(col["s"][fb], col["d"][fb], "x", color="tab:red", label="MPC fallback")
    a.set_xlim(col["s"][0] - 5, col["s"][-1] + 30)
    a.set_ylabel("d (m)")
    a.legend(loc="upper right", fontsize=8)
    a.set_title(f"{name}: {reason}, fallbacks {int(fb.sum())}, min gap {summary['min_gap_m']}")
    ax[1].plot(col["s"], col["solve_ms"])
    ax[1].axhline(1000 * float(cfg["method"]["mpc"]["solver"]["max_cpu_time"]), color="tab:red", ls=":",
                  label="max_cpu_time")
    ax[1].set_ylabel("solve (ms)")
    ax[1].legend(fontsize=8)
    ax[2].plot(col["s"], col["v"], label="v")
    ax[2].plot(col["s"], col["ref_v"], "k--", label="MPC v_ref")
    ax[2].set_ylabel("speed (m/s)")
    ax[2].legend(fontsize=8)
    ax[3].plot(col["s"], col["u_s"], label="u_s")
    ax[3].plot(col["s"], col["u_tb"], label="u_tb")
    ax[3].set_ylabel("command")
    ax[3].set_xlabel("s (m)")
    ax[3].legend(fontsize=8)
    for x in ax[1:]:
        x.set_xlim(ax[0].get_xlim())
        x.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)

    print(f"\nend: {reason} after {summary['duration_s']:.1f} s, {summary['distance_m']:.0f} m")
    print(f"  fallbacks {summary['fallbacks']} {reasons or ''}, tube retries {summary['tube_retries']}")
    print(f"  solve time p50 {summary['solve_ms_p50']:.0f} ms, p95 {summary['solve_ms_p95']:.0f} ms, "
          f"max {summary['solve_ms_max']:.0f} ms")
    print(f"  min gap {summary['min_gap_m']} m, tracking err RMS {summary['tracking_err_rms_m']:.2f} m "
          f"(max {summary['tracking_err_max_m']:.2f})")
    print(f"wrote {out}/")


# =============================================================================== cli
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit", help="place waypoints and static cars")
    e.add_argument("--scenario", default=None, help="existing scenario JSON to edit")
    e.add_argument("--out", default=None, help=f"where Save writes (default: --scenario or {SCENARIO_DIR}/scenario.json)")
    e.add_argument("--map-seed", dest="map_seed", type=int, default=None)
    e.add_argument("--ego-lane", dest="ego_lane", type=int, default=None, help="0 = lane next to the centre line")
    e.add_argument("--ego-speed", dest="ego_speed_kmh", type=float, default=None, help="spawn speed (km/h)")
    e.add_argument("--target-speed", dest="target_speed_kmh", type=float, default=None, help="M3 target (km/h)")
    r = sub.add_parser("run", help="run the MPC pipeline on a scenario")
    r.add_argument("scenario")
    r.add_argument("--target-speed", type=float, default=None,
                   help="desired trajectory speed (km/h); overrides the scenario's target_speed_kmh")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (e.g. method.mpc.weights.lateral=4.0)")
    r.add_argument("--fps", type=float, default=10.0, help="render rate cap (10 = real time); 0 = as fast as possible")
    r.add_argument("--no-render", action="store_true", help="headless (logs only)")
    r.add_argument("--hide-corridor", action="store_true", help="do not draw the current corridor box")
    r.add_argument("--demo", action="store_true",
                   help="demo mode: only the cars and the road (no overlays, text panel, MetaDrive navigation "
                        "marks/dashboard); M toggles the overlays + text live")
    r.add_argument("--no-corridor-constraints", action="store_true",
                   help="start with the MPC's corridor constraints off (toggle live with K)")
    r.add_argument("--corridor", choices=["road", "stock"], default="road",
                   help="road = harness road-aligned, edge-to-edge corridor minus occupied lanes (default); "
                        "stock = M3's CorridorBuilder (rotates with the trajectory during lane changes)")
    r.add_argument("--corridor-inset", type=float, default=None,
                   help="road corridor: pull the fixed lateral faces in from the exterior lane centres (m); "
                        "negative widens. Default -0.3 (faces 0.3 m outside the exterior lane centres: "
                        "[-7.3, +0.3]; up to about -0.8 keeps the ego body on the road)")
    r.add_argument("--out", default=None, help="output dir (default runs/m3_harness/<scenario>/<timestamp>)")
    args = p.parse_args()
    if args.cmd == "edit":
        cmd_edit(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
