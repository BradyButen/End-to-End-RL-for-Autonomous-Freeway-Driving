#!/usr/bin/env python
"""Closed-loop tracking test of M2's PID controller in MetaDrive on a sinusoidal reference.

No traffic, fixed ego dynamics (vehicle_model "static_default": engine 800, brake 150), stock freeway map.
The reference weaves around the middle lane: after a smooth ramp from the spawn position,
    d_ref(s) = c + A sin(2 pi (s - s_ramp) / wavelength)
    v_ref(t) = v0 + dv sin(2 pi t / speed_period)
and the controller is CascadedPIDController with the gains in configs/method/m2.yaml (or --old-gains).
The reference is one long trajectory (no replanning), so this isolates tracking from M2's RL/replan loop.

Sim in the loop: MetaDrive's 3D window shows the run live, with the reference drawn as a yellow line and
the tracking errors in the text panel (needs a display: WSLg / X11). --topdown uses the 2D renderer instead,
--no-render runs headless. Writes <out>/log.csv, <out>/summary.json and <out>/tracking.png.

    python scripts/track_sine.py
    python scripts/track_sine.py --amp 3.0 --wavelength 150 --speed 20 --speed-amp 2
    python scripts/track_sine.py --old-gains            # previous m2.yaml gains, for comparison
    python scripts/track_sine.py --set method.pid.lateral=[0.1,0.0005,0.0]
    python scripts/track_sine.py --no-render            # headless
    python scripts/track_sine.py --teleop               # keyboard speed: Up/Down +/-1 m/s, Space stop, X finish

Teleop keeps the lateral sine (looked up by arc length) and replaces the speed sine with a keyboard setpoint,
ramped at M2's trajectory accel/decel limits (2 / 4 m/s^2) so the controller sees the same kind of reference
as in M2. It runs at real time (--fps 10) unless --fps is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

OLD_GAINS = {"heading": [1.7, 0.01, 3.5], "lateral": [0.3, 0.002, 0.05], "speed": [0.6, 0.02, 0.0]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp", type=float, default=1.75, help="lateral amplitude A (m); <= 3 stays on the road")
    p.add_argument("--wavelength", type=float, default=250.0,
                   help="lateral wavelength (m); 250 m at 20 m/s = 12.5 s period")
    p.add_argument("--speed", type=float, default=20.0, help="mean speed v0 (m/s); v0 + dv <= 22.2")
    p.add_argument("--speed-amp", type=float, default=2.0, help="speed amplitude dv (m/s); 0 = constant")
    p.add_argument("--speed-period", type=float, default=40.0, help="speed period (s)")
    p.add_argument("--ramp", type=float, default=60.0, help="distance to blend from the spawn d into the sine (m)")
    p.add_argument("--lane", type=int, default=1, help="lane whose centre the sine is centred on (0 = leftmost)")
    p.add_argument("--map-seed", type=int, default=0, help="MetaDrive seed (map block lengths)")
    p.add_argument("--old-gains", action="store_true", help="use the pre-tuning m2.yaml gains")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable")
    p.add_argument("--out", default=None, help="output dir (default runs/track_sine/<timestamp>)")
    p.add_argument("--no-render", action="store_true", help="run headless (no MetaDrive window)")
    p.add_argument("--topdown", action="store_true", help="2D top-down renderer instead of the 3D window")
    p.add_argument("--fps", type=float, default=None,
                   help="cap the render rate (e.g. 10 = real time at 0.1 s control steps); default as fast as possible")
    p.add_argument("--teleop", action="store_true",
                   help="set the speed from the keyboard instead of the sine (starts at --speed)")
    p.add_argument("--speed-step", type=float, default=1.0, help="teleop: m/s per key press")
    return p.parse_args()


class SpeedTeleop:
    """Keyboard speed setpoint, shaped like M2's speed profile (ramp at the trajectory accel/decel limits).

    Keys (in the MetaDrive window): Up / Down = +/- speed_step, Space = stop (target 0), X = finish and save.
    MetaDrive's own keys still work (B/Q camera, mouse wheel zoom); avoid R (resets the episode) and Esc (exits
    without saving).
    """

    V_MAX = 80 / 3.6

    def __init__(self, v0, step, accel, decel):
        self.target = float(v0)
        self.v = float(v0)
        self.a = 0.0
        self.step_size = float(step)
        self.accel, self.decel = float(accel), float(decel)
        self.finish = False
        self._prev_keys = {}

    def up(self):
        self.target = min(self.target + self.step_size, self.V_MAX)

    def down(self):
        self.target = max(self.target - self.step_size, 0.0)

    def stop(self):
        self.target = 0.0

    def quit(self):
        self.finish = True

    def bind_panda(self, engine):
        for key, fn in (("arrow_up", self.up), ("arrow_up-repeat", self.up), ("arrow_down", self.down),
                        ("arrow_down-repeat", self.down), ("space", self.stop), ("x", self.quit)):
            engine.accept(key, fn)

    def poll_pygame(self):
        """Top-down mode: edge-detect key presses (MetaDrive's renderer pumps the pygame events)."""
        import pygame

        pressed = pygame.key.get_pressed()
        for key, fn in ((pygame.K_UP, self.up), (pygame.K_DOWN, self.down), (pygame.K_SPACE, self.stop),
                        (pygame.K_x, self.quit)):
            if pressed[key] and not self._prev_keys.get(key, False):
                fn()
            self._prev_keys[key] = bool(pressed[key])

    def advance(self, dt):
        """Next control step's commanded speed and acceleration (rate-limited toward the target)."""
        dv = float(np.clip(self.target - self.v, -self.decel * dt, self.accel * dt))
        self.a = dv / dt
        self.v += dv
        return self.v, self.a


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build_reference(road, s0, d0, args, dt=0.05):
    """One long reference from the ego's spawn (s0, d0) to near the end of the route."""
    from handoff.rl.methods.common import Reference

    s_end = road.length - 15.0
    v_min = max(args.speed - abs(args.speed_amp), 1.0)
    T = (s_end - s0) / v_min + 5.0
    t = np.arange(0.0, T, dt)
    v = args.speed + args.speed_amp * np.sin(2 * np.pi * t / args.speed_period)
    a = np.gradient(v, dt)
    s = s0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    keep = s <= s_end
    t, v, a, s = t[keep], v[keep], a[keep], s[keep]
    c = road.lane_centers[args.lane]
    u = s - s0
    sine = c + args.amp * np.sin(2 * np.pi * np.maximum(u - args.ramp, 0.0) / args.wavelength)
    w = smoothstep(u / args.ramp)
    d = (1 - w) * d0 + w * sine
    dd_ds = np.gradient(d, s)
    d_dot = dd_ds * v
    d_ddot = np.gradient(d_dot, dt)
    x, y = road.to_world(s, d)
    psi = road.heading_at(s) + np.arctan(dd_ds)
    return Reference(t=t, s=s, d=d, x=np.asarray(x), y=np.asarray(y), psi=np.asarray(psi), v=v, a=a,
                     d_dot=d_dot, d_ddot=d_ddot)


def main():
    args = parse_args()
    render = not args.no_render
    if args.topdown and render:
        # the project defaults pygame to the dummy driver for headless BEV; the top-down window needs a real one
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            os.environ.pop("SDL_VIDEODRIVER")
        os.environ["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER_VIEW", "x11")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from metadrive.envs.top_down_env import TopDownMetaDrive

    from handoff.config import load_config
    from handoff.rl.methods.common import ActuationMap
    from handoff.rl.methods.m2_pid import CascadedPIDController
    from handoff.simulator.metadrive_env import build_road_frame, make_metadrive_config
    from handoff.simulator.overlays import OverlayDrawer
    from handoff.simulator.state import make_ego_state

    if args.teleop:
        if args.no_render:
            raise SystemExit("--teleop needs a window (drop --no-render)")
        args.speed_amp = 0.0                     # the reference is built at constant speed; keys set v_ref live
        if not args.fps:
            args.fps = 10.0                      # real time, so key presses are usable
    if args.speed + abs(args.speed_amp) > 80 / 3.6:
        print(f"WARNING: peak speed {args.speed + abs(args.speed_amp):.1f} m/s exceeds MetaDrive's 22.2 m/s engine cut")
    overrides = [
        "scenario.traffic.density=0",
        "sim.ego.random_spawn_lane=false",
        f"sim.ego.spawn_speed_km_h={args.speed * 3.6}",
        "sim.on_continuous_line_done=false",     # keep logging if the car touches a solid line
    ] + (["sim.horizon=100000"] if args.teleop else []) + list(args.overrides)
    cfg = load_config("m2", overrides=overrides)
    if args.old_gains:
        cfg["method"]["pid"].update(OLD_GAINS)
    pid_cfg = cfg["method"]["pid"]
    dt = float(cfg["sim"]["control_dt"])

    md_cfg = make_metadrive_config(cfg, render=render and not args.topdown)
    md_cfg["vehicle_config"]["vehicle_model"] = "static_default"   # fixed dynamics (no per-episode sampling)
    md_cfg["num_scenarios"] = 1
    md_cfg["start_seed"] = args.map_seed
    if args.fps:
        md_cfg["force_render_fps"] = args.fps
    env = TopDownMetaDrive(md_cfg)
    env.reset(seed=args.map_seed)
    road = build_road_frame(env)
    agent = env.agent
    dyn = agent.get_dynamics_parameters()

    def ego_state(k):
        x, y = agent.position
        return make_ego_state(k * dt, x, y, agent.heading_theta, agent.speed, float(agent.steering),
                              float(agent.throttle_brake), agent.LENGTH, agent.WIDTH, road)

    ego = ego_state(0)
    ref = build_reference(road, ego.s, ego.d, args)
    ctrl = CascadedPIDController.from_config(pid_cfg, ActuationMap(cfg["actuation"]["table"]), dt)
    ctrl.reset()
    ctrl.new_reference()

    print(f"vehicle {type(agent).__name__}: " + ", ".join(f"{k} {float(v):g}" for k, v in dyn.items()))
    print(f"gains: heading {pid_cfg['heading']}  lateral {pid_cfg['lateral']}  speed {pid_cfg['speed']}")
    print(f"route {road.length:.0f} m, reference {ref.s[-1] - ref.s[0]:.0f} m over {ref.t[-1]:.1f} s, "
          f"peak ref a_y {np.max(np.abs(ref.v ** 2 * np.gradient(np.gradient(ref.d, ref.s), ref.s))):.2f} m/s^2")

    drawer = OverlayDrawer(env.engine) if render and not args.topdown else None

    teleop = None
    if args.teleop:
        tr = cfg["method"]["trajectory"]
        teleop = SpeedTeleop(args.speed, args.speed_step, tr["accel_limit"], tr["decel_limit"])
        if not args.topdown:
            teleop.bind_panda(env.engine)
        print("teleop: Up/Down = speed +/-%g m/s, Space = stop, X = finish and save "
              "(avoid R = MetaDrive reset, Esc = exit without saving)" % args.speed_step)

    def show(row):
        """Draw the reference ahead of the ego and the live errors in MetaDrive's window."""
        if not render:
            return
        text = {
            "t": f"{row['t']:.1f} s",
            "cross-track": f"{row['cross_track']:+.3f} m",
            "heading err": f"{np.degrees(row['heading_err']):+.2f} deg",
            "speed": f"{row['v']:.2f} / ref {row['ref_v']:.2f} m/s",
            "along-track lag": f"{row['along_track']:+.2f} m",
            "u_s / u_tb": f"{row['u_s']:+.3f} / {row['u_tb']:+.3f}",
        }
        if teleop is not None:
            text["target speed"] = f"{teleop.target:.1f} m/s ({teleop.target * 3.6:.0f} km/h)"
            text["keys"] = "Up/Down speed, Space stop, X finish"
        if drawer is not None:
            m = (ref.s >= ego.s - 10.0) & (ref.s <= ego.s + 150.0)
            drawer.draw({"ref_xy": np.stack([ref.x[m], ref.y[m]], axis=1)})
            env.render(text=text)
        else:
            env.render(mode="topdown", text=text, screen_size=(900, 900), film_size=(4000, 4000),
                       target_agent_heading_up=False, semantic_map=False)

    rows = []
    reason = "end_of_reference"
    k = 0
    s_cmd = ego.s            # teleop: arc length of the commanded speed profile (for the along-track lag)
    while True:
        t = k * dt
        if teleop is not None:
            if args.topdown:
                teleop.poll_pygame()
            if teleop.finish:
                reason = "teleop_finish"
                break
            # The lateral reference is looked up by arc length, so only the speed channel changes: the
            # controller reads v/a at t + dt, so fill them with the next commanded (v, a).
            v_next, a_next = teleop.advance(dt)
            ref.v[:] = v_next
            ref.a[:] = a_next
            s_cmd += v_next * dt
            s_ref_t = s_cmd
        else:
            s_ref_t = ref.at_time(t)["s"]
        u, diag = ctrl.step(ego, ref, [])
        rows.append(dict(
            t=t, x=ego.x, y=ego.y, s=ego.s, d=ego.d, psi=ego.psi, v=ego.v,
            ref_d=diag["ref_d"], ref_psi=diag["ref_psi"], ref_v=diag["ref_v"], ref_s_time=s_ref_t,
            cross_track=diag["cross_track"], heading_err=diag["heading_error"], speed_err=diag["speed_error"],
            along_track=s_ref_t - ego.s, u_s=float(u[0]), u_tb=float(u[1])))
        _, _, term, trunc, info = env.step(u)
        k += 1
        ego = ego_state(k)
        show(rows[-1])
        if term or trunc:
            reason = next((key for key in ("crash_vehicle", "crash_object", "out_of_road", "arrive_dest",
                                           "max_step") if info.get(key)), "terminated" if term else "truncated")
            break
        if ego.s >= ref.s[-1] - 1.0 or (teleop is None and k * dt >= ref.t[-1]):
            break
    env.close()
    if not rows:
        raise SystemExit("finished before the first control step; nothing to save")

    out = Path(args.out) if args.out else Path("runs/track_sine") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    col = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    settle = col["s"] - col["s"][0] >= args.ramp            # metrics once the sine has started
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))   # noqa: E731
    summary = {
        "termination": reason, "duration_s": float(col["t"][-1]), "distance_m": float(col["s"][-1] - col["s"][0]),
        "gains": {key: pid_cfg[key] for key in ("heading", "lateral", "speed")},
        "args": vars(args), "dynamics": {key: float(v) for key, v in dyn.items()},
        "cross_track_rms_m": rms(col["cross_track"][settle]),
        "cross_track_max_m": float(np.max(np.abs(col["cross_track"][settle]))),
        "heading_err_rms_deg": float(np.degrees(rms(col["heading_err"][settle]))),
        "heading_err_max_deg": float(np.degrees(np.max(np.abs(col["heading_err"][settle])))),
        "speed_err_rms": rms(col["speed_err"][settle]),
        "speed_err_max": float(np.max(np.abs(col["speed_err"][settle]))),
        "along_track_rms_m": rms(col["along_track"][settle]),
        "along_track_max_m": float(np.max(np.abs(col["along_track"][settle]))),
        "u_s_saturated_pct": float(100 * np.mean(np.abs(col["u_s"]) >= 1.0)),
        "u_tb_saturated_pct": float(100 * np.mean(np.abs(col["u_tb"]) >= 1.0)),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    sx = col["s"] - col["s"][0]
    ax[0].plot(sx, col["ref_d"], "k--", label="d_ref (at ego s)")
    ax[0].plot(sx, col["d"], label="d")
    for c in road.lane_centers:
        ax[0].axhline(c, color="0.85", lw=0.8)
    ax[0].axhline(road.d_min, color="0.5", lw=1)
    ax[0].axhline(road.d_max, color="0.5", lw=1)
    ax[0].set_ylabel("lateral d (m)")
    ax[0].legend(loc="upper right")
    ax[1].plot(sx, col["cross_track"], label="cross-track (m)")
    ax[1].plot(sx, np.degrees(col["heading_err"]), label="heading (deg)")
    ax[1].set_ylabel("error")
    ax[1].legend(loc="upper right")
    ax[2].plot(sx, col["ref_v"], "k--", label="v_ref (t + dt)")
    ax[2].plot(sx, col["v"], label="v")
    ax[2].set_ylabel("speed (m/s)")
    ax[2].legend(loc="upper right")
    ax[3].plot(sx, col["along_track"])
    ax[3].set_ylabel("along-track lag (m)\n(not fed back)")
    ax[4].plot(sx, col["u_s"], label="u_s")
    ax[4].plot(sx, col["u_tb"], label="u_tb")
    ax[4].set_ylabel("command")
    ax[4].set_xlabel("distance along route (m)")
    ax[4].legend(loc="upper right")
    for a in ax:
        a.grid(True, alpha=0.3)
        a.axvline(args.ramp, color="0.6", ls=":", lw=1)
    gains = "old m2.yaml gains" if args.old_gains else "m2.yaml gains"
    fig.suptitle(f"M2 PID on sine: A={args.amp} m, wavelength={args.wavelength} m, "
                 + ("v = keyboard teleop" if args.teleop else f"v={args.speed}+-{args.speed_amp} m/s")
                 + f" ({gains}); end: {reason}")
    fig.tight_layout()
    fig.savefig(out / "tracking.png", dpi=110)

    print(f"\nend: {reason} after {summary['duration_s']:.1f} s, {summary['distance_m']:.0f} m")
    print(f"metrics after the {args.ramp:g} m ramp:")
    print(f"  cross-track  RMS {summary['cross_track_rms_m']:.3f} m   max {summary['cross_track_max_m']:.3f} m")
    print(f"  heading      RMS {summary['heading_err_rms_deg']:.2f} deg max {summary['heading_err_max_deg']:.2f} deg")
    print(f"  speed        RMS {summary['speed_err_rms']:.3f} m/s max {summary['speed_err_max']:.3f} m/s")
    print(f"  along-track  RMS {summary['along_track_rms_m']:.2f} m   max {summary['along_track_max_m']:.2f} m")
    print(f"  saturated    u_s {summary['u_s_saturated_pct']:.1f}%  u_tb {summary['u_tb_saturated_pct']:.1f}%")
    print(f"wrote {out}/log.csv, summary.json, tracking.png")


if __name__ == "__main__":
    main()
