#!/usr/bin/env python
"""Validation suite (spec §17) against the real simulator. One MetaDrive per process, so each check runs in
its own subprocess when invoked with --all.

    python tests/validate.py --all                 # every check, report in runs/validation/
    python tests/validate.py --check env           # one check in-process

Checks:
  env          ego dynamics pinned (static_default) + spawn speed/lane centre; RoadFrame round-trips on the
               MetaDrive map; forced off-road terminates; determinism
  pid          §17.3 M2 PID standalone (no traffic): lane-change quintic tracking + speed hold, scripted (d*, v*)
  mpc          §17.4 M3 as trained (road corridor, e-brake gate) with scripted waypoints and scripted cars:
               (a) passes a 35 km/h lead, (b) stops for a stopped car in its path, (c) fallback chain + recovery
  smoke        §17.6 each wrapper (m1/m2/m3) runs complete episodes with random actions; logs valid
  b0           §17.7 B0 completes evaluation episodes through the harness (all eval seeds)
  train_plumbing  SB3 PPO entry point runs end-to-end for a few hundred control steps (plumbing only)
(§17.2 throughput: tests/benchmark_throughput.py; §17.5 safe-set: pytest tests/unit/test_controllers.py)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from handoff.config import REPO_ROOT, kmh, load_config  # noqa: E402

OUT = REPO_ROOT / "runs" / "validation"
PID_TOL_LANE_CHANGE = 0.30   # m, max cross-track error during a lane change
PID_TOL_STEADY = 0.10        # m, cross-track error once settled
PID_TOL_SPEED = 0.5          # m/s, speed error once settled


# ----------------------------------------------------------------------------------------------- helpers
def _controlled_env(method, ego_lane=1, ego_kmh=54.0, overrides=(), run_dir=None):
    """Method env on map seed 0 with NO MetaDrive traffic, ego spawned in ``ego_lane`` at ``ego_kmh``. Add cars
    after ``env.reset`` with ``_spawn_cars``."""
    from metadrive.component.pgblock.first_block import FirstPGBlock

    from handoff.rl.env.factory import make_env

    cfg = load_config(method, overrides=["scenario.traffic.density=0", "sim.ego.random_spawn_lane=false",
                                         f"sim.ego.spawn_speed_km_h={float(ego_kmh)}", *overrides])
    cfg["sim"]["metadrive_overrides"]["agent_configs"] = {"default_agent": {
        "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, int(ego_lane))}}
    env = make_env(cfg, is_eval=True, run_dir=run_dir)
    return env, cfg


def _spawn_cars(env, cars):
    """cars: [(s, lane, speed_kmh)]. Spawned as TrafficDefaultVehicle on the lane centre; speed 0 = static,
    otherwise held at that speed along the road after every control step (no IDM). Returns the vehicles."""
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

    core = env.core
    road = core.road
    held = []
    for s, lane, kmh_ in cars:
        x, y = road.to_world(float(s), float(road.lane_centers[lane]))
        h = float(road.heading_at(float(s)))
        v = core.md.engine.spawn_object(TrafficDefaultVehicle, vehicle_config={"spawn_velocity": None},
                                        position=[x, y], heading=h)
        if kmh_ <= 0:
            v.set_velocity([0.0, 0.0])
            v.set_static(True)
        held.append((v, kmh(kmh_)))
        core.scripted_vehicles.append(v)

    def hold(core_, _dec, _diag):
        for v, spd in held:
            if spd > 0:
                h = float(road.heading_at(road.to_frenet(*v.position)[0]))
                v.set_velocity([spd * np.cos(h), spd * np.sin(h)])

    core.overlay_hook = hold
    hold(core, None, None)
    core.objects = core._extract_objects()
    return [v for v, _ in held]


def _car_s(env, veh) -> float:
    return float(env.core.road.to_frenet(*veh.position)[0])


# ----------------------------------------------------------------------------------------------- checks
def check_env() -> dict:
    """Ego dynamics pinned (static_default: no per-episode resampling), spawn speed + lane centre, RoadFrame,
    forced off-road termination, determinism."""
    from handoff.rl.env.factory import make_env

    cfg = load_config("m1")
    env = make_env(cfg, is_eval=True)
    core = env.core
    res = {"spawn": []}
    ok = True
    want = {"max_engine_force": 800, "max_brake_force": 150, "wheel_friction": 0.9, "max_steering": 40}
    v0 = kmh(cfg["sim"]["ego"]["spawn_speed_km_h"])
    for seed in cfg["scenario"]["seeds"]["eval"][:6]:
        env.reset(options={"episode_seed": seed})
        dyn = core.md.agent.get_dynamics_parameters()
        dyn_ok = all(abs(float(dyn[k]) - v) < 1e-6 for k, v in want.items())
        e = core.ego
        lane = core.road.nearest_lane(e.d)
        spawn_err = [abs(e.d - core.road.lane_centers[lane]), abs(e.v - v0)]
        good = dyn_ok and max(spawn_err) < 0.05
        res["spawn"].append({"seed": seed, "vehicle": type(core.md.agent).__name__, "lane": lane,
                             "err_d_v": [round(x, 4) for x in spawn_err],
                             "dynamics": {k: float(v) for k, v in dyn.items()}, "ok": good})
        ok &= good

    # RoadFrame round trip on the MetaDrive map + agreement with MetaDrive lane geometry
    road = core.road
    net = core.md.current_map.road_network
    errs = []
    for lanes in [net.graph[">>>"]["1S0_0_"], net.graph["3S0_0_"]["4S0_0_"]]:
        for i, ln in enumerate(lanes):
            for lon in np.linspace(1, ln.length - 1, 9):
                x, y = ln.position(lon, 0.0)
                s, d = road.to_frenet(x, y)
                x2, y2 = road.to_world(s, d)
                errs.append(max(abs(x2 - x), abs(y2 - y), abs(d - road.lane_centers[i])))
    rt_ok = max(errs) < 1e-6
    ok &= rt_ok
    res["roadframe"] = {"max_err": float(max(errs)), "length": road.length, "lane_centers": road.lane_centers,
                        "d_min": road.d_min, "d_max": road.d_max, "ok": rt_ok}

    # termination: steer hard right until off-road
    env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][0]})
    reason = None
    for _ in range(300):
        _, _, te, tr, info = env.step(np.array([-0.6, 0.3]))
        if te or tr:
            reason = info["termination"]
            break
    res["forced_off_road_termination"] = reason
    ok &= reason == "off_road"
    env.close()

    # determinism (needed for replay by re-simulation): same seed + same actions -> identical trajectory
    res["determinism"] = _determinism()
    ok &= res["determinism"]["ok"]
    res["ok"] = bool(ok)
    return res


def _determinism() -> dict:
    from handoff.rl.env.factory import make_env

    cfg = load_config("m3")
    env = make_env(cfg, is_eval=True)
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1, 1, (25, env.action_space.shape[0]))
    trajs = []
    for _ in range(2):
        env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][3]})
        xs = []
        for a in actions:
            _, _, te, tr, _ = env.step(a)
            xs.append((env.core.ego.x, env.core.ego.y, env.core.ego.v))
            if te or tr:
                break
        trajs.append(np.array(xs))
    env.close()
    same_len = len(trajs[0]) == len(trajs[1])
    dev = float(np.max(np.abs(trajs[0] - trajs[1]))) if same_len else float("inf")
    return {"max_abs_diff": dev, "steps": len(trajs[0]), "ok": bool(same_len and dev == 0.0)}


def check_pid() -> dict:
    """M2: lane 1 -> lane 0 -> lane 2 -> lane 1 with speed changes; no traffic; scripted (d*, v*) actions."""
    env, cfg = _controlled_env("m2", ego_lane=1, ego_kmh=54.0)
    env.reset(options={"episode_seed": 0})
    core = env.core
    road = core.road
    plan = [(road.lane_centers[1], kmh(60), 6), (road.lane_centers[0], kmh(60), 10),
            (road.lane_centers[2], kmh(70), 14),
            (road.lane_centers[1], kmh(50), 10)]
    rows = []
    for d_star, v_star, n_dec in plan:
        for j in range(n_dec):
            obs, r, te, tr, info = env.step(env.encode(d_star, v_star))
            rows.append((j, d_star, v_star, core.ego.d, core.ego.v,
                         core.recorder.steps["tracking_error"][-env.period:]))
            if te or tr:
                break
    env.close()
    max_cte = max(float(np.nanmax(r[5])) for r in rows)
    # settled = last 4 decisions (2 s) of each segment
    settled = [r for r in rows if r[0] >= 6] if rows else []
    steady_cte = max(abs(r[3] - r[1]) for r in rows if r[0] >= (6 if r[1] != road.lane_centers[1] else 3))
    speed_err = [abs(r[4] - r[2]) for r in settled]
    res = {"gains": {k: cfg["method"]["pid"][k] for k in ("heading", "lateral", "speed")},
           "max_cross_track_m": max_cte, "steady_cross_track_m": steady_cte,
           "max_settled_speed_err_ms": max(speed_err) if speed_err else None,
           "tolerances": {"lane_change": PID_TOL_LANE_CHANGE, "steady": PID_TOL_STEADY, "speed": PID_TOL_SPEED},
           "terminated_early": len(rows) < sum(p[2] for p in plan)}
    res["ok"] = bool(max_cte < PID_TOL_LANE_CHANGE and steady_cte < PID_TOL_STEADY and speed_err
                     and max(speed_err) < PID_TOL_SPEED and not res["terminated_early"])
    return res


def check_mpc() -> dict:
    """M3 as trained (road corridor, e-brake gate, no safe set), scripted waypoints via ``encode``."""
    out = {}
    # (a) pass a slow lead: lane 1 -> lane 0 while passing -> back to lane 1
    env, cfg = _controlled_env("m3", ego_lane=1, ego_kmh=60.0)
    env.reset(options={"episode_seed": 0})
    core = env.core
    L = core.road.lane_centers
    (lead,) = _spawn_cars(env, [(130.0, 1, 35.0)])
    passed, reason, min_gap = False, None, np.inf
    for _k in range(80):
        ahead = _car_s(env, lead) - core.ego.s
        d_tgt = L[0] if ahead > -15 else L[1]
        obs, r, te, tr, info = env.step(env.encode([40.0, 40.0], [d_tgt, d_tgt], v=kmh(70)))
        min_gap = min(min_gap, min(core.recorder.steps["min_gap"][-env.period:]))
        if core.ego.s > _car_s(env, lead) + 20:
            passed = True
        if te or tr:
            reason = info["termination"]
            break
    s = core.recorder.steps
    st = np.array(s["solve_time"])
    st = st[st > 0]
    out["pass_lead"] = {"passed": passed, "termination": reason, "min_gap": float(min_gap),
                        "final_d": core.ego.d,
                        "solve_ms_mean": float(st.mean() * 1e3) if len(st) else None,
                        "solve_ms_p95": float(np.percentile(st, 95) * 1e3) if len(st) else None,
                        "fallbacks": int(np.sum(s["fallback"])), "gated_brake_steps": int(np.sum(s["gated_brake"]))}
    out["pass_lead"]["ok"] = bool(passed and reason in (None, "success") and min_gap > 0.3)
    env.close()

    # (b) stopped car in the ego's lane, waypoints straight through it: the gate / MPC must stop the ego
    env, cfg = _controlled_env("m3", ego_lane=1, ego_kmh=50.0)
    env.reset(options={"episode_seed": 0})
    core = env.core
    (stopped,) = _spawn_cars(env, [(170.0, 1, 0.0)])
    reason, min_gap = None, np.inf
    for _k in range(40):
        obs, r, te, tr, info = env.step(env.encode([25.0, 25.0], [L[1], L[1]], v=kmh(50)))
        min_gap = min(min_gap, min(core.recorder.steps["min_gap"][-env.period:]))
        if te or tr:
            reason = info["termination"]
            break
    out["stopped_vehicle"] = {"termination": reason, "min_gap": float(min_gap), "final_s": core.ego.s,
                              "final_v": core.ego.v, "final_d": core.ego.d, "stopped_vehicle_s": _car_s(env, stopped),
                              "gated_brake_steps": int(np.sum(core.recorder.steps["gated_brake"]))}
    out["stopped_vehicle"]["ok"] = bool(reason != "collision" and min_gap > 0.2 and core.ego.v < 0.5)
    env.close()

    # (c) fallback chain: break the solver -> (shifted previous plan if max_fallback_shift > 0) -> emergency
    # brake; restore it -> recovery
    env, cfg = _controlled_env("m3", ego_lane=1, ego_kmh=60.0)
    env.reset(options={"episode_seed": 0})
    core = env.core
    env.step(env.encode([30.0], [L[1]]))  # healthy solves first
    healthy = int(np.sum(core.recorder.steps["fallback"]))
    real = env.mpc.solver

    def broken(**kw):
        raise RuntimeError("forced solver failure (validation)")

    env.mpc.solver = broken
    for _ in range(3):
        _, _, te, tr, _ = env.step(env.encode([30.0], [L[1]]))
        if te or tr:
            break
    statuses = core.recorder.status[-30:]
    env.mpc.solver = real
    env.step(env.encode([30.0], [L[1]]))
    recovered = not core.recorder.steps["fallback"][-1]
    modes = {m.split(":")[-1] for m in statuses if m.startswith("fallback")}
    expected = {"emergency_brake"} | ({"shifted_previous"} if int(cfg["method"]["mpc"]["max_fallback_shift"]) > 0
                                      else set())
    out["fallback"] = {"healthy_fallbacks": healthy, "modes_seen": sorted(modes), "modes_expected": sorted(expected),
                       "recovered": bool(recovered), "speed_after_brake": core.ego.v}
    out["fallback"]["ok"] = bool(healthy == 0 and expected <= modes and recovered)
    env.close()
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def check_smoke(n_episodes: int = 2) -> dict:
    from handoff.rl.env.factory import make_env
    from handoff.rl.env.rewards import TERM_NAMES
    from handoff.rl.records.recorder import load_episode
    from handoff.rl.records.run_dir import CsvAppender

    out = {}
    tmp = OUT / "smoke"
    for method in ("m1", "m2", "m3"):
        cfg = load_config(method)
        run_dir = tmp / method
        env = make_env(cfg, run_dir=run_dir, is_eval=True)
        rng = np.random.default_rng(0)
        sums = []
        t0 = time.time()
        for ep in range(n_episodes):
            env.reset(options={"episode_seed": cfg["scenario"]["seeds"]["eval"][ep]})
            while True:
                _, r, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
                assert np.isfinite(r)
                if te or tr:
                    sums.append(info["episode_summary"])
                    break
        env.close()
        CsvAppender(run_dir / "episodes.csv").write(sums)
        problems = []
        for s in sums:
            for k in ("return", "mean_speed", "sim_time", "control_steps", "decisions"):
                if not np.isfinite(float(s[k])):
                    problems.append(f"{k} not finite")
            ep = load_episode(s["log_file"])
            d = ep["data"]
            for n in TERM_NAMES:
                if f"term_{n}" not in d or len(d[f"term_{n}"]) != int(s["control_steps"]):
                    problems.append(f"term {n} missing/short")
            if len(d["dec_k"]) != int(s["decisions"]):
                problems.append("decision count mismatch")
            if abs(float(np.sum(d["step_reward"])) - float(s["return"])) > 1e-6:
                problems.append("return != sum of step rewards")
            if method == "m3" and len(d["dec_waypoints_raw"]) != int(s["decisions"]):
                problems.append("waypoint log length mismatch")
        out[method] = {"episodes": [(s["termination"], int(s["control_steps"]), round(float(s["return"]), 2))
                                    for s in sums], "wall_s": round(time.time() - t0, 1), "problems": problems,
                       "ok": not problems}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def check_b0() -> dict:
    out_dir = OUT / "b0_eval"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--method", "b0", "--out", str(out_dir),
           "--n-workers", "4"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    summary = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    res = {"returncode": p.returncode, "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]
           if p.returncode else "", "summary": {k: summary.get(k) for k in ("n_episodes", "success", "collision",
                                                                           "off_road", "timeout", "mean_speed",
                                                                           "time_to_complete", "min_gap")}}
    res["ok"] = bool(p.returncode == 0 and summary.get("n_episodes") == 20)
    return res


def check_train_plumbing() -> dict:
    """PPO entry point end-to-end (vec env, callbacks, eval subprocess, checkpoints, CSVs) for a few hundred
    control steps per method. Not training: no meaningful learning happens at this budget."""
    from handoff.rl.train import train

    out = {}
    for method in ("m1", "m2", "m3"):
        cfg = load_config(method, overrides=[
            "training.total_control_steps=300", "training.n_envs=2", "eval.n_episodes=2",
            "eval.freq_control_steps=200", "training.checkpoint_freq_control_steps=200", "ppo.n_steps=32",
            "ppo.batch_size=32", "ppo.n_epochs=1"])
        run_dir = OUT / "train_plumbing" / method
        t0 = time.time()
        try:
            train(cfg, run_dir=run_dir, verbose=0)
            files = {f: (run_dir / f).exists() for f in ("train_progress.csv", "episodes.csv", "config.yaml",
                                                         "checkpoints/final.zip", "checkpoints/best.zip")}
            import pandas as pd

            ev = pd.read_csv(run_dir / "episodes.csv")
            out[method] = {"files": files, "eval_rows": int((ev["is_eval"] == 1).sum()),
                           "train_rows": int((ev["is_eval"] == 0).sum()), "wall_s": round(time.time() - t0, 1)}
            out[method]["ok"] = all(files.values()) and out[method]["eval_rows"] >= 4
        except Exception:  # noqa: BLE001
            out[method] = {"error": traceback.format_exc()[-2000:], "ok": False}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


CHECKS = {"env": check_env, "pid": check_pid, "mpc": check_mpc, "smoke": check_smoke, "b0": check_b0,
          "train_plumbing": check_train_plumbing}


def _run_one(name: str) -> dict:
    t0 = time.time()
    try:
        res = CHECKS[name]()
    except Exception:  # noqa: BLE001
        res = {"ok": False, "error": traceback.format_exc()}
    res["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", choices=list(CHECKS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--skip", nargs="*", default=[])
    args = p.parse_args()
    if args.check:
        res = _run_one(args.check)
        print(json.dumps(res, indent=2, default=str))
        sys.exit(0 if res["ok"] else 1)
    if not args.all:
        p.error("--check NAME or --all")
    summary = {}
    for name in CHECKS:
        if name in args.skip:
            continue
        r = subprocess.run([sys.executable, __file__, "--check", name], capture_output=True, text=True)
        rep = OUT / f"{name}.json"
        res = json.loads(rep.read_text()) if rep.exists() else {"ok": False}
        summary[name] = res.get("ok", False)
        print(f"{'PASS' if summary[name] else 'FAIL'}  {name}  ({res.get('wall_s', '?')} s)", flush=True)
        if not summary[name] and r.returncode != 0 and "error" in res:
            print(res["error"][-1500:])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(0 if all(summary.values()) else 1)


if __name__ == "__main__":
    main()
