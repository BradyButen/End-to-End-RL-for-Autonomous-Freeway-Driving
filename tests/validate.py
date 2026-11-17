#!/usr/bin/env python
"""Validation suite (spec §17) against the real simulator. One MetaDrive per process, so each check runs in
its own subprocess when invoked with --all.

    python tests/validate.py --all                 # every check, report in runs/validation/
    python tests/validate.py --check env           # one check in-process

Checks:
  env          scenario spawns as specified; ego dynamics pinned; RoadFrame round-trips on the MetaDrive map;
               traffic speed overrides hold; geometric off-road check agrees with MetaDrive; determinism
  pid          §17.3 PID standalone: lane-change quintic tracking + speed hold (no traffic near, no RL)
  mpc          §17.4 MPC standalone with hand-placed waypoints: (a) passes the lead, (b) avoids a stopped
               vehicle in its path with projection disabled, (c) fallback path exercised
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
def _spec(ego_lane=1, ego_s=80.0, ego_v=18.0, lead_lane=1, lead_s=None, lead_v=0.0, lead_model="xl", bg=()):
    from handoff.simulator.scenario import ScenarioSpec, VehicleSpec

    lead = VehicleSpec(lane=lead_lane, s=lead_s if lead_s is not None else 990.0, speed=lead_v, model=lead_model,
                       role="lead")
    return ScenarioSpec(seed=-1, ego=VehicleSpec(lane=ego_lane, s=ego_s, speed=ego_v, role="ego"), lead=lead,
                        background=[VehicleSpec(lane=b[0], s=b[1], speed=b[2], model="default") for b in bg])


class _FixedScenario:
    def __init__(self, spec):
        self.spec = spec

    def sample(self, rng, seed=-1):
        return self.spec


def _env_with_spec(method, spec, overrides=(), run_dir=None):
    from handoff.rl.env.factory import make_env

    cfg = load_config(method, overrides=list(overrides))
    env = make_env(cfg, is_eval=True, run_dir=run_dir)
    env.core.scenario = _FixedScenario(spec)
    return env, cfg


def _all_traffic(core, role=None):
    """TrackedObjects for every traffic vehicle (no range limit)."""
    from handoff.simulator.state import make_tracked_object

    out = []
    for v in core.md.engine.traffic_manager._traffic_vehicles:
        r = core.md.traffic_roles.get(v.name, "vehicle")
        if role is None or r == role:
            x, y = v.position
            vx, vy = v.velocity
            out.append(make_tracked_object(v.name, x, y, v.heading_theta, vx, vy, v.LENGTH, v.WIDTH, r, core.road))
    return out


# ----------------------------------------------------------------------------------------------- checks
def check_env() -> dict:
    from handoff.rl.env.factory import make_env
    from handoff.simulator.metadrive_env import geometric_off_road

    cfg = load_config("m1")
    env = make_env(cfg, is_eval=True)
    core = env.core
    res = {"spawn": [], "dynamics": [], "lead_speed": []}
    ok = True
    for seed in cfg["scenario"]["seeds"]["eval"][:6]:
        env.reset(options={"episode_seed": seed})
        spec = core.spec
        md = core.md
        dyn = md.agent.get_dynamics_parameters()
        want = {"max_engine_force": 800, "max_brake_force": 150, "wheel_friction": 0.9, "max_steering": 40}
        dyn_ok = all(abs(dyn[k] - v) < 1e-6 for k, v in want.items())
        e = core.ego
        spawn_err = [abs(e.s - spec.ego.s), abs(e.d - core.road.lane_centers[spec.ego.lane]), abs(e.v - spec.ego.speed)]
        lead = _all_traffic(core, "lead")  # may spawn beyond R, so read MetaDrive directly
        lead_err = (abs(lead[0].s - spec.lead.s), abs(lead[0].d - core.road.lane_centers[spec.lead.lane]),
                    abs(lead[0].v - spec.lead.speed), lead[0].length) if lead else None
        n_bg_spawned = len(md.engine.traffic_manager._traffic_vehicles) - 1
        good = dyn_ok and max(spawn_err) < 0.05 and lead_err is not None and max(lead_err[:3]) < 0.05 \
            and abs(lead_err[3] - 5.74) < 1e-6 and n_bg_spawned == len(spec.background)
        # traffic holds its overridden speed (IDM; lead is front-most in its lane)
        for _ in range(30):
            env.step(np.array([0.0, 0.0]))
            if core._done:
                break
        lead_now = _all_traffic(core, "lead")
        if lead_now:
            res["lead_speed"].append(abs(lead_now[0].v - spec.lead.speed))
        res["spawn"].append({"seed": seed, "ego_err": [round(x, 4) for x in spawn_err],
                             "lead_err": [round(x, 4) for x in lead_err[:3]] if lead_err else None,
                             "background": f"{n_bg_spawned}/{len(spec.background)}", "dynamics": dyn, "ok": good})
        ok &= good
    lead_speed_ok = bool(res["lead_speed"]) and max(res["lead_speed"]) < kmh(2.0)
    ok &= lead_speed_ok
    res["lead_speed_max_err_ms"] = max(res["lead_speed"]) if res["lead_speed"] else None

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

    # geometric off-road check vs MetaDrive's lane surface on clear cases
    clear_in = [road.to_world(s, d) for s in (100, 400, 900) for d in road.lane_centers]
    clear_out = [road.to_world(s, d) for s in (100, 400, 900) for d in (road.d_max + 1.0, road.d_min - 1.0)]
    geo_ok = not any(geometric_off_road(road, p) for p in clear_in) and all(geometric_off_road(road, p)
                                                                            for p in clear_out)
    ok &= geo_ok
    res["geometric_off_road_ok"] = geo_ok

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
        env.reset(options={"episode_seed": 1000003})
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
    """Lane 1 -> lane 0 -> lane 2 -> lane 1 with speed changes; no traffic nearby; scripted (d*, v*) actions."""
    spec = _spec(ego_lane=1, ego_v=15.0, lead_lane=2, lead_s=995.0)
    env, cfg = _env_with_spec("m2", spec)
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
    res = {"max_cross_track_m": max_cte, "steady_cross_track_m": steady_cte,
           "max_settled_speed_err_ms": max(speed_err) if speed_err else None,
           "tolerances": {"lane_change": PID_TOL_LANE_CHANGE, "steady": PID_TOL_STEADY, "speed": PID_TOL_SPEED},
           "terminated_early": len(rows) < sum(p[2] for p in plan)}
    res["ok"] = bool(max_cte < PID_TOL_LANE_CHANGE and steady_cte < PID_TOL_STEADY and speed_err
                     and max(speed_err) < PID_TOL_SPEED and not res["terminated_early"])
    return res


def check_mpc() -> dict:
    out = {}
    # (a) pass a slow lead: lane 1 -> lane 0 while passing -> back to lane 1
    spec = _spec(ego_lane=1, ego_v=kmh(60), lead_lane=1, lead_s=130.0, lead_v=kmh(35))
    env, cfg = _env_with_spec("m3", spec)
    env.reset(options={"episode_seed": 0})
    core = env.core
    L = core.road.lane_centers
    passed, reason, min_gap = False, None, np.inf
    for _k in range(80):
        lead = _all_traffic(core, "lead")
        ahead = (lead[0].s - core.ego.s) if lead else -999
        d_tgt = L[0] if ahead > -15 else L[1]
        obs, r, te, tr, info = env.step(env.encode([40.0, 40.0], [d_tgt, d_tgt]))
        min_gap = min(min_gap, min(core.recorder.steps["min_gap"][-env.period:]))
        if lead and core.ego.s > lead[0].s + 20:
            passed = True
        if te or tr:
            reason = info["termination"]
            break
    s = core.recorder.steps
    st = np.array(s["solve_time"])
    out["pass_lead"] = {"passed": passed, "termination": reason, "min_gap": float(min_gap),
                        "final_d": core.ego.d, "solve_ms_mean": float(st.mean() * 1e3),
                        "solve_ms_p95": float(np.percentile(st, 95) * 1e3), "fallbacks": int(np.sum(s["fallback"]))}
    out["pass_lead"]["ok"] = bool(passed and reason in (None, "success") and min_gap > 0.3)
    env.close()

    # (b) stopped vehicle in the path, waypoints straight through it, safe-set projection disabled
    spec = _spec(ego_lane=1, ego_v=kmh(50), lead_lane=1, lead_s=170.0, lead_v=0.0, lead_model="default")
    env, cfg = _env_with_spec("m3", spec, overrides=["method.safe_set.enabled=false"])
    env.reset(options={"episode_seed": 0})
    core = env.core
    reason, min_gap = None, np.inf
    for _k in range(40):
        obs, r, te, tr, info = env.step(env.encode([25.0, 25.0], [L[1], L[1]]))
        min_gap = min(min_gap, min(core.recorder.steps["min_gap"][-env.period:]))
        if te or tr:
            reason = info["termination"]
            break
    stopped = _all_traffic(core, "lead")
    out["stopped_vehicle"] = {"termination": reason, "min_gap": float(min_gap), "final_s": core.ego.s,
                              "final_v": core.ego.v, "final_d": core.ego.d,
                              "stopped_vehicle_s": stopped[0].s if stopped else None}
    out["stopped_vehicle"]["ok"] = bool(reason != "collision" and min_gap > 0.2)
    env.close()

    # (c) fallback path: cripple the solver (1 iteration) -> shifted previous plan, then emergency brake
    spec = _spec(ego_lane=1, ego_v=kmh(60), lead_lane=2, lead_s=990.0)
    env, cfg = _env_with_spec("m3", spec)
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
    out["fallback"] = {"healthy_fallbacks": healthy, "modes_seen": sorted(modes), "recovered": bool(recovered),
                       "speed_after_brake": core.ego.v}
    out["fallback"]["ok"] = bool(healthy == 0 and {"shifted_previous", "emergency_brake"} <= modes and recovered)
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
