#!/usr/bin/env python
"""Turn human drives (scripts/drive.py logs) into M1's behaviour-cloning dataset (docs/warm-start.md).

The logs hold the ego / car states and the human's controls; the policy's observation is rebuilt by REPLAYING each
episode headless: same seed, the logged [steering, throttle_brake] fed back step by step (M1 with raw commands,
method.steer_lat_accel / accel_range null, so the drive is reproduced exactly), and the observation and ego state
recorded before each control step. Every episode is checked against its log (ego position at every step,
termination); a mismatch is reported and the episode skipped. A drive only replays under the config it was driven
with (traffic, map); the spawn speed is read from the drives themselves (their first logged speed), so drives
recorded at 40 km/h still replay when the training spawn speed differs.

    python scripts/build_demos.py                        # data/human_demos/*.npz -> runs/demos/m1.npz
    python scripts/build_demos.py data/human_demos/*_s318.npz --out runs/demos/test.npz

Labels, every control step, in M1's action: [lateral accel / steer_lat_accel, target accel / accel_range]: the
logged steering through the kinematic bicycle at the speed it was applied (m1.steer_to_lat_accel), the logged
throttle_brake through the actuation table; both clipped to [-1, 1].

Output (.npz, compressed):
    obs_<key>  (N, .) float32      the observation (env.observation_space keys: ego, lanes)
    action     (N, 2) float32      the label in M1's [-1, 1] action space
    action_raw (N, 2) float32      [steering, throttle_brake] as driven
    state      (N, 3) float32      ego (s, d, v) before the step
    episode    (N,) int32          index into ``files`` / ``seeds``
    step       (N,) int32          control step within the episode
    files, seeds, method
Pairing: sample t = (observation before control step t, the action taken at t).
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np

from handoff.config import DEFAULT_REWARD

POS_TOL = 0.05   # m; the replay must reproduce the logged ego position this closely at every step


def to_m1_action(raw: np.ndarray, v: np.ndarray, cfg: dict, max_steering_rad: float) -> np.ndarray:
    """[steering, throttle_brake] driven at speed v -> M1's [lateral accel, accel] action (inverse of M1.decide)."""
    from handoff.m1 import steer_to_lat_accel
    from handoff.sim import ActuationMap

    m, ego = cfg["method"], cfg["sim"]["ego"]
    wheelbase = float(ego["front_wheelbase"]) + float(ego["rear_wheelbase"])
    lat = steer_to_lat_accel(raw[:, 0], v, wheelbase, max_steering_rad) / float(m["steer_lat_accel"])
    a_min, a_max = (float(x) for x in m["accel_range"])
    amap = ActuationMap(cfg["actuation"]["table"])
    acc = np.array([amap.command_to_accel(u) for u in raw[:, 1]])
    x = np.where(acc >= 0, acc / a_max, acc / -a_min)
    return np.clip(np.stack([lat, x], axis=1), -1.0, 1.0).astype(np.float32)


def replay(env, path: str):
    """Replay one logged drive. ``state`` row t = ego (s, d, v) before control step t."""
    from handoff.logs import load_episode

    e = load_episode(path)
    d, meta = e["data"], e["meta"]
    seed = int(meta["seed"])
    acts = np.stack([d["step_steering"], d["step_throttle_brake"]], axis=1).astype(np.float32)
    obs, _ = env.reset(options={"episode_seed": seed})
    obs_rows, xy, state, end = [], [], [], None
    for t, a in enumerate(acts):
        obs_rows.append(obs)
        state.append((env.ego.s, env.ego.d, env.ego.v))
        obs, _, terminated, truncated, info = env.step(a)
        xy.append((env.ego.x, env.ego.y))
        if terminated or truncated:
            end = (t + 1, info.get("termination"))
            break
    xy = np.asarray(xy)
    n = len(xy)
    err = float(np.max(np.hypot(xy[:, 0] - d["step_x"][:n], xy[:, 1] - d["step_y"][:n])))
    problems = []
    if n != len(acts):
        problems.append(f"replay ended after {n} of {len(acts)} steps")
    if err > POS_TOL:
        problems.append(f"position error {err:.4f} m")
    if end is not None and end[1] != meta["termination"]:
        problems.append(f"ended as {end[1]}, log says {meta['termination']}")
    obs = {k: np.stack([o[k] for o in obs_rows]) for k in obs_rows[0]}
    return {"seed": seed, "obs": obs, "raw": acts[:n], "err": err, "state": np.asarray(state, np.float64),
            "termination": meta["termination"], "problems": problems}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="*", help="drive logs (default: data/human_demos/*.npz)")
    p.add_argument("--out", default="runs/demos/m1.npz")
    p.add_argument("--reward", default=DEFAULT_REWARD, help="only affects the env's reward, not the dataset")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config the drives were recorded with, if not today's (e.g. sim.ego.spawn_speed_km_h=80)")
    args = p.parse_args()
    files = sorted(args.episodes or glob.glob("data/human_demos/*.npz"))
    if not files:
        raise SystemExit("no episodes found")
    out = Path(args.out)

    from handoff.config import load_config
    from handoff.env import make_env
    from handoff.logs import load_episode

    v0 = sorted({int(round(float(load_episode(f)["data"]["step_v"][0]) * 3.6 / 5.0) * 5) for f in files})
    if len(v0) != 1:
        raise SystemExit(f"drives start at different speeds {v0} km/h: build them separately")
    spawn = [f"sim.ego.spawn_speed_km_h={v0[0]}"]                     # replay at the speed they were driven at
    print(f"replaying at the drives' spawn speed: {v0[0]} km/h")
    cfg = load_config("m1", reward=args.reward, overrides=[*spawn, *args.overrides])
    raw_cfg = load_config("m1", reward=args.reward, overrides=[
        *spawn, *args.overrides, "method.accel_range=null", "method.steer_lat_accel=null"])   # replay as driven
    env = make_env(raw_cfg, render=False, is_eval=True, run_dir=None)
    cols: dict[str, list] = {k: [] for k in ("action", "action_raw", "state", "episode", "step")}
    kept, seeds, t0 = [], [], time.time()
    try:
        for f in files:
            r = replay(env, f)
            if r["problems"]:
                print(f"seed {r['seed']:>4}  {len(r['raw']):4d} steps  {r['termination']:>8}  SKIP: "
                      + "; ".join(r["problems"]), flush=True)
                continue
            print(f"seed {r['seed']:>4}  {len(r['raw']):4d} samples  {r['termination']:>8}  "
                  f"max pos err {r['err']:.1e} m  ok", flush=True)
            i = len(kept)
            kept.append(f)
            seeds.append(r["seed"])
            for k, v in r["obs"].items():
                cols.setdefault(f"obs_{k}", []).append(v.astype(np.float32))
            cols["action"].append(to_m1_action(r["raw"], r["state"][:, 2], cfg, env.max_steering_rad))
            cols["action_raw"].append(r["raw"])
            cols["state"].append(r["state"].astype(np.float32))
            cols["episode"].append(np.full(len(r["raw"]), i, np.int32))
            cols["step"].append(np.arange(len(r["raw"]), dtype=np.int32))
    finally:
        env.close()
    if not kept:
        raise SystemExit("no episode replayed cleanly")

    data = {k: np.concatenate(v) for k, v in cols.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, files=np.array([Path(f).name for f in kept]),
                        seeds=np.array(seeds, np.int32), method=np.array("m1"))
    a = data["action"]
    print(f"\n{len(kept)}/{len(files)} episodes, {len(a)} samples -> {out} ({time.time() - t0:.0f} s)")
    print(f"  lateral action: mean {a[:, 0].mean():+.3f}  std {a[:, 0].std():.3f}  "
          f"clipped {np.mean(np.abs(a[:, 0]) >= 1.0):.1%}  (+-1 = {cfg['method']['steer_lat_accel']} m/s^2)")
    print(f"  accel action:   mean {a[:, 1].mean():+.3f}  std {a[:, 1].std():.3f}  "
          f"braking {np.mean(a[:, 1] < -0.01):.1%}  (range {cfg['method']['accel_range']} m/s^2)")


if __name__ == "__main__":
    main()
