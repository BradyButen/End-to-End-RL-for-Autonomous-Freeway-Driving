#!/usr/bin/env python
"""Turn human drives (scripts/drive.py logs) into an M1 behaviour-cloning dataset.

The logs hold the ego / car states and the human's controls, but not the observation the policy sees (the BEV).
MetaDrive is deterministic for a given seed and action sequence, so each episode is REPLAYED headless in the M1 env:
same seed, the logged [steering, throttle_brake] fed back step by step, and the real observation recorded before each
action. Every episode is checked against its log (ego position at every step, termination); a mismatch is reported
and the episode skipped.

    python scripts/build_demos.py                                       # data/human_demos/*.npz -> runs/demos/m1.npz
    python scripts/build_demos.py data/human_demos/*_s318.npz --out runs/demos/test.npz

Output (.npz, compressed):
    bev      (N, 5, 84, 84) uint8   obs["bev"] * 255 (MetaDrive's BEV is 0..1; restore with / 255)
    ego      (N, 5) float32         obs["ego"]  [v, d, psi_err, steering, throttle_brake] (configs/base.yaml)
    action   (N, 2) float32         the human's [steering, throttle_brake] taken after seeing that observation
    episode  (N,) int32             index into ``files`` / ``seeds``
    step     (N,) int32             control step within the episode
    files, seeds                    source logs and their MetaDrive seeds
Pairing: sample t = (observation before control step t, action applied at control step t). The log's ego row t is
the state AFTER step t, and its steering / throttle_brake are the action applied at step t.
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np

POS_TOL = 0.05   # m; replay must reproduce the logged ego position this closely at every step. Reusing one env
                 # across episodes drifts by a few mm (a BEV pixel is 100 m / 84 = 1.2 m, so this is invisible)


def replay(env, path: str):
    from handoff.logs import load_episode

    e = load_episode(path)
    d, meta = e["data"], e["meta"]
    seed = int(meta["seed"])
    acts = np.stack([d["step_steering"], d["step_throttle_brake"]], axis=1).astype(np.float32)
    obs, _ = env.reset(options={"episode_seed": seed})
    bev, ego, xy, end = [], [], [], None
    for t, a in enumerate(acts):
        bev.append(obs["bev"])
        ego.append(obs["ego"])
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
    return {"seed": seed, "bev": np.stack(bev), "ego": np.stack(ego), "action": acts[:n], "err": err,
            "termination": meta["termination"], "problems": problems}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="*", help="drive logs (default: data/human_demos/*.npz)")
    p.add_argument("--out", default="runs/demos/m1.npz")
    p.add_argument("--reward", default="r4_straight", help="only affects the env's reward, not the dataset")
    args = p.parse_args()
    files = sorted(args.episodes or glob.glob("data/human_demos/*.npz"))
    if not files:
        raise SystemExit("no episodes found")

    from handoff.config import load_config
    from handoff.env import make_env

    env = make_env(load_config("m1", reward=args.reward), render=False, is_eval=True, run_dir=None)
    kept, seeds, cols = [], [], {"bev": [], "ego": [], "action": [], "episode": [], "step": []}
    t0 = time.time()
    try:
        for f in files:
            r = replay(env, f)
            status = "SKIP: " + "; ".join(r["problems"]) if r["problems"] else "ok"
            print(f"seed {r['seed']:>4}  {len(r['action']):4d} steps  {r['termination']:>8}  "
                  f"max pos err {r['err']:.1e} m  {status}", flush=True)
            if r["problems"]:
                continue
            i = len(kept)
            kept.append(f)
            seeds.append(r["seed"])
            bev = r["bev"]
            if not np.all((bev >= 0.0) & (bev <= 1.0)):
                raise SystemExit(f"{f}: BEV outside [0, 1]; the uint8 storage assumes MetaDrive's norm_pixel")
            cols["bev"].append(np.round(bev * 255.0).astype(np.uint8))
            cols["ego"].append(r["ego"].astype(np.float32))
            cols["action"].append(r["action"])
            cols["episode"].append(np.full(len(bev), i, np.int32))
            cols["step"].append(np.arange(len(bev), dtype=np.int32))
    finally:
        env.close()
    if not kept:
        raise SystemExit("no episode replayed cleanly")

    data = {k: np.concatenate(v) for k, v in cols.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, files=np.array([Path(f).name for f in kept]),
                        seeds=np.array(seeds, np.int32))
    a = data["action"]
    print(f"\n{len(kept)}/{len(files)} episodes, {len(a)} samples -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f} s)")
    print(f"  steering:       mean {a[:, 0].mean():+.3f}  std {a[:, 0].std():.3f}  zero {np.mean(np.abs(a[:, 0]) < 0.01):.0%}")
    print(f"  throttle_brake: mean {a[:, 1].mean():+.3f}  std {a[:, 1].std():.3f}  braking {np.mean(a[:, 1] < -0.01):.0%}")


if __name__ == "__main__":
    main()
