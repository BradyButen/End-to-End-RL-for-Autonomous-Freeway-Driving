#!/usr/bin/env python
"""Turn human drives (scripts/drive.py logs) into a behaviour-cloning dataset for M1, M2 or M3.

The logs hold the ego / car states and the human's controls, but not the observation the policy sees (the BEV).
MetaDrive is deterministic for a given seed and action sequence, so each episode is REPLAYED headless: same seed, the
logged [steering, throttle_brake] fed back step by step (an M1 env with raw throttle_brake actions,
method.accel_range null, so the logged commands reproduce the drive exactly), and the real observation and ego state
recorded before each control step. Every episode is checked against its log (ego position at every step,
termination); a mismatch is reported and the episode skipped. All methods see the same observation (BEV + ego vector;
MetaDrive builds the BEV every control step whatever the method), so one replay serves all three.

    python scripts/build_demos.py                        # M1: data/human_demos/*.npz -> runs/demos/m1.npz
    python scripts/build_demos.py --method m2            # -> runs/demos/m2.npz
    python scripts/build_demos.py --method m3            # -> runs/demos/m3.npz
    python scripts/build_demos.py data/human_demos/*_s318.npz --out runs/demos/test.npz

Labels (docs/human-data-pipeline.md, "Converting human drives into M2 / M3 actions"):
  M1  every control step: [steering, target accel / accel_range] - the logged command through the actuation table
      (+1 = full throttle = +2.9 m/s^2 in both; brakes beyond -accel_range[0] clip to -1).
  M2  every decision (0.5 s): hindsight (d*, v*) = the human's lateral offset and speed --m2-ahead-s (1.3 s) later;
      the quintic (trajectory.horizon 2 s) + PID then reproduce the human path to ~0.14 m rms with no timing
      offset (offline sweep over the 17 drives: the flown path matches when the target is read ~0.7 x horizon
      ahead; 2 s ahead with a 2 s horizon ran 0.7 s early and sideswiped the cars the human waited for).
  M3  every decision (1 s): 5 waypoints on the human's future path, spaced by the human's speed x --m3-spacing-s
      (default 2 s, clipped to method.ds_range: ~36-40 m at 77 km/h -> ~180 m / ~8 s ahead; 1 s = ~20 m spacing
      follows the human path less closely after smoothing: 90 % within 2.7 m vs 1.8 m), all slots used (k = 5); v* = the
      human's highest speed over the next --m3-speed-ahead-s (3 s). v* is M3's speed target AND hard ceiling: the
      mean over the whole ~8 s waypoint stretch slowed it ~175 m before the human; the next-second speed capped it
      at the human's speed, so it could never catch up after falling behind. Plans whose peak lateral acceleration
      exceeds mpc.max_lat_accel (the check that would end the episode as infeasible_plan) are, with --m3-infeasible:
        smooth (default)  rate-limited: each waypoint moves at most the feasible |dd| from the previous one (the
                          lateral move is spread over more distance, same direction / target lane), then re-checked
        drop              left out. ~50 % of the human plans fail the check: every plan whose 100-180 m lookahead
                          contains a keyboard lane change (~40 m long at 77 km/h), so dropping removes all lane
                          changes from the data.
  Labels are encoded with the method's own ``encode`` into its [-1, 1] action; past the end of the drive the path is
  held at its last lateral offset (as the spline extends flat).

Output (.npz, compressed):
    bev      (N, 5, 84, 84) uint8   obs["bev"] * 255 (MetaDrive's BEV is 0..1; restore with / 255)
    ego      (N, 5) float32         obs["ego"]  [v, d, psi_err, steering, throttle_brake] (configs/base.yaml)
    action   (N, A) float32         the label in the method's [-1, 1] action space
    action_raw (N, .) float32       physical label: M1 [steering, throttle_brake] as driven; M2 [d*, v*] (m, m/s);
                                    M3 [ds, d1..d5, v*] (m, m, m/s)
    flags    (N, A) bool            M2 / M3: rare "event" per action dim (lateral move > 0.5 m, slowing > 1 m/s)
                                    that scripts/bc.py up-weights (M1: computed there from steering / braking)
    state    (N, 3) float32         ego (s, d, v) at the decision
    episode  (N,) int32             index into ``files`` / ``seeds``
    step     (N,) int32             control step within the episode
    files, seeds, method
Pairing: sample t = (observation before control step t, label for the action taken at t).
"""

import argparse
import glob
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

POS_TOL = 0.05   # m; replay must reproduce the logged ego position this closely at every step. Reusing one env
                 # across episodes drifts by a few mm (a BEV pixel is 100 m / 84 = 1.2 m, so this is invisible)
LATERAL_EVENT = 0.5   # m: a label that moves the car sideways this much is a (rare) lateral manoeuvre
SLOW_EVENT = 1.0      # m/s: a label this much below the current speed is a (rare) slow-down


def to_accel_action(raw: np.ndarray, cfg: dict) -> np.ndarray:
    """[steering, throttle_brake] -> M1's [steering, accel / range] (inverse of M1.decide's mapping)."""
    from handoff.sim import ActuationMap

    a_min, a_max = (float(x) for x in cfg["method"]["accel_range"])
    amap = ActuationMap(cfg["actuation"]["table"])
    acc = np.array([amap.command_to_accel(u) for u in raw[:, 1]])
    x = np.where(acc >= 0, acc / a_max, acc / -a_min)
    return np.stack([raw[:, 0], np.clip(x, -1.0, 1.0)], axis=1).astype(np.float32)


def replay(env, path: str):
    """Replay one logged drive. ``state`` row t = ego (s, d, v) before control step t (one extra final row)."""
    from handoff.logs import load_episode

    e = load_episode(path)
    d, meta = e["data"], e["meta"]
    seed = int(meta["seed"])
    acts = np.stack([d["step_steering"], d["step_throttle_brake"]], axis=1).astype(np.float32)
    obs, _ = env.reset(options={"episode_seed": seed})
    bev, ego, xy, state, end = [], [], [], [], None
    for t, a in enumerate(acts):
        bev.append(obs["bev"])
        ego.append(obs["ego"])
        state.append((env.ego.s, env.ego.d, env.ego.v))
        obs, _, terminated, truncated, info = env.step(a)
        xy.append((env.ego.x, env.ego.y))
        if terminated or truncated:
            end = (t + 1, info.get("termination"))
            break
    state.append((env.ego.s, env.ego.d, env.ego.v))
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
    return {"seed": seed, "bev": np.stack(bev), "ego": np.stack(ego), "raw": acts[:n], "err": err,
            "state": np.asarray(state, np.float64), "termination": meta["termination"], "problems": problems}


# ------------------------------------------------------------------------------------------------ labels
def labels_m1(r, cfg, method, args):
    idx = np.arange(len(r["raw"]))
    return idx, to_accel_action(r["raw"], cfg), r["raw"], None, 0


def labels_m2(r, cfg, method, args):
    """Hindsight (d*, v*): where the human is method.trajectory.horizon after each decision."""
    st, n, P = r["state"], len(r["raw"]), method.period
    ahead = getattr(args, "m2_ahead_s", None) or 1.3
    H = int(round(ahead / float(cfg["sim"]["control_dt"])))
    v_cap = method.env.v_cap
    idx, act, phys, flags = [], [], [], []
    for t in range(0, n, P):
        k = min(t + H, n)
        d_star, v_star = float(st[k, 1]), float(np.clip(st[k, 2], 0.0, v_cap))
        idx.append(t)
        act.append(np.clip(method.encode(d_star, v_star), -1.0, 1.0))
        phys.append((d_star, v_star))
        flags.append((abs(d_star - st[t, 1]) > LATERAL_EVENT, v_star < st[t, 2] - SLOW_EVENT))
    return np.array(idx), np.array(act, np.float32), np.array(phys, np.float32), np.array(flags), 0


def smooth_waypoints(ego, s_w: np.ndarray, d_w: np.ndarray, v_star: float, method) -> np.ndarray:
    """Rate-limit the lateral offsets so the plan passes M3's feasibility check (m3.plan_lat_accel): a segment of
    length ds at speed v may move at most |dd| = limit * ds^2 / (5.77 v^2) (95 % of it, for margin)."""
    brake, lim = method.corridor.brake, 0.95 * method.lat_accel_limit
    v0 = max(ego.v, 0.0)
    floor = min(v0, v_star)
    out, s_prev, d_prev = np.empty_like(d_w), ego.s, ego.d
    for i, (si, di) in enumerate(zip(s_w, d_w)):
        v = np.sqrt(max(v0**2 - 2.0 * brake * (s_prev - ego.s), floor**2, 1e-6))
        dd_max = lim * (si - s_prev) ** 2 / (5.77 * v**2)
        out[i] = d_prev + np.clip(di - d_prev, -dd_max, dd_max)
        s_prev, d_prev = si, out[i]
    return out


def labels_m3(r, cfg, method, args):
    """5 waypoints on the human's future path at speed-scaled spacing; v* = mean speed over that stretch."""
    from handoff.m3 import plan_lat_accel

    st, n, P, ns = r["state"], len(r["raw"]), method.period, method.n_slots
    S = np.maximum.accumulate(st[:, 0])          # s never decreases (guards np.interp against tiny reversals)
    D, V = st[:, 1], st[:, 2]
    lo, hi = method.ds_range
    vlo, vhi = method.v_range
    idx, act, phys, flags, dropped, smoothed = [], [], [], [], 0, 0
    for t in range(0, n, P):
        ds = float(np.clip(V[t] * args.m3_spacing_s, lo, hi))
        s_w = S[t] + ds * np.arange(1, ns + 1)
        d_w = np.interp(s_w, S[t:], D[t:])        # held at the last offset beyond the end of the drive
        k_ahead = int(round(args.m3_speed_ahead_s / float(cfg["sim"]["control_dt"])))
        v_star = float(np.clip(np.max(V[t:min(t + k_ahead, n) + 1]), vlo, vhi))   # ceiling: human's max, next 3 s
        ego = SimpleNamespace(s=S[t], d=D[t], v=V[t])

        def feasible(d, ego=ego, s_w=s_w, v_star=v_star):
            return plan_lat_accel(ego, np.stack([s_w, d], axis=1), v_star, method.corridor.brake) \
                <= method.lat_accel_limit

        if not feasible(d_w):
            if args.m3_infeasible == "smooth":
                d_w = smooth_waypoints(ego, s_w, d_w, v_star, method)
                smoothed += 1
            if args.m3_infeasible == "drop" or not feasible(d_w):
                dropped += 1
                continue
        idx.append(t)
        act.append(method.encode([ds] * ns, d_w.tolist(), v_star))
        phys.append([ds, *d_w, v_star])
        lat = bool(np.max(np.abs(d_w - D[t])) > LATERAL_EVENT)
        slow = v_star < V[t] - SLOW_EVENT
        flags.append([False, lat] * ns + [slow, False])     # [ds, d] per slot, then v*, count
    labels_m3.smoothed = getattr(labels_m3, "smoothed", 0) + smoothed
    return np.array(idx), np.array(act, np.float32), np.array(phys, np.float32), np.array(flags), dropped


LABELS = {"m1": labels_m1, "m2": labels_m2, "m3": labels_m3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="*", help="drive logs (default: data/human_demos/*.npz)")
    p.add_argument("--method", default="m1", choices=sorted(LABELS))
    p.add_argument("--out", default=None, help="default runs/demos/<method>.npz")
    p.add_argument("--m3-spacing-s", type=float, default=2.0,
                   help="M3 waypoint spacing = human speed x this (s), clipped to method.ds_range")
    p.add_argument("--m2-ahead-s", type=float, default=1.3,
                   help="M2 d*, v* = the human's offset / speed this far ahead (s); ~0.7 x trajectory.horizon")
    p.add_argument("--m3-speed-ahead-s", type=float, default=3.0,
                   help="M3 v* = the human's highest speed over this many seconds ahead")
    p.add_argument("--m3-infeasible", choices=["smooth", "drop"], default="smooth",
                   help="M3 plans over the lateral-accel limit: rate-limit them (default) or leave them out")
    p.add_argument("--reward", default="r4_straight", help="only affects the env's reward, not the dataset")
    args = p.parse_args()
    files = sorted(args.episodes or glob.glob("data/human_demos/*.npz"))
    if not files:
        raise SystemExit("no episodes found")
    out = Path(args.out or f"runs/demos/{args.method}.npz")

    from handoff.config import load_config
    from handoff.env import make_env, method_class

    cfg = load_config(args.method, reward=args.reward)
    if args.method == "m1" and cfg["method"].get("accel_range") is None:
        raise SystemExit("configs/m1.yaml method.accel_range is null: nothing to convert the labels to")
    raw_cfg = load_config("m1", reward=args.reward, overrides=["method.accel_range=null"])   # replay as driven
    env = make_env(raw_cfg, render=False, is_eval=True, run_dir=None)
    method = method_class(args.method)(cfg, env)    # its encode / ranges / feasibility check (env: road, v_cap)
    cols = {k: [] for k in ("bev", "ego", "action", "action_raw", "flags", "state", "episode", "step")}
    kept, seeds, n_dropped, t0 = [], [], 0, time.time()
    try:
        for f in files:
            r = replay(env, f)
            status = "SKIP: " + "; ".join(r["problems"]) if r["problems"] else "ok"
            if r["problems"]:
                print(f"seed {r['seed']:>4}  {len(r['raw']):4d} steps  {r['termination']:>8}  {status}", flush=True)
                continue
            if not np.all((r["bev"] >= 0.0) & (r["bev"] <= 1.0)):
                raise SystemExit(f"{f}: BEV outside [0, 1]; the uint8 storage assumes MetaDrive's norm_pixel")
            idx, act, phys, flags, dropped = LABELS[args.method](r, cfg, method, args)
            n_dropped += dropped
            print(f"seed {r['seed']:>4}  {len(r['raw']):4d} steps -> {len(idx):4d} samples"
                  f"{f' ({dropped} infeasible dropped)' if dropped else ''}  {r['termination']:>8}  "
                  f"max pos err {r['err']:.1e} m  ok", flush=True)
            i = len(kept)
            kept.append(f)
            seeds.append(r["seed"])
            cols["bev"].append(np.round(r["bev"][idx] * 255.0).astype(np.uint8))
            cols["ego"].append(r["ego"][idx].astype(np.float32))
            cols["action"].append(act)
            cols["action_raw"].append(phys)
            cols["flags"].append(flags if flags is not None else np.zeros_like(act, dtype=bool))
            cols["state"].append(r["state"][idx].astype(np.float32))
            cols["episode"].append(np.full(len(idx), i, np.int32))
            cols["step"].append(idx.astype(np.int32))
    finally:
        env.close()
    if not kept:
        raise SystemExit("no episode replayed cleanly")

    data = {k: np.concatenate(v) for k, v in cols.items()}
    if args.method == "m1":
        data.pop("flags")                          # bc.py derives M1's weights from steering / braking
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, files=np.array([Path(f).name for f in kept]),
                        seeds=np.array(seeds, np.int32), method=np.array(args.method))
    a, ph = data["action"], data["action_raw"]
    print(f"\n{len(kept)}/{len(files)} episodes, {len(a)} samples -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f} s)")
    if args.method == "m1":
        s, x = a[:, 0], a[:, 1]
        print(f"  steering:       mean {s.mean():+.3f}  std {s.std():.3f}  zero {np.mean(np.abs(s) < 0.01):.0%}")
        print(f"  accel action:   mean {x.mean():+.3f}  std {x.std():.3f}  braking {np.mean(x < -0.01):.0%}"
              f"  (range {cfg['method']['accel_range']} m/s^2)")
    else:
        fl = data["flags"]
        v = ph[:, -1]
        print(f"  v*: mean {v.mean() * 3.6:.1f} km/h (min {v.min() * 3.6:.1f}, max {v.max() * 3.6:.1f}); "
              f"lateral-move samples {fl.any(axis=1).mean():.0%}")
        if args.method == "m2":
            print(f"  d*: mean {ph[:, 0].mean():+.2f} m, |d* - d| > {LATERAL_EVENT} m on {fl[:, 0].mean():.0%}")
        else:
            n_sm = getattr(labels_m3, "smoothed", 0)
            print(f"  spacing ds: mean {ph[:, 0].mean():.1f} m; over the lateral-accel limit: "
                  f"{n_sm if args.m3_infeasible == 'smooth' else n_dropped} of {n_dropped + len(a)} plans -> "
                  f"{args.m3_infeasible}ed; dropped {n_dropped}")


if __name__ == "__main__":
    main()
