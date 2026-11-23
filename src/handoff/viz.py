"""Seeing what the car does: the MetaDrive viewer with overlays (live, replay, validation) and the plots.

Viewer overlays (keys: M = overlays on / off, X = end the episode, K = M3 corridor constraints on / off):
  all  white line     ego trail (last 20 s)
  M2   yellow line    quintic reference          white ball  its end          magenta ball  PID tracking point
  M3   yellow line    lane plan (validation: waypoint spline)   cyan line   MPC predicted path
       blue lines     corridor over the MPC horizon (left / right bound per stage; cross lines = car faces)
       magenta boxes  car keep-out zones         orange line gate check (ego -> next plan point; red while it fires)
       white balls    plan points (validation: the placed waypoints)
Every viewed / validated episode writes episodes/<id>.npz, <id>_overview.png and <id>_steps.csv.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

WHITE = (1.0, 1.0, 1.0, 1.0)
YELLOW, CYAN = (1.0, 0.85, 0.1, 1.0), (0.1, 0.9, 1.0, 1.0)
BLUE, MAGENTA = (0.3, 0.5, 1.0, 1.0), (1.0, 0.3, 1.0, 1.0)
ORANGE, RED, TRAIL = (1.0, 0.6, 0.2, 1.0), (1.0, 0.15, 0.15, 1.0), (0.85, 0.85, 0.85, 1.0)
Z = 0.6   # draw height above the road (m)


# ======================================================================================= overlay geometry
def box_xy(road, s_lo, s_hi, d_lo, d_hi) -> np.ndarray:
    x, y = road.to_world(np.array([s_lo, s_hi, s_hi, s_lo, s_lo]), np.array([d_lo, d_lo, d_hi, d_hi, d_lo]))
    return np.stack([x, y], axis=1)


def corridor_lines(tube, road) -> list[np.ndarray]:
    """The corridor the MPC solves with: per stage the left / right bound on the ego centre at that stage's point,
    joined into two ribbons; car-made front / back faces as cross lines."""
    big = 5.0e3
    left, right, out, cuts = [], [], [], set()
    for k in range(len(tube.traj)):
        sk, dk = float(tube.traj[k, 0]), float(tube.traj[k, 1])
        back, front, lo, hi = (float(x) for x in tube.extents[k])
        left.append(road.to_world(sk, dk + hi))
        right.append(road.to_world(sk, dk + lo))
        for ext in (back, front):
            key = round(sk + ext, 1)
            if abs(ext) < big and key not in cuts:
                cuts.add(key)
                x, y = road.to_world(np.full(2, sk + ext), np.array([dk + lo, dk + hi]))
                out.append(np.stack([x, y], axis=1))
    return out + [np.array(left), np.array(right)]


def lead_car(ego, objects):
    """Nearest car ahead overlapping the ego laterally: (bumper gap, closing speed, car) or None."""
    best = None
    for o in objects:
        if o.s > ego.s and abs(o.d - ego.d) < 0.5 * (o.width + ego.width) + 0.3:
            gap = o.s - ego.s - 0.5 * (o.length + ego.length)
            if best is None or gap < best[0]:
                best = (gap, ego.v - o.v, o)
    return best


def overlay(env, diag: dict) -> tuple[list, list, dict]:
    """(lines [(xy, colour)], points [((x, y), colour)], text panel) for the current control step."""
    ego, road, objs = env.ego, env.road, env.objects
    name, method, dec = env.cfg["method"]["name"], env.method, env.decision
    ref = method.reference
    text = {"method": f"{name}  seed {env.episode_seed}", "t": f"{ego.t:5.1f} s", "speed": f"{ego.v * 3.6:5.1f} km/h",
            "s / d": f"{ego.s:6.1f} / {ego.d:5.2f} m"}
    lines, points = [], []
    if name in ("m2", "m3") and dec:
        cmd = int(dec["vector"][0])
        text["RL lane command"] = (("keep", "LEFT", "RIGHT")[cmd]
                                   + ("  (ignored: changing lanes)" if cmd and method.lane.changing(ego, road) else ""))
        sc = dec.get("speed_cmd")
        text["target lane / v*"] = (f"{dec.get('lane', '-')} (d {dec['d_target']:+.1f} m) / "
                                    f"{dec['v_target'] * 3.6:4.0f} km/h"
                                    + ("" if sc is None else f"  ({('hold', '+10', '-10')[sc]})"))
    if name == "m3" and dec:
        gated = bool(diag.get("gated_brake"))
        text["plan lateral accel"] = (f"{dec['lat_accel']:4.1f} m/s2" + ("" if dec["feasible"] else
                                      f"  INFEASIBLE (> {method.lat_accel_limit:g})"))
        text["MPC"] = f"{str(diag.get('status', ''))[:22]}  {1000 * float(diag.get('solve_time', 0) or 0):4.0f} ms"
        if diag.get("fallback"):
            text["FALLBACK"] = f"{diag.get('fallback_reason', '')} / {diag.get('fallback_mode', '')}"
        text["EMERGENCY BRAKE"] = f"gate: car {diag.get('blocker')} in the path" if gated else "-"
        if not method.mpc.constraints_on:
            text["corridor constraints"] = "OFF (K)"
        m = (ref.s >= ego.s - 5) & (ref.s <= ego.s + 150)
        lines.append((np.stack([ref.x[m], ref.y[m]], axis=1), YELLOW))
        if diag.get("pred_xy") is not None:
            lines.append((diag["pred_xy"], CYAN))
        if method.tube is not None and method.mpc.constraints_on:
            lines += [(seg, BLUE) for seg in corridor_lines(method.tube, road)]
        for o in objs:
            zl, zw = method.corridor.keepout_half(o)
            lines.append((box_xy(road, o.s - zl, o.s + zl, o.d - zw, o.d + zw), MAGENTA))
        ahead = method.waypoints[method.waypoints[:, 0] > ego.s]
        if len(ahead):
            lines.append((np.array([[ego.x, ego.y], road.to_world(*map(float, ahead[0]))]), RED if gated else ORANGE))
        points += [(road.to_world(float(s), float(d)), WHITE) for s, d in dec["waypoints"]]
    elif name == "m2" and dec:
        text["ref d / v"] = f"{diag.get('ref_d', 0):5.2f} m / {diag.get('ref_v', 0) * 3.6:5.1f} km/h"
        text["cross-track / heading err"] = (f"{diag.get('cross_track', 0):+5.2f} m / "
                                             f"{np.rad2deg(diag.get('heading_error', 0)):+5.1f} deg")
        text["speed err"] = f"{diag.get('speed_error', 0) * 3.6:+5.1f} km/h"
        lines.append((np.stack([ref.x, ref.y], axis=1), YELLOW))
        points.append(((ref.x[-1], ref.y[-1]), WHITE))
        points.append((road.to_world(float(ego.s), float(ref.at_s(ego.s)["d"])), MAGENTA))
    text["steering / throttle"] = f"{ego.steering:+5.2f} / {ego.throttle_brake:+5.2f}"
    lead = lead_car(ego, objs)
    if lead:
        text["lead car gap / closing"] = f"{lead[0]:5.1f} m / {lead[1] * 3.6:+5.1f} km/h"
    for n, w in env.reward.weights.items():
        if w != 0 and env.recorder.terms[n]:
            text[f"r.{n}"] = f"{w * env.recorder.terms[n][-1]:+.3f}"
    return lines, points, text


# ======================================================================================= viewer
class Viewer:
    """Draws the overlays into MetaDrive's 3D window after every control step (set as ``env.on_step``)."""

    def __init__(self, fps: float = 10.0, extra_points=(), allow_constraint_toggle: bool = False,
                 show_return: bool = False):
        self.fps = fps                        # 10 = real time, 0 = as fast as it renders
        self.show_return = show_return        # text panel: return so far + each reward term's running total
        self.extra_points = list(extra_points)  # world (x, y) drawn white (validation: every placed waypoint)
        self.extra_text: dict = {}
        self.allow_toggle = allow_constraint_toggle
        self.show, self.finish = True, False
        self.trail: list = []
        self._lines = self._points = None
        self._t_last = time.time()

    def new_episode(self) -> None:
        self.trail, self.finish = [], False

    def _setup(self, env) -> None:
        eng = env.md.engine
        self._lines = eng.make_line_drawer(thickness=3.0)
        self._lines.setShaderOff(1)   # line geometry has no normals: draw unlit or it is invisible
        self._lines.setLightOff(1)
        self._points = eng.make_point_drawer(scale=4.0)
        eng.accept("x", lambda: setattr(self, "finish", True))
        eng.accept("m", lambda: setattr(self, "show", not self.show))
        if self.allow_toggle and hasattr(env.method, "mpc"):
            def toggle():
                mpc = env.method.mpc
                mpc.constraints_on = not mpc.constraints_on
                mpc._prev_lam = None   # the old multipliers belong to the other constraint set
                print(f"corridor constraints {'ON' if mpc.constraints_on else 'OFF'}")
            eng.accept("k", toggle)

    def __call__(self, env, diag: dict) -> None:
        if self._lines is None:
            self._setup(env)
        self.trail = (self.trail + [(env.ego.x, env.ego.y)])[-200:]
        self._lines.reset()
        self._points.reset()
        if self.show:
            lines, points, text = overlay(env, diag)
            lines.insert(0, (np.array(self.trail), TRAIL))
            points += [(p, WHITE) for p in self.extra_points]
            self._draw(lines, points)
            env.md.render(text={**self._return_text(env), **self.extra_text, **text,
                                "keys": "M overlays, X end episode"
                                + (", K constraints" if self.allow_toggle else "")})
        else:
            env.md.render()
        if self.fps > 0:
            wait = 1.0 / self.fps - (time.time() - self._t_last)
            if wait > 0:
                time.sleep(wait)
        self._t_last = time.time()

    def _return_text(self, env) -> dict:
        if not self.show_return:
            return {}
        sums = env.recorder.weighted_sums
        return {"RETURN so far": f"{sum(sums.values()):+8.1f}",
                **{f"  {n}": f"{v:+8.1f}" for n, v in sums.items() if env.reward.weights[n] != 0}}

    def _draw(self, lines, points) -> None:
        ls, cs = [], []
        for xy, col in lines:
            xy = np.asarray(xy, float).reshape(-1, 2)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) >= 2:
                ls.append([(float(x), float(y), Z) for x, y in xy])
                cs.append([col] * (len(xy) - 1))
        if ls:
            self._lines.draw_lines(ls, cs)
        pts = [((float(x), float(y), Z + 0.3), col) for (x, y), col in points if np.isfinite(x) and np.isfinite(y)]
        if pts:
            self._points.draw_points([p for p, _ in pts], [c for _, c in pts])


def drive_episode(env, act: Callable, seed: int, tag: str, viewer: Viewer | None = None,
                  stop: Callable | None = None) -> dict:
    """One episode: ``act(obs) -> action`` every decision; ends on termination, the viewer's X key, or
    ``stop() -> reason | None``. Returns the episode summary."""
    if viewer is not None:
        viewer.new_episode()
        env.on_step = viewer
    obs, _ = env.reset(options={"episode_seed": int(seed), "tag": tag})
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        if term or trunc:
            return info["episode_summary"]
        reason = "user_finish" if viewer is not None and viewer.finish else (stop() if stop else None)
        if reason:
            return env.end_episode(reason)


# ======================================================================================= live / replay
def live(cfg: dict, checkpoint, seeds: list[int], out_dir: Path, render: bool = True, fps: float = 10.0,
         deterministic: bool = True) -> list[dict]:
    """Run a policy (checkpoint, or B0 with None) on ``seeds`` with the viewer; a report per episode."""
    from handoff.env import make_env
    from handoff.evaluate import load_policy_fn

    policy = load_policy_fn(checkpoint, deterministic, cfg)      # checked before the window opens
    env = make_env(cfg, render=render, is_eval=True, run_dir=out_dir)
    act = policy or (lambda obs: np.zeros(env.action_space.shape, np.float32))
    viewer = Viewer(fps=fps) if render else None
    summaries = []
    try:
        for seed in seeds:
            summ = drive_episode(env, act, seed, "view", viewer)
            if summ["log_file"]:
                episode_report(summ, out_dir, env.road)
            summaries.append(summ)
    finally:
        env.close()
    return summaries


def replay(cfg: dict, episode_npz: str, checkpoint, tolerance: float = 0.5) -> dict:
    """Re-simulate a logged episode (seed + frozen checkpoint; bit-exact thanks to force_destroy) with the viewer,
    checking the ego pose against the log."""
    from handoff.env import make_env
    from handoff.evaluate import load_policy_fn
    from handoff.logs import load_episode

    ep = load_episode(episode_npz)
    xs, ys = ep["data"]["step_x"], ep["data"]["step_y"]
    policy = load_policy_fn(checkpoint, cfg=cfg)
    env = make_env(cfg, render=True, is_eval=True)
    act = policy or (lambda obs: np.zeros(env.action_space.shape, np.float32))
    viewer = Viewer()
    state = {"i": 0, "max_dev": 0.0}

    def hook(env_, diag):
        i = state["i"]
        if i < len(xs):
            dev = float(np.hypot(env_.ego.x - xs[i], env_.ego.y - ys[i]))
            state["max_dev"] = max(state["max_dev"], dev)
            viewer.extra_text = {"replay step": f"{i} / {len(xs)}", "deviation from log": f"{dev:.3f} m"}
        state["i"] += 1
        viewer(env_, diag)

    try:
        viewer.new_episode()
        env.on_step = hook
        obs, _ = env.reset(options={"episode_seed": int(ep["meta"]["seed"]), "tag": "replay"})
        while True:
            obs, _, term, trunc, _ = env.step(act(obs))
            if term or trunc or viewer.finish:
                break
    finally:
        env.close()
    return {"max_deviation": state["max_dev"], "steps": state["i"], "logged_steps": len(xs),
            "diverged": state["max_dev"] > tolerance}


# ======================================================================================= plots
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _lanes(ax, road) -> None:
    ax.axhspan(road.d_min, road.d_max, color="0.94", zorder=0)
    ax.axhline(road.d_max, color="goldenrod", lw=2, zorder=1)    # centre line (solid)
    ax.axhline(road.d_min, color="0.3", lw=2, zorder=1)          # right road edge
    for c in road.lane_centers[:-1]:
        ax.axhline(c - road.lane_width / 2, color="0.6", lw=0.8, ls="--", zorder=1)


class _Lanes:
    """Stand-in road for plots made from a log alone (lane 0 on the reference line, lanes to the right)."""

    def __init__(self, lane_num: int = 3, lane_width: float = 3.5):
        self.lane_centers = [-i * lane_width for i in range(lane_num)]
        self.lane_width, self.d_max = lane_width, lane_width / 2
        self.d_min = self.lane_centers[-1] - lane_width / 2


def episode_report(summary_or_npz, out_dir: str | Path, road=None, placed=None, cars=None,
                   solver_limit_s: float | None = None) -> Path:
    """<id>_overview.png + <id>_steps.csv from an episode log (an episode summary with ``log_file``, or a .npz path).
    ``placed`` / ``cars``: validation waypoints and scripted cars (s, d), drawn on the road map."""
    from handoff.logs import load_episode

    npz = summary_or_npz["log_file"] if isinstance(summary_or_npz, dict) else summary_or_npz
    ep = load_episode(npz)
    data, meta = ep["data"], ep["meta"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(npz).stem
    road = road or _Lanes()

    cols = {k[5:]: data[k] for k in data if k.startswith("step_")}
    cols.update({f"r_{n}": data[f"term_{n}"] for n in data["term_names"]})
    with open(out_dir / f"{stem}_steps.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for row in zip(*cols.values()):
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")

    method = meta.get("method", "")
    t, s, d, v = data["step_t"], data["step_s"], data["step_d"], data["step_v"]
    dec_idx = np.clip(data["step_decision_index"].astype(int), 0, max(len(data["dec_k"]) - 1, 0))
    decoded = data["dec_decoded"] if data["dec_decoded"].ndim == 2 else np.zeros((len(data["dec_k"]), 2))
    v_star = d_star = cmd = np.full_like(t, np.nan)
    if method in ("m2", "m3") and len(decoded) and decoded.shape[1] == 4:   # [command, target lane, d*, v*]
        cmd, d_star, v_star = decoded[dec_idx, 0], decoded[dec_idx, 2], decoded[dec_idx, 3]
    bad = (data["step_gated_brake"] > 0) | (data["step_fallback"] > 0)

    plt = _plt()
    n_rows = 4 if method in ("m1", "b0", "manual") else 6
    fig, ax = plt.subplots(n_rows, 1, figsize=(14, 3 + 2.3 * n_rows),
                           gridspec_kw={"height_ratios": [2.4] + [1] * (n_rows - 1)})
    a = ax[0]                                                    # road map in (s, d)
    _lanes(a, road)
    obj_s, obj_d = data["objects"][..., 6], data["objects"][..., 7]
    for oid in {i for row in data["object_ids"] for i in row if i}:
        m = data["object_ids"] == oid
        a.plot(obj_s[m], obj_d[m], ".", color="tab:blue", ms=1.5, alpha=0.5)
    a.plot([], [], ".", color="tab:blue", label="cars (while within 50 m)")
    for cs, cd in cars or []:
        a.add_patch(plt.Rectangle((cs - 2.26, cd - 0.93), 4.515, 1.852, color="tab:blue", zorder=3))
    if placed is not None and len(placed):
        a.plot(np.asarray(placed)[:, 0], np.asarray(placed)[:, 1], "o", color="crimson", label="placed waypoints")
    if method in ("m2", "m3"):
        a.plot(s, data["step_ref_d"], "--", color="goldenrod", lw=1.2, label="reference d at ego s")
    if method in ("m2", "m3"):
        a.step(s, d_star, where="post", color="crimson", lw=1, label="RL target lane")
    a.plot(s, d, color="k", lw=1.6, label="ego")
    if bad.any():
        a.plot(s[bad], d[bad], "x", color="red", label="gate / fallback")
    a.set_xlim(s[0] - 5, s[-1] + 40)
    a.set_ylabel("d (m)")
    a.legend(loc="upper right", fontsize=8, ncol=3)
    a.set_title(f"{method} seed {meta.get('seed')}: {meta.get('termination')}, {t[-1]:.1f} s, "
                f"mean {np.mean(v) * 3.6:.1f} km/h, return {float(meta.get('return', 'nan')):.1f}  (x axis: s)")
    a = ax[1]
    a.plot(t, v * 3.6, "k", label="ego v")
    if method in ("m2", "m3"):
        a.plot(t, v_star * 3.6, color="crimson", lw=1, label="RL v*")
        a.plot(t, data["step_ref_v"] * 3.6, "--", color="goldenrod", lw=1, label="controller ref v")
    a.set_ylabel("km/h")
    a.legend(fontsize=8, ncol=3)
    a = ax[2]
    a.plot(t, d, "k", label="ego d")
    if method in ("m2", "m3"):
        a.plot(t, data["step_ref_d"], "--", color="goldenrod", label="ref d")
    if method in ("m2", "m3"):
        a.step(t, d_star, where="post", color="crimson", lw=1, label="RL target lane")
    a.set_ylabel("d (m)")
    a.legend(fontsize=8, ncol=3)
    a = ax[3]
    a.plot(t, data["step_steering"], label="steering")
    a.plot(t, data["step_throttle_brake"], label="throttle_brake")
    a.set_ylabel("command")
    a.legend(fontsize=8, ncol=2)
    if n_rows == 6:
        a = ax[4]
        a.plot(t, np.minimum(data["step_min_gap"], 50), color="tab:blue", label="min gap to any car (m)")
        a.set_ylabel("gap (m)")
        a.legend(fontsize=8)
        a = ax[5]
        if method == "m3":
            a.plot(t, 1000 * data["step_solve_time"], label="MPC solve (ms)")
            a.plot(t, 1000 * data["step_cpu_time"], lw=0.8, label="cpu (ms)")
            if solver_limit_s:
                a.axhline(1000 * solver_limit_s, color="tab:red", ls=":", label="max_cpu_time")
            a.set_ylabel("ms")
            a2 = a.twinx()
            a2.step(t, cmd, where="post", color="tab:green", lw=1, label="RL command (0 keep, 1 left, 2 right)")
            a2.set_ylabel("command")
            a2.legend(fontsize=8, loc="upper right")
        else:
            a.plot(t, data["step_tracking_error"], label="|cross-track| (m)")
            a.set_ylabel("PID error")
        a.legend(fontsize=8, loc="upper left")
    for a in ax[1:]:
        a.set_xlim(t[0], t[-1])
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("t (s)")
    fig.tight_layout()
    path = out_dir / f"{stem}_overview.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    if isinstance(summary_or_npz, dict):
        (out_dir / f"{stem}_summary.json").write_text(json.dumps(
            {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in summary_or_npz.items()}, indent=1))
    return path


def reward_weights(reward_name: str) -> dict:
    """Weights of configs/<reward_name>.yaml."""
    from handoff.config import CONFIG_DIR, load_yaml

    return {k: float(v) for k, v in load_yaml(CONFIG_DIR / f"{reward_name}.yaml")["reward"]["weights"].items()}


def reward_report(npz, out_dir: str | Path | None = None, weights: dict | None = None,
                  label: str | None = None) -> tuple[Path, dict]:
    """How the return built up over an episode: <id>_reward.png (cumulative return and each term's share, the
    reward rate, speed and lane) -> (path, {term: total}).

    ``weights`` re-scores the episode with another reward (e.g. ``reward_weights("r5")``): the reward is linear in
    the logged raw terms, so this is exact for any weights over the same terms. Default: the weights it ran with."""
    from handoff.logs import load_episode
    from handoff.reward import TERMINAL

    ep = load_episode(npz)
    data, meta = ep["data"], ep["meta"]
    names = [str(n) for n in data["term_names"]]
    if weights is None:
        logged = data.get("term_weights")
        if logged is not None:
            weights = dict(zip(names, logged))
        else:   # older logs: weight = (weighted return of the term) / (its raw sum), from the episode summary
            weights = {n: float(meta[f"return_{n}"]) / float(meta[f"raw_{n}"]) if float(meta[f"raw_{n}"]) else 0.0
                       for n in names}
        rescored = False
    else:
        missing = [n for n, w in weights.items() if w != 0 and n not in names]
        if missing:
            raise ValueError(f"terms {missing} were not logged in {npz}; drive again with the new reward")
        rescored = True
    t = data["step_t"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.1
    contrib = {n: float(weights.get(n, 0.0)) * data[f"term_{n}"] for n in names}
    active = [n for n in names if np.any(contrib[n] != 0)]
    total = sum(contrib.values())
    totals = {n: float(np.sum(contrib[n])) for n in names}
    totals["return"] = float(np.sum(total))

    plt = _plt()
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1]})
    a = ax[0]
    a.plot(t, np.cumsum(total), "k", lw=2.2, label=f"return  {totals['return']:+.1f}")
    for n in active:
        a.plot(t, np.cumsum(contrib[n]), lw=1.3, label=f"{n}  {totals[n]:+.1f}")
    a.axhline(0, color="0.5", lw=0.8)
    a.set_ylabel("cumulative reward")
    a.legend(loc="best", fontsize=9)
    a.grid(alpha=0.3)
    a = ax[1]                       # reward per second (1 s trailing window), terminal terms shown separately
    shaping = sum(contrib[n] for n in names if n not in TERMINAL)
    n_win = max(int(round(1.0 / dt)), 1)
    a.plot(t, np.convolve(shaping, np.ones(n_win))[: len(t)] / (n_win * dt), color="tab:blue",
           label="reward per second (non-terminal terms, 1 s window)")
    for n in TERMINAL:
        if n in contrib:
            for i in np.flatnonzero(contrib[n]):
                a.annotate(f"{n} {contrib[n][i]:+.0f}", (t[i], 0), xytext=(-60, 15), textcoords="offset points",
                           color="tab:red", fontsize=9, arrowprops=dict(arrowstyle="->", color="tab:red"))
    a.axhline(0, color="0.5", lw=0.8)
    a.set_ylabel("reward / s")
    a.legend(loc="best", fontsize=9)
    a.grid(alpha=0.3)
    a = ax[2]
    a.plot(t, data["step_v"] * 3.6, "k", label="speed (km/h)")
    a.set_ylabel("km/h")
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.step(t, data["step_lane_index"], where="post", color="tab:green", lw=1, label="lane (0 = left)")
    a2.set_ylabel("lane")
    a2.invert_yaxis()
    a.legend(loc="upper left", fontsize=9)
    a2.legend(loc="upper right", fontsize=9)
    ax[-1].set_xlabel("t (s)")
    what = label or meta.get("method", "")
    ax[0].set_title(f"{what} seed {meta.get('seed')}: {meta.get('termination')}, {t[-1]:.1f} s, "
                    f"return {totals['return']:+.1f}" + ("  (re-scored)" if rescored else ""))
    fig.tight_layout()
    out_dir = Path(out_dir) if out_dir else Path(npz).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(npz).stem}_reward{'_rescored' if rescored else ''}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, totals


def learning_curves(run_dirs, metric: str = "success", x: str = "control_steps", out: str | Path = "curves.png"):
    """Mean +- std over seeds of an eval metric vs training progress, one line per (method, reward)."""
    import pandas as pd

    frames = []
    for rd in run_dirs:
        f = Path(rd) / "episodes.csv"
        if f.exists():
            df = pd.read_csv(f)
            frames.append(df[df["is_eval"] == 1].assign(run_dir=str(rd)))
    if not frames:
        raise ValueError("no evaluation rows found")
    df = pd.concat(frames, ignore_index=True)
    xcol = f"train_{x}"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, variant), g in df.groupby(["method", "reward_variant"]):
        piv = g.groupby(["run_dir", xcol])[metric].mean().reset_index().pivot(index=xcol, columns="run_dir",
                                                                             values=metric).sort_index()
        mean, std = piv.mean(axis=1), piv.std(axis=1).fillna(0)
        ax.plot(piv.index, mean, label=f"{method}/{variant} (n={piv.shape[1]})")
        ax.fill_between(piv.index, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def comparison_table(csv_files, metrics=None):
    """Mean +- std of final-evaluation metrics per (method, reward)."""
    import pandas as pd

    metrics = metrics or ["success", "collision", "emergency_brake", "infeasible_count", "off_road", "timeout",
                          "time_to_complete",
                          "mean_speed", "min_gap", "min_ttc", "rms_long_jerk", "rms_lat_accel", "rms_steer_rate",
                          "tracking_error_rms", "solve_time_mean", "fallback_count", "lane_changes"]
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    g = df.groupby(["method", "reward_variant"])[[m for m in metrics if m in df.columns]]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
