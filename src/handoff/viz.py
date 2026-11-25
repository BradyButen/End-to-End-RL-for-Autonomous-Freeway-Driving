"""Seeing what the car does: the MetaDrive 3D viewer and its overlays (live, replay, human drives).

Viewer keys:
  Space / P  pause / resume          N  play 0.2 s, then pause        X  end the episode
  M  all overlays + panel on / off   I  text panel on / off           D  move the panel (cycles the 4 corners)
  1-6  one overlay layer on / off: 1 trail, 2 reference / lane plan, 3 MPC prediction, 4 corridor,
       5 keep-out boxes, 6 points (balls)
  O  corridor display style (cycles): all stages / first stage per face / every 1 s / fade over horizon / box at 1 s
  C  screenshot (MetaDrive: main_index_<seed>_step_<n>_<time>.png in the working folder)
  B / Q  top-down / chase camera (MetaDrive); mouse wheel zooms.
  Avoid R, [ and ] (MetaDrive resets / changes the map behind this env's back).
Viewer overlays:
  all  white line     ego trail (last 20 s)
  M2   yellow line    quintic reference          white ball  its end          magenta ball  PID tracking point
  M3   yellow line    lane plan                  cyan line   MPC predicted path
       blue lines     corridor over the MPC horizon (left / right bound per stage; cross lines = car faces, one per
                      stage as the car moves: O switches to a less busy style)
       magenta boxes  car keep-out zones
Every live episode writes its full log to <out>/episodes/<id>.npz.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

WHITE = (1.0, 1.0, 1.0, 1.0)
YELLOW, CYAN = (1.0, 0.85, 0.1, 1.0), (0.1, 0.9, 1.0, 1.0)
BLUE, MAGENTA = (0.3, 0.5, 1.0, 1.0), (1.0, 0.3, 1.0, 1.0)
TRAIL = (0.85, 0.85, 0.85, 1.0)
Z = 0.6   # draw height above the road (m)


# ======================================================================================= overlay geometry
def box_xy(road, s_lo, s_hi, d_lo, d_hi) -> np.ndarray:
    x, y = road.to_world(np.array([s_lo, s_hi, s_hi, s_lo, s_lo]), np.array([d_lo, d_lo, d_hi, d_hi, d_lo]))
    return np.stack([x, y], axis=1)


# Corridor display modes (O cycles them; the MPC is unaffected). Each car face moves with its car, so "all" draws
# one cross line per stage (up to N per car, ~v * dt apart).
CORRIDOR_MODES = ("all stages", "first stage per face", "every 1 s", "fade over horizon", "box at 1 s")
FACE_RUN_GAP = 3.0     # m: a face this close to the previous stage's face of the same kind is the same moving face
STRIDE_S, BOX_S, BOX_HALF_LEN = 1.0, 1.0, 15.0   # "every 1 s" spacing, "box at 1 s" stage, box length cap (m)
FADE_TO = (0.55, 0.55, 0.58, 1.0)                # road-ish grey that "fade" blends the far stages into


def _corridor_stages(tube):
    """Per stage: (s, d, back, front, lo, hi) with lo / hi absolute d; back / front None when there is no face."""
    big = 5.0e3
    out = []
    for k in range(len(tube.traj)):
        sk, dk = float(tube.traj[k, 0]), float(tube.traj[k, 1])
        back, front, lo, hi = (float(x) for x in tube.extents[k])
        out.append((sk, dk, sk + back if abs(back) < big else None, sk + front if abs(front) < big else None,
                    dk + lo, dk + hi))
    return out


def corridor_lines(tube, road, mode: str = CORRIDOR_MODES[0], dt: float = 0.1) -> list[tuple[np.ndarray, tuple]]:
    """The corridor the MPC solves with, as [(xy, colour)]: per stage the left / right bound on the ego centre at
    that stage's point (two ribbons) and the car-made front / back faces as cross lines. ``mode``: CORRIDOR_MODES."""
    st = _corridor_stages(tube)
    n = len(st)

    def cross(s_face, lo, hi):
        x, y = road.to_world(np.full(2, s_face), np.array([lo, hi]))
        return np.stack([x, y], axis=1)

    def fade(k):
        a = k / max(n - 1, 1)
        return tuple((1 - a) * c + a * f for c, f in zip(BLUE, FADE_TO))

    if mode == "box at 1 s":
        sk, dk, back, front, lo, hi = st[min(int(round(BOX_S / dt)), n - 1)]
        s0 = back if back is not None else sk - BOX_HALF_LEN
        s1 = front if front is not None else sk + BOX_HALF_LEN
        out = [(np.array([road.to_world(s0, d), road.to_world(s1, d)]), BLUE) for d in (lo, hi)]
        return out + [(cross(f, lo, hi), BLUE) for f in (back, front) if f is not None]

    left = [road.to_world(sk, hi) for sk, _, _, _, _, hi in st]
    right = [road.to_world(sk, lo) for sk, _, _, _, lo, _ in st]
    if mode == "fade over horizon":
        out = [(np.array(r[k:k + 2]), fade(k)) for r in (left, right) for k in range(n - 1)]
    else:
        out = [(np.array(left), BLUE), (np.array(right), BLUE)]
    stride = max(int(round(STRIDE_S / dt)), 1)
    seen = set()
    for j, kind in ((2, "back"), (3, "front")):     # j: index of the face in a stage tuple
        start, prev = 0, None
        for k, stage in enumerate(st):
            f = stage[j]
            if f is None:
                prev = None
                continue
            if prev is None or abs(f - prev) > FACE_RUN_GAP:
                start = k                           # a new face (a different car, or one that just appeared)
            prev = f
            if mode == "first stage per face" and k != start:
                continue
            if mode == "every 1 s" and (k - start) % stride:
                continue
            key = (kind, round(f, 1))
            if key in seen:                         # a parked car: the same line at every stage
                continue
            seen.add(key)
            out.append((cross(f, stage[4], stage[5]), fade(k) if mode == "fade over horizon" else BLUE))
    return out


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
    """(lines [(xy, colour, layer)], points [((x, y), colour, layer)], text panel) for the current control step.
    ``layer`` is a key of ``LAYERS`` (each can be hidden in the viewer)."""
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
        text["plan lateral accel"] = f"{dec['lat_accel']:4.1f} m/s2"
        text["MPC"] = f"{str(diag.get('status', ''))[:22]}  {1000 * float(diag.get('solve_time', 0) or 0):4.0f} ms"
        if diag.get("fallback"):
            text["FALLBACK"] = f"{diag.get('fallback_reason', '')} / {diag.get('fallback_mode', '')}"
        m = (ref.s >= ego.s - 5) & (ref.s <= ego.s + 150)
        lines.append((np.stack([ref.x[m], ref.y[m]], axis=1), YELLOW, "reference"))
        if diag.get("pred_xy") is not None:
            lines.append((diag["pred_xy"], CYAN, "mpc"))
        if method.tube is not None:
            for mode in CORRIDOR_MODES:              # every mode; the viewer shows the chosen one (O)
                lines += [(seg, col, f"corridor/{mode}")
                          for seg, col in corridor_lines(method.tube, road, mode, method.mpc.dt)]
        for o in objs:
            zl, zw = method.corridor.keepout_half(o)
            lines.append((box_xy(road, o.s - zl, o.s + zl, o.d - zw, o.d + zw), MAGENTA, "keepout"))
    elif name == "m2" and dec:
        text["ref d / v"] = f"{diag.get('ref_d', 0):5.2f} m / {diag.get('ref_v', 0) * 3.6:5.1f} km/h"
        text["cross-track / heading err"] = (f"{diag.get('cross_track', 0):+5.2f} m / "
                                             f"{np.rad2deg(diag.get('heading_error', 0)):+5.1f} deg")
        text["speed err"] = f"{diag.get('speed_error', 0) * 3.6:+5.1f} km/h"
        lines.append((np.stack([ref.x, ref.y], axis=1), YELLOW, "reference"))
        points.append(((ref.x[-1], ref.y[-1]), WHITE, "points"))
        points.append((road.to_world(float(ego.s), float(ref.at_s(ego.s)["d"])), MAGENTA, "points"))
    text["steering / throttle"] = f"{ego.steering:+5.2f} / {ego.throttle_brake:+5.2f}"
    lead = lead_car(ego, objs)
    if lead:
        text["lead car gap / closing"] = f"{lead[0]:5.1f} m / {lead[1] * 3.6:+5.1f} km/h"
    for n, w in env.reward.weights.items():
        if w != 0 and env.recorder.terms[n]:
            text[f"r.{n}"] = f"{w * env.recorder.terms[n][-1]:+.3f}"
    return lines, points, text


# ======================================================================================= viewer
# layer -> toggle key; the keys and the rest of the viewer's controls are listed in the module docstring
LAYERS = {"trail": "1", "reference": "2", "mpc": "3", "corridor": "4", "keepout": "5", "points": "6"}
PANEL_CORNERS = ("top-left", "top-right", "bottom-right", "bottom-left")
STEP_S = 0.2      # N: play this long, then pause again


class Viewer:
    """Draws the overlays into MetaDrive's 3D window after every control step (set as ``env.on_step``), and owns the
    viewer keys (module docstring)."""

    def __init__(self, fps: float = 10.0, show_return: bool = False, paused: bool = False, panel: bool = True):
        self.fps = fps                        # 10 = real time, 0 = as fast as it renders
        self.show_return = show_return        # text panel: return so far + each reward term's running total
        self.extra_text: dict = {}
        self.show, self.finish = True, False  # M: everything (lines, points, panel)
        self.layers = dict.fromkeys(LAYERS, True)
        self.panel, self.panel_corner = panel, 0   # I: text panel on / off, D: its corner
        self.corridor_mode = 0                # O: index into CORRIDOR_MODES
        self.paused, self.steps_left, self._step_request = paused, 0, False   # N: steps_left control steps to play
        self.trail: list = []
        self._frame = None                    # (lines, points, text) of the last control step, redrawn on a key
        self._dirty = False
        self._lines = self._points = None
        self._t_last = time.time()

    def new_episode(self) -> None:
        self.trail, self.finish = [], False

    def _key(self, fn):
        def handler():
            fn()
            self._dirty = True
        return handler

    def _setup(self, env) -> None:
        eng = env.md.engine
        self._lines = eng.make_line_drawer(thickness=3.0)
        self._lines.setShaderOff(1)   # line geometry has no normals: draw unlit or it is invisible
        self._lines.setLightOff(1)
        self._points = eng.make_point_drawer(scale=4.0)
        msg = eng.on_screen_message
        if msg is not None:           # MetaDrive redraws the panel on every physics step too: place it after each
            draw = msg.render

            def render(data=None):
                draw(data)
                self._place_panel(env)
            msg.render = render
        k = self._key
        eng.accept("x", k(lambda: setattr(self, "finish", True)))
        eng.accept("m", k(lambda: setattr(self, "show", not self.show)))
        eng.accept("i", k(lambda: setattr(self, "panel", not self.panel)))
        eng.accept("d", k(lambda: setattr(self, "panel_corner", (self.panel_corner + 1) % len(PANEL_CORNERS))))
        for name, key in LAYERS.items():
            eng.accept(key, k(lambda n=name: self.layers.__setitem__(n, not self.layers[n])))
        for key in ("space", "p"):    # replaces MetaDrive's own P pause, which cannot step
            eng.accept(key, k(lambda: setattr(self, "paused", not self.paused)))
        eng.accept("n", k(lambda: setattr(self, "_step_request", True)))
        eng.accept("o", k(lambda: setattr(self, "corridor_mode", (self.corridor_mode + 1) % len(CORRIDOR_MODES))))

    def __call__(self, env, diag: dict) -> None:
        if self._lines is None:
            self._setup(env)
        if self._step_request:        # N while running: this control step is the first of STEP_S
            self._step_request, self.steps_left = False, max(int(round(STEP_S / env.dt)), 1)
        self.trail = (self.trail + [(env.ego.x, env.ego.y)])[-200:]
        lines, points, text = overlay(env, diag)
        lines.insert(0, (np.array(self.trail), TRAIL, "trail"))
        self._frame = (lines, points, {**self._return_text(env), **self.extra_text, **text})
        self._render(env)
        if self.steps_left:
            self.steps_left -= 1
            if not self.steps_left:
                self.paused = True
        if self.fps > 0:
            wait = 1.0 / self.fps - (time.time() - self._t_last)
            if wait > 0:
                time.sleep(wait)
        self._dirty |= self.paused             # show 'paused' on the panel
        while self.paused and not self.finish:   # keep the window live (keys, camera) while paused
            if self._dirty:
                self._render(env)
            env.md.engine.taskMgr.step()
            time.sleep(1 / 60)
            if self._step_request:    # N while paused: play the next STEP_S, then pause again
                self._step_request, self.steps_left = False, max(int(round(STEP_S / env.dt)), 1)
                self.paused = False
        self._t_last = time.time()

    def _render(self, env) -> None:
        self._dirty = False
        self._lines.reset()
        self._points.reset()
        lines, points, text = self._frame
        mode = CORRIDOR_MODES[self.corridor_mode]
        if self.show:
            self._draw([(xy, c) for xy, c, layer in lines if self._visible(layer, mode)],
                       [(p, c) for p, c, layer in points if self.layers[layer]])
        off = [f"{key} {n}" for n, key in LAYERS.items() if not self.layers[n]]
        env.md.render(text={**text, "keys": "Space pause, N step 0.2 s, M all, I/D panel hide/move, 1-6 layers, "
                            "O corridor style, C screenshot, X end",
                            "layers off": ", ".join(off) or "-",
                            "corridor style": mode,   # MetaDrive keeps old keys: always set both
                            "paused": "YES (Space resume, N step)" if self.paused else "-"})

    def _visible(self, layer: str, mode: str) -> bool:
        """Layer on, and for the corridor ("corridor/<mode>") only the chosen display mode."""
        base, _, sub = layer.partition("/")
        return self.layers[base] and (not sub or sub == mode)

    def _place_panel(self, env) -> None:
        """MetaDrive rebuilds its text panel on every render: hide it, or move it to the chosen corner."""
        from panda3d.core import TextNode

        msg = env.md.engine.on_screen_message
        t = getattr(msg, "onScreenText", None)
        if not t or t.textNode is None:
            return
        if not (self.show and self.panel):
            t.hide()
            return
        corner = PANEL_CORNERS[self.panel_corner]
        eng = env.md.engine
        parent = {"top-left": eng.a2dTopLeft, "top-right": eng.a2dTopRight,
                  "bottom-right": eng.a2dBottomRight, "bottom-left": eng.a2dBottomLeft}[corner]
        right, bottom = corner.endswith("right"), corner.startswith("bottom")
        tn = t.textNode
        tn.setAlign(TextNode.ARight if right else TextNode.ALeft)
        height = tn.getNumRows() * tn.getLineHeight() * t.getTextScale()[1]
        t.reparentTo(parent)
        t.setTextPos(-0.05 if right else 0.05, height + 0.03 if bottom else -0.08)
        t.show()

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
         deterministic: bool = True, paused: bool = False, panel: bool = True) -> list[dict]:
    """Run a policy (checkpoint, or B0 with None) on ``seeds`` with the viewer; each episode is logged to out_dir."""
    from handoff.env import make_env
    from handoff.evaluate import load_policy_fn

    policy = load_policy_fn(checkpoint, deterministic, cfg)      # checked before the window opens
    env = make_env(cfg, render=render, is_eval=True, run_dir=out_dir)
    act = policy or (lambda obs: np.zeros(env.action_space.shape, np.float32))
    viewer = Viewer(fps=fps, paused=paused, panel=panel) if render else None
    summaries = []
    try:
        for seed in seeds:
            summaries.append(drive_episode(env, act, seed, "view", viewer))
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
