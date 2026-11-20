"""Overlays drawn into MetaDrive's own 3D viewer (spec §13) with ``engine.make_line_drawer`` /
``engine.make_point_drawer`` and the on-screen text panel.

Frames are plain dicts so live mode (built from the running controllers) and replay mode (built from the
logged npz arrays) share the drawing code:
    ref_xy (n,2), pred_xy (m,2), wp_raw_xy (k,2), wp_proj_xy (k,2), boxes (list of (4,2) polygons),
    obj_pred (list of (4,2) polygons), text (dict)
"""

from __future__ import annotations

import numpy as np

Z = 0.6  # draw height (m) above the road

COLORS = {
    "ref": (1.0, 0.85, 0.1, 1.0),        # reference trajectory / spline
    "pred": (0.1, 0.9, 1.0, 1.0),        # MPC predicted path
    "wp_raw": (1.0, 0.2, 0.2, 1.0),      # raw RL waypoints
    "wp_proj": (0.2, 1.0, 0.3, 1.0),     # projected waypoints
    "box": (1.0, 0.3, 1.0, 1.0),         # safe-set footprints
    "obj_pred": (1.0, 0.6, 0.2, 1.0),    # MPC object predictions (end of horizon)
    "tube": (0.3, 0.5, 1.0, 1.0),        # MPC corridor box at the current stage
    "tube_future": (0.15, 0.25, 0.6, 1.0),  # corridor boxes further along the MPC horizon
    "wp_unused": (0.55, 0.55, 0.55, 1.0),  # waypoint slots the RL output but switched off with its count k
    "wp_link": (1.0, 0.45, 0.45, 1.0),   # ego -> waypoint 1 -> ... polyline (what the emergency gate checks)
}


def frenet_box_polygon(road, box) -> np.ndarray:
    s_lo, s_hi, d_lo, d_hi = box
    s = np.array([s_lo, s_hi, s_hi, s_lo, s_lo])
    d = np.array([d_lo, d_lo, d_hi, d_hi, d_lo])
    x, y = road.to_world(s, d)
    return np.stack([x, y], axis=1)


def oriented_box(x, y, psi, length, width) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2],
                    [-length / 2, width / 2], [length / 2, width / 2]])
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1)


class OverlayDrawer:
    def __init__(self, engine):
        self.engine = engine
        self.lines = engine.make_line_drawer(thickness=3.0)
        # MetaDrive's line drawer enables the auto shader + scene lights; line geometry has no normals, so the lines
        # render invisible. Draw them unlit with the fixed-function colours instead.
        self.lines.setShaderOff(1)
        self.lines.setLightOff(1)
        self.points = engine.make_point_drawer(scale=4.0)

    def draw(self, frame: dict) -> None:
        self.lines.reset()
        self.points.reset()
        lines, colors = [], []

        def add_line(xy, color):
            xy = np.asarray(xy, float)
            xy = xy[np.all(np.isfinite(xy), axis=1)] if xy.ndim == 2 else xy
            if xy.ndim != 2 or len(xy) < 2:
                return
            lines.append([(float(p[0]), float(p[1]), Z) for p in xy])
            colors.append([color] * (len(xy) - 1))

        add_line(frame.get("ref_xy", []), COLORS["ref"])
        add_line(frame.get("pred_xy", []), COLORS["pred"])
        for poly in frame.get("boxes", []):
            add_line(poly, COLORS["box"])
        for poly in frame.get("obj_pred", []):
            add_line(poly, COLORS["obj_pred"])
        for poly in frame.get("tube", []):
            add_line(np.vstack([poly, poly[:1]]), COLORS["tube"])
        for poly in frame.get("tube_future", []):
            add_line(np.vstack([poly, poly[:1]]), COLORS["tube_future"])
        add_line(frame.get("wp_link", []), COLORS["wp_link"])
        if lines:
            self.lines.draw_lines(lines, colors)
        pts, pcols = [], []
        for key, col in (("wp_unused_xy", "wp_unused"), ("wp_proj_xy", "wp_proj"), ("wp_raw_xy", "wp_raw")):
            arr = np.asarray(frame.get(key, []), float).reshape(-1, 2)
            for p in arr:
                if np.all(np.isfinite(p)):
                    pts.append((float(p[0]), float(p[1]), Z + 0.3))
                    pcols.append(COLORS[col])
        if pts:
            self.points.draw_points(pts, pcols)


def live_frame(core, decision_ctx: dict | None, diag: dict | None, mpc_horizon_s: float = 2.0,
               tube_every: int = 10) -> dict:
    """Build an overlay frame from the running env/controllers (live view and re-simulated replay)."""
    diag = diag or {}
    frame: dict = {}
    road = core.road
    ref = (decision_ctx or {}).get("reference")
    if ref is not None:
        m = ref.s <= core.ego.s + 120.0
        frame["ref_xy"] = np.stack([ref.x[m], ref.y[m]], axis=1)
    if diag.get("pred_xy") is not None:
        frame["pred_xy"] = diag["pred_xy"]
    if diag.get("tube_corners") is not None:
        frame["tube"] = [np.asarray(diag["tube_corners"]).reshape(-1, 4, 2)[0]]   # current stage
    tube = diag.get("tube_full")
    if tube is not None and getattr(tube, "corners", None) is not None:
        frame["tube_future"] = list(tube.corners[tube_every::tube_every])   # every 1 s along the 5 s horizon
    dec = (decision_ctx or {}).get("decision") or {}
    if "waypoints_raw" in dec:
        raw = np.asarray(dec["waypoints_raw"]).reshape(-1, 2)
        frame["wp_raw_xy"] = np.stack(road.to_world(raw[:, 0], raw[:, 1]), axis=1)
        frame["wp_link"] = np.vstack([[core.ego.x, core.ego.y], frame["wp_raw_xy"]])
        k = len(raw)
        vec = np.asarray(dec.get("vector", []), float)
        n_slots = (len(vec) - 2) // 2
        if n_slots > k:
            unused = vec[: 2 * n_slots].reshape(-1, 2)[k:]
            frame["wp_unused_xy"] = np.stack(road.to_world(unused[:, 0], unused[:, 1]), axis=1)
        proj = np.asarray(dec["waypoints_projected"]).reshape(-1, 2)
        ok = np.all(np.isfinite(proj), axis=1)
        if ok.any():
            frame["wp_proj_xy"] = np.stack(road.to_world(proj[ok, 0], proj[ok, 1]), axis=1)
        frame["boxes"] = [frenet_box_polygon(road, b) for bl in dec.get("boxes", [])[:1] for b in bl]
        frame["obj_pred"] = [oriented_box(o.x + o.vx * mpc_horizon_s, o.y + o.vy * mpc_horizon_s, o.psi, o.length,
                                          o.width) for o in core.objects]
    frame["text"] = status_text(core, dec, diag)
    return frame


def status_text(core, dec: dict, diag: dict) -> dict:
    e = core.ego
    w = core.reward.weights
    terms = core.recorder.step_terms
    active = {n: terms[n][-1] for n in w if w[n] != 0 and terms.get(n)}
    txt = {
        "method": core.cfg["method"]["name"],
        "speed": f"{e.v * 3.6:5.1f} km/h",
        "s / d": f"{e.s:6.1f} / {e.d:5.2f} m",
        "decision": core.episode_decisions,
    }
    if "d_target" in dec:
        txt["RL d* / v*"] = f"{dec['d_target']:5.2f} m / {dec['v_target'] * 3.6:5.1f} km/h"
    if "k" in dec and core.cfg["method"]["name"] == "m3":
        txt["RL v* (max speed)"] = f"{dec['v_target'] * 3.6:5.1f} km/h"
        raw = np.asarray(dec.get("waypoints_raw", []), float).reshape(-1, 2)
        txt["RL waypoints k"] = dec["k"]
        for j, (s, d) in enumerate(raw):
            txt[f"  wp{j + 1} ahead / d"] = f"{s - e.s:5.1f} / {d:5.2f} m"
    if diag.get("solve_time"):
        txt["solve time"] = f"{diag['solve_time'] * 1e3:5.1f} ms (cpu {diag.get('cpu_time', 0) * 1e3:5.1f})"
    if diag.get("status"):
        txt["MPC status"] = str(diag["status"])[:28]
    if diag.get("tube_blocked") is not None:
        txt["corridor faces from cars"] = diag["tube_blocked"]
    if core.objects:
        ahead = [o for o in core.objects if o.s > e.s and abs(o.d - e.d) < 2.0]
        if ahead:
            o = min(ahead, key=lambda o: o.s)
            txt["lead car gap / dv"] = f"{o.s - e.s - (o.length + e.length) / 2:5.1f} m / " \
                                       f"{(e.v - o.v) * 3.6:+5.1f} km/h"
    if diag.get("gated_brake"):
        txt["EMERGENCY BRAKE"] = "path to next waypoint blocked"
    if diag.get("fallback"):
        txt["FALLBACK"] = f"{diag.get('fallback_reason', '')} ({diag.get('fallback_mode', '')})"
    for n, v in active.items():
        txt[f"r.{n}"] = f"{v:+.3f}"
    return txt


def replay_frame(data: dict, i: int, road) -> dict:
    """Overlay frame for control step ``i`` from a logged episode (``load_episode(...)['data']``)."""
    frame: dict = {}
    dec_idx = int(data["step_decision_index"][i]) if "step_decision_index" in data else 0
    refs = data.get("references")
    if refs is not None and refs.ndim == 3 and dec_idx < len(refs):
        r = refs[dec_idx]
        r = r[np.isfinite(r[:, 0])]
        if len(r):
            frame["ref_xy"] = r[:, 3:5]
    pp = data.get("pred_paths")
    if pp is not None and pp.ndim == 3 and i < len(pp):
        frame["pred_xy"] = pp[i]
    wr = data.get("dec_waypoints_raw")
    if wr is not None and wr.ndim == 3 and dec_idx < len(wr):
        raw = wr[dec_idx]
        raw = raw[np.all(np.isfinite(raw), axis=1)]
        if len(raw):
            frame["wp_raw_xy"] = np.stack(road.to_world(raw[:, 0], raw[:, 1]), axis=1)
        pj = data["dec_waypoints_projected"][dec_idx]
        pj = pj[np.all(np.isfinite(pj), axis=1)]
        if len(pj):
            frame["wp_proj_xy"] = np.stack(road.to_world(pj[:, 0], pj[:, 1]), axis=1)
    frame["text"] = {
        "replay step": i,
        "speed": f"{data['step_v'][i] * 3.6:5.1f} km/h",
        "s / d": f"{data['step_s'][i]:6.1f} / {data['step_d'][i]:5.2f} m",
        "solve time": f"{data['step_solve_time'][i] * 1e3:5.1f} ms",
    }
    if "status" in data and i < len(data["status"]) and str(data["status"][i]).startswith("fallback"):
        frame["text"]["FALLBACK"] = str(data["status"][i])
    return frame
