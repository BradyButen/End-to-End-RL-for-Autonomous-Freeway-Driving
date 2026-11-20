"""Corridor (tube) construction for the M3 MPC: an initial trajectory plus one convex box per stage, each box
given by 4 half-spaces on the ego centre. Pure numpy; rebuilt at a slower rate than the MPC solves.

Initial trajectory (road/Frenet coordinates, stage k at t_k = k·dt over the tube horizon):
  * follows the M3 spline reference laterally: d_k = d_ref(s_k);
  * speed ramps toward the target speed within ±accel/brake; whenever a car's inflated, constant-velocity-
    predicted footprint blocks the path, the speed is limited so the ego stops ``stop_buffer`` behind it
    (v_k ≤ sqrt(2·brake·(s_cap − s_k)) for every later cap).

Box for stage k (frame "road" = aligned with the road; frame "trajectory" = rotated by the trajectory's
heading relative to the road when that exceeds ``lane_change_heading``, i.e. while lane-changing):
  * the ego is treated as a point (its centre); every car is inflated by the ego's half length/width plus
    margins, predicted at t_k, and taken as its bounding rectangle in the box frame;
  * the drivable band is the road shrunk by the ego half-width plus ``edge_margin``;
  * growth from the trajectory point: lateral faces first (to the nearest inflated car overlapping the
    centre line, the road edge, or ``cap_lateral``), then longitudinal faces (to the nearest car overlapping
    the lateral span, or the caps), then — for rotated boxes — the lateral faces are pulled in until all four
    corners are on the road. The result is an obstacle-free convex box containing the trajectory point,
    unless the point itself is inside an inflated car (stage marked ``blocked``; box degenerate).
  * returned as A·p ≤ b in world coordinates (4 rows per stage) plus corner polygons for visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handoff.rl.methods.common import Reference
from handoff.simulator.road import RoadFrame, wrap_angle
from handoff.simulator.state import EgoState, ObjectList, frenet_velocity


@dataclass
class Tube:
    A: np.ndarray          # (K+1, 4, 2) half-space normals (world)
    b: np.ndarray          # (K+1, 4)    offsets: A[k] @ p <= b[k]
    corners: np.ndarray    # (K+1, 4, 2) box corners (world), for visualisation/logging
    traj: np.ndarray       # (K+1, 6)    initial trajectory [s, d, v, x, y, psi]
    theta: np.ndarray      # (K+1,)      box frame rotation relative to the road (0 = road-aligned)
    blocked: np.ndarray    # (K+1,) bool trajectory point inside an inflated car
    extents: np.ndarray    # (K+1, 4)    [u_minus, u_plus, w_minus, w_plus] in the box frame

    def __len__(self) -> int:
        return len(self.b)


class CorridorBuilder:
    def __init__(self, cfg: dict, ego_length: float, ego_width: float, dt: float, v_target: float, v_cap: float):
        self.cfg = cfg
        self.L, self.W = ego_length, ego_width
        self.dt = dt
        self.v_target = v_target
        self.v_cap = v_cap
        self.K = int(round(float(cfg["horizon"]) / dt))

    # ------------------------------------------------------------------ initial trajectory
    def _predict(self, objects: ObjectList, road: RoadFrame):
        """Per object: (s(t_k), d(t_k)) arrays, inflated half sizes, heading relative to the road."""
        c = self.cfg
        t = np.arange(self.K + 1) * self.dt
        out = []
        for o in objects:
            sdot, ddot = frenet_velocity(o, road)
            hl = 0.5 * (o.length + self.L) + c["margin_s"]
            hw = 0.5 * (o.width + self.W) + c["margin_d"]
            rel = float(wrap_angle(o.psi - road.heading_at(o.s)))
            out.append((o.s + sdot * t, o.d + ddot * t, hl, hw, rel))
        return out

    def initial_trajectory(self, ego: EgoState, reference: Reference, preds) -> np.ndarray:
        c, dt, K = self.cfg, self.dt, self.K
        a_up, b_dn = float(c["accel"]), float(c["brake"])
        s_end = reference.s[-1]
        # target speed: the reference's own (M3: the RL's v* for this decision); fixed target if it has none
        v_goal = float(reference.v[0]) if reference.v is not None and len(reference.v) else self.v_target

        def path_d(s):
            return np.interp(np.minimum(s, s_end), reference.s, reference.d)

        # caps from cars that block the path (lateral overlap with the path at the car's predicted s)
        s_cap = np.full(K + 1, np.inf)
        for so, do, hl, hw, _ in preds:
            for k in range(K + 1):
                if so[k] <= ego.s:
                    continue
                if abs(path_d(so[k]) - do[k]) < hw:
                    s_cap[k] = min(s_cap[k], so[k] - hl - float(c["stop_buffer"]))
        fut_cap = np.minimum.accumulate(s_cap[::-1])[::-1]  # tightest cap at or after stage k

        s = np.empty(K + 1)
        v = np.empty(K + 1)
        s[0], v[0] = ego.s, max(ego.v, 0.0)
        for k in range(1, K + 1):
            v_des = min(v[k - 1] + a_up * dt, v_goal, self.v_cap)
            room = fut_cap[k] - s[k - 1]
            if np.isfinite(room):
                v_des = min(v_des, np.sqrt(2.0 * b_dn * max(room, 0.0)))
            v[k] = max(v_des, v[k - 1] - b_dn * dt, 0.0)
            s[k] = s[k - 1] + 0.5 * (v[k - 1] + v[k]) * dt
        d = path_d(s)
        x, y = reference_xy(reference, s)
        psi = np.array([reference.at_s(min(sk, s_end))["psi"] for sk in s])
        return np.stack([s, d, v, x, y, psi], axis=1)

    # ------------------------------------------------------------------ boxes
    def build(self, ego: EgoState, reference: Reference, objects: ObjectList, road: RoadFrame) -> Tube:
        c = self.cfg
        preds = self._predict(objects, road)
        traj = self.initial_trajectory(ego, reference, preds)
        K = self.K
        d_lo = road.d_min + 0.5 * self.W + float(c["edge_margin"])
        d_hi = road.d_max - 0.5 * self.W - float(c["edge_margin"])
        cap_f, cap_b, cap_l = float(c["cap_forward"]), float(c["cap_backward"]), float(c["cap_lateral"])
        A = np.zeros((K + 1, 4, 2))
        b = np.zeros((K + 1, 4))
        corners = np.zeros((K + 1, 4, 2))
        theta = np.zeros(K + 1)
        blocked = np.zeros(K + 1, bool)
        extents = np.zeros((K + 1, 4))
        for k in range(K + 1):
            sk, dk = traj[k, 0], traj[k, 1]
            th = float(wrap_angle(traj[k, 5] - road.heading_at(sk)))
            th = th if abs(th) > float(c["lane_change_heading"]) else 0.0
            ct, st = np.cos(th), np.sin(th)
            # inflated cars as AABBs in the box frame (u along th, w left of it), centre at the trajectory point
            boxes = []
            for so, do, hl, hw, rel in preds:
                cr, sr = np.cos(rel), np.sin(rel)
                pts = []
                for su, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                    ps = so[k] + cr * su * hl - sr * sw * hw - sk
                    pd = do[k] + sr * su * hl + cr * sw * hw - dk
                    pts.append((ct * ps + st * pd, -st * ps + ct * pd))
                pts = np.array(pts)
                boxes.append((pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()))
            inside = any(ua < 0 < ub and wa < 0 < wb for ua, ub, wa, wb in boxes)
            if inside or not (d_lo - 1e-9 <= dk <= d_hi + 1e-9):
                blocked[k] = True
                um = up = wm = wp = 0.0
            else:
                # 1) lateral faces (u-span = {0})
                wp = min(cap_l, (d_hi - dk) / ct)
                wm = max(-cap_l, (d_lo - dk) / ct)
                for ua, ub, wa, wb in boxes:
                    if ua <= 0 <= ub:
                        if wa >= 0:
                            wp = min(wp, wa)
                        elif wb <= 0:
                            wm = max(wm, wb)
                # 2) longitudinal faces (w-span = [wm, wp])
                up, um = cap_f, -cap_b
                for ua, ub, wa, wb in boxes:
                    if wa < wp and wb > wm:
                        if ua >= 0:
                            up = min(up, ua)
                        elif ub <= 0:
                            um = max(um, ub)
                # 3) rotated boxes: keep all four corners on the road
                if th != 0.0:
                    for _ in range(20):
                        hi_c = max(um * st, up * st)
                        lo_c = min(um * st, up * st)
                        wp2 = min(wp, (d_hi - dk - hi_c) / ct)
                        wm2 = max(wm, (d_lo - dk - lo_c) / ct)
                        if wm2 <= 0 <= wp2:
                            wp, wm = wp2, wm2
                            break
                        up *= 0.5
                        um *= 0.5
                    else:
                        blocked[k] = True
                        um = up = wm = wp = 0.0
            extents[k] = [um, up, wm, wp]
            theta[k] = th
            # world-frame axes of the box
            h = road.heading_at(sk)
            t_w = np.array([np.cos(h), np.sin(h)])
            n_w = np.array([-np.sin(h), np.cos(h)])
            eu = ct * t_w + st * n_w
            ew = -st * t_w + ct * n_w
            cw = np.array(road.to_world(sk, dk))
            A[k] = [eu, -eu, ew, -ew]
            b[k] = [up + eu @ cw, -um - eu @ cw, wp + ew @ cw, -wm - ew @ cw]
            corners[k] = [cw + um * eu + wm * ew, cw + up * eu + wm * ew, cw + up * eu + wp * ew,
                          cw + um * eu + wp * ew]
        return Tube(A=A, b=b, corners=corners, traj=traj, theta=theta, blocked=blocked, extents=extents)


class RoadCorridor(CorridorBuilder):
    """Road-aligned corridor (M3 default, ``corridor.type: road``; developed in scripts/m3_harness.py):
    fixed-width, road-aligned boxes on the ego CENTRE.

    Lateral faces are static: the centres of the two exterior lanes, pulled in by `inset` (default -0.3,
    i.e. 0.3 m wider on each side):
        d_centre in [min(lane_centers) + inset, max(lane_centers) - inset]   (= [-7.3, +0.3] by default)
    Car keep-out zone (the physical zone drawn on screen): the car's own footprint extended keepout_s
    (default 1.0 m) forward and rearward along the road, and margin_d to each side:
        zone half-length = L_car/2 + keepout_s,   zone half-width = W_car/2 + margin_d
    The box constrains the ego CENTRE, so the zone is grown by the ego's half-size before it enters the
    corridor (the ego body then stays out of the zone):
        constraint half-length = zone half-length + L_ego/2,   constraint half-width = zone half-width + W_ego/2
    keepout_s and inset (default -0.3) are read from method.mpc.corridor.
    No braking in the corridor: the stage points follow the reference with M3's speed ramp toward the
    target speed, ignoring cars (so `corridor.brake` / `stop_buffer` are unused). Stopping is the MPC's job.
    With traffic, per stage k (rebuilt every control step), strictly from the keep-out boxes (grown by
    the ego half-size, so "overlap" = the ego body would touch the zone):
      * in path (the reference passes through the box within the box's own s-range; NOT just "the stage
        point's d is inside the box's lateral span", which wrongly caught lane changes that go around the
        car), decided by where the car is relative to the ego NOW:
          ahead  -> front face at the box's rear end, for every stage (also stages whose planned point is
                    at or past the car: the box then ends behind the point, so the MPC must brake);
          behind -> back face at the box's front end;
          alongside -> lateral face on the ego's side of the box;
      * left / right: a box level with the stage point (s_a < s_k < s_b) on that side shrinks the lateral
        face to the box's side edge; as soon as no box is level with s_k it grows back to the fixed band.
    No stage is ever blocked. A box that cannot be reached in time makes the MPC infeasible, which falls
    back to the emergency brake (immediately with max_fallback_shift = 0).
    Every face is fixed in the world: with no car in the path the box has no front/back face (BIG), so it
    does not slide with the trajectory point or jump at rebuilds. Same A·p <= b output as M3's builder."""

    BIG = 1.0e4    # "no face" longitudinal extent (m)

    def __init__(self, cfg: dict, ego_length: float, ego_width: float, dt: float, v_target: float, v_cap: float):
        super().__init__(cfg, ego_length, ego_width, dt, v_target, v_cap)
        self.inset = float(cfg.get("inset", -0.3))   # pull-in from the exterior lane centres (m); negative = wider

    def zone_half(self, o):
        """Physical keep-out zone half-sizes (along, across the road) of a car."""
        c = self.cfg
        return 0.5 * o.length + float(c.get("keepout_s", 1.0)), 0.5 * o.width + float(c["margin_d"])

    def predict_keepout(self, ego, objects, road):
        """Like CorridorBuilder._predict: keep-out zone grown by the ego half-size (constraint on the centre)."""
        t = np.arange(self.K + 1) * self.dt
        out = []
        for o in objects:
            sdot, ddot = frenet_velocity(o, road)
            zl, zw = self.zone_half(o)
            hl = zl + 0.5 * self.L
            hw = zw + 0.5 * self.W
            rel = float(wrap_angle(o.psi - road.heading_at(o.s)))
            out.append((o.s + sdot * t, o.d + ddot * t, hl, hw, rel))
        return out

    def build(self, ego, reference, objects, road):
        preds = self.predict_keepout(ego, objects, road)
        # No braking in the corridor: the stage points follow the reference at the target-speed ramp and
        # ignore cars; stopping for a car is left to the MPC through the world-fixed keep-out faces below.
        traj = self.initial_trajectory(ego, reference, [])
        K = self.K
        inset = float(self.inset)
        d_lo0 = min(road.lane_centers) + inset                          # exterior (right) lane centre
        d_hi0 = max(road.lane_centers) - inset                          # exterior (left) lane centre
        A = np.zeros((K + 1, 4, 2))
        b = np.zeros((K + 1, 4))
        corners = np.zeros((K + 1, 4, 2))
        blocked = np.zeros(K + 1, bool)
        extents = np.zeros((K + 1, 4))
        for k in range(K + 1):
            sk, dk = traj[k, 0], traj[k, 1]
            boxes = []    # inflated cars as road-frame AABBs (s_a, s_b, d_a, d_b), absolute
            for so, do, hl, hw, rel in preds:
                cr, sr = np.cos(rel), np.sin(rel)
                ss = [so[k] + cr * su * hl - sr * sw * hw for su, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                dd = [do[k] + sr * su * hl + cr * sw * hw for su, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                boxes.append((min(ss), max(ss), min(dd), max(dd), do[k]))
            up, um = self.BIG, -self.BIG
            lo, hi = d_lo0, d_hi0
            for sa, sb, da, db, dc in boxes:
                # in path = the reference itself runs through the box over the box's own s-range
                seg = (reference.s >= sa) & (reference.s <= sb)
                in_path = bool(np.any((reference.d[seg] > da) & (reference.d[seg] < db)))
                if in_path:                                  # 1) car on the reference
                    if sa >= ego.s:          # ahead of the ego now: stay behind its keep-out rear face
                        up = min(up, sa - sk)
                    elif sb <= ego.s:        # behind the ego now: stay ahead of its keep-out front face
                        um = max(um, sb - sk)
                    elif ego.d >= dc:        # alongside now: stay on the ego's side of it
                        lo = max(lo, db)
                    else:
                        hi = min(hi, da)
                elif sa < sk < sb:                           # 2) left/right: boxes level with s_k
                    if da >= dk:      # car to the left
                        hi = min(hi, da)
                    else:             # car to the right
                        lo = max(lo, db)
            wm, wp = lo - dk, hi - dk
            extents[k] = [um, up, wm, wp]
            h = road.heading_at(sk)
            eu = np.array([np.cos(h), np.sin(h)])            # along the road
            ew = np.array([-np.sin(h), np.cos(h)])           # left of the road
            cw = np.array(road.to_world(sk, dk))
            A[k] = [eu, -eu, ew, -ew]
            b[k] = [up + eu @ cw, -um - eu @ cw, wp + ew @ cw, -wm - ew @ cw]
            corners[k] = [cw + um * eu + wm * ew, cw + up * eu + wm * ew, cw + up * eu + wp * ew,
                          cw + um * eu + wp * ew]
        return Tube(A=A, b=b, corners=corners, traj=traj, theta=np.zeros(K + 1), blocked=blocked,
                    extents=extents)


def reference_xy(reference: Reference, s: np.ndarray):
    s_end = reference.s[-1]
    sc = np.minimum(s, s_end)
    return np.interp(sc, reference.s, reference.x), np.interp(sc, reference.s, reference.y)


def path_blocked(ego: EgoState, waypoints: np.ndarray, objects: ObjectList, keepout_s: float,
                 margin_d: float, road: RoadFrame | None = None, max_decel: float = 6.0,
                 ego_length: float = 4.515) -> str | None:
    """Emergency-brake gate for M3: id of the first car that (1) sits in the planned path and (2) the ego is
    closing on too fast to stop for; None otherwise.

    (1) In path: the straight line (s, d) from the ego centre to the next waypoint ahead (sampled every 0.25 m)
        crosses the car's keep-out zone (car + keepout_s fore/aft, + margin_d sideways). Only cars whose zone
        starts ahead of the ego's front bumper count (cars alongside/behind are the corridor's job).
    (2) Too fast: with closing speed Δv = ṡ_ego − ṡ_car (along the road) and gap g = zone rear − ego front,
        the deceleration needed to match the car's speed before the zone is a = Δv² / (2 g). The gate fires
        when Δv > 0 and (g <= 0 or a > max_decel). A car at the same speed or pulling away never fires it, so
        the RL can follow a lead by placing waypoints behind (or even past) it while the MPC holds the gap;
        a stopped car fires it only once stopping needs more than max_decel.
    Without ``road`` (static-car callers), object speeds are taken as 0 and the ego speed as v."""
    wps = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    ahead = wps[wps[:, 0] > ego.s]
    if len(ahead) == 0 or not objects:
        return None
    s1, d1 = ahead[0]
    n = max(int(np.hypot(s1 - ego.s, d1 - ego.d) / 0.25), 2)
    s_line, d_line = np.linspace(ego.s, s1, n), np.linspace(ego.d, d1, n)
    v_ego = frenet_velocity(ego, road)[0] if road is not None else ego.v
    front = ego.s + 0.5 * ego_length
    for o in objects:
        hl = 0.5 * o.length + keepout_s
        hw = 0.5 * o.width + margin_d
        rear = o.s - hl
        if rear < front - 0.5 * ego_length:          # zone starts behind the ego centre: alongside / behind
            continue
        if not np.any((np.abs(s_line - o.s) < hl) & (np.abs(d_line - o.d) < hw)):
            continue
        v_obj = frenet_velocity(o, road)[0] if road is not None else 0.0
        closing = v_ego - v_obj
        if closing <= 0.0:
            continue
        gap = rear - front
        if gap <= 0.0 or closing**2 / (2.0 * gap) > max_decel:
            return o.id
    return None
