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

from handoff.controllers.base import Reference
from handoff.sim.road import RoadFrame, wrap_angle
from handoff.sim.state import EgoState, ObjectList, frenet_velocity


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
            v_des = min(v[k - 1] + a_up * dt, self.v_target, self.v_cap)
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


def reference_xy(reference: Reference, s: np.ndarray):
    s_end = reference.s[-1]
    sc = np.minimum(s, s_end)
    return np.interp(sc, reference.s, reference.x), np.interp(sc, reference.s, reference.y)
