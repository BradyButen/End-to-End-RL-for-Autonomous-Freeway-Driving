"""M3 — RL + MPC. Every 10 control steps (1 s) the RL picks up to 5 waypoints and a speed; every control step the
pipeline below turns them into controls.

    action [-1, 1]^12 = [ds_1, d_1, ..., ds_5, d_5, v*, count]
      decode   k = 1..5 waypoints (s, d): ds in [5, 40] m per slot (cumulative from the ego), d in
               [right edge - 1 m, left edge + 1 m]; v* in [20, 80] km/h                          (per decision)
      spline   piecewise quintic in s through the waypoints, flat 80 m beyond the last, speed v* (per decision)
      gate     a car in the planned path (the line ego -> next waypoint crosses its keep-out zone) that the ego
               closes on faster than it can stop for (dv^2 / 2 gap > gate_decel) -> emergency brake, no MPC solve
      corridor one road-aligned box per MPC stage around the trajectory (spline at a ramp to v*):
               lateral faces at the exterior lane centres (+- inset), cut by the cars' keep-out zones
      MPC      mpc.py: tracks the trajectory inside the corridor (hard constraints); v* is also its speed ceiling

Episode end (ebrake.ends_episode): the gate, or an emergency brake after an INFEASIBLE MPC, ends the episode as
"emergency_brake" with the collision penalty (the controller gave up: a predicted crash). Timeouts never do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handoff.env import Method, affine, unaffine
from handoff.mpc import MPC
from handoff.reference import Reference, eval_quintic, quintic_coefficients
from handoff.sim import ActuationMap, EgoState, ObjectList, RoadFrame, frenet_velocity, wrap_angle


# ======================================================================================= spline
def spline_reference(ego: EgoState, road: RoadFrame, waypoints: np.ndarray, v: float, ds: float = 1.0,
                     extend: float = 80.0, max_start_slope: float = 0.5) -> Reference:
    """Piecewise quintic d(s) from the ego through the (k, 2) waypoints (s, d), increasing in s. The first segment
    starts at the ego's d with slope tan(psi_err); every waypoint is passed with zero slope and curvature (C^2, no
    overshoot between waypoints). Extended flat ``extend`` m beyond the last waypoint."""
    wps = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    knots_s = np.concatenate([[ego.s], wps[:, 0]])
    knots_d = np.concatenate([[ego.d], wps[:, 1]])
    s = np.arange(ego.s, knots_s[-1] + extend + 1e-9, ds)
    d = np.full_like(s, knots_d[-1])
    slope = np.zeros_like(s)
    slope0 = float(np.clip(np.tan(ego.psi_err), -max_start_slope, max_start_slope))
    n_seg = len(knots_s) - 1
    for j in range(n_seg):
        s0, s1 = knots_s[j], knots_s[j + 1]
        m = (s >= s0) & ((s < s1) if j < n_seg - 1 else (s <= s1))
        if np.any(m):
            c = quintic_coefficients(knots_d[j], slope0 if j == 0 else 0.0, 0.0, knots_d[j + 1], s1 - s0)
            d[m], slope[m], _ = eval_quintic(c, s[m] - s0, s1 - s0)
    x, y = road.to_world(s, d)
    return Reference(t=(s - ego.s) / max(v, 0.1), s=s, d=d, x=np.asarray(x), y=np.asarray(y),
                     psi=np.asarray(road.heading_at(s) + np.arctan(slope)), v=np.full_like(s, v), a=np.zeros_like(s))


# ======================================================================================= corridor
@dataclass
class Tube:
    """One box per MPC stage (stage k at t = k dt), as 4 half-spaces on the ego centre: A[k] @ p <= b[k]."""
    A: np.ndarray        # (N+1, 4, 2) normals (world): +along, -along, +left, -left
    b: np.ndarray        # (N+1, 4)
    corners: np.ndarray  # (N+1, 4, 2) box corners (world), for drawing / logs
    traj: np.ndarray     # (N+1, 6) the stage points [s, d, v, x, y, psi] (what the MPC tracks)
    extents: np.ndarray  # (N+1, 4) [back, front, right, left] face offsets from the stage point (road frame, m)


class Corridor:
    """Road-aligned corridor, rebuilt every control step.

    Stage points follow the spline at a speed ramp to v* (+accel / -brake m/s^2), ignoring cars: stopping for a car
    is the MPC's job, through the keep-out faces. Every face is fixed in the world, so boxes do not slide with the
    stage points:
      * lateral faces: the exterior lane centres pulled in by ``inset`` (-0.3 = 0.3 m wider: d in [-7.3, +0.3]);
      * keep-out zone of a car: its footprint + keepout_s fore / aft + margin_d sideways, predicted at constant
        velocity, grown by the ego half-size (the box constrains the ego CENTRE, so the body stays out of the zone);
      * a zone the spline runs through (within the zone's own s-range) is "in path", and depending on where the car
        is NOW: ahead -> front face at the zone's rear (every stage: past the car the box ends behind the stage
        point, so the MPC must brake); behind -> back face at its front; alongside -> lateral face on the ego's side;
      * a zone NOT in path but level with the stage point (s_rear < s_k < s_front) cuts that side's lateral face.
    No front / back face = BIG. A face that cannot be reached in time makes the MPC infeasible (-> emergency brake).
    """

    BIG = 1.0e4

    def __init__(self, cfg: dict, n_stages: int, dt: float, ego_length: float, ego_width: float, v_cap: float):
        self.inset = float(cfg["inset"])
        self.keepout_s, self.margin_d = float(cfg["keepout_s"]), float(cfg["margin_d"])
        self.accel, self.brake = float(cfg["accel"]), float(cfg["brake"])
        self.K, self.dt = int(n_stages), float(dt)
        self.L, self.W = ego_length, ego_width
        self.v_cap = v_cap

    def keepout_half(self, o) -> tuple[float, float]:
        """Half-sizes (along, across the road) of a car's physical keep-out zone."""
        return 0.5 * o.length + self.keepout_s, 0.5 * o.width + self.margin_d

    def trajectory(self, ego: EgoState, ref: Reference) -> np.ndarray:
        """Stage points [s, d, v, x, y, psi]: the spline at a ramp from the ego's speed to the reference speed."""
        K, dt = self.K, self.dt
        v_goal = float(ref.v[0])
        s, v = np.empty(K + 1), np.empty(K + 1)
        s[0], v[0] = ego.s, max(ego.v, 0.0)
        for k in range(1, K + 1):
            v_up = min(v[k - 1] + self.accel * dt, v_goal, self.v_cap)
            v[k] = max(v_up, v[k - 1] - self.brake * dt, 0.0)
            s[k] = s[k - 1] + 0.5 * (v[k - 1] + v[k]) * dt
        sc = np.minimum(s, ref.s[-1])
        interp = lambda arr: np.interp(sc, ref.s, arr)  # noqa: E731
        return np.stack([s, interp(ref.d), v, interp(ref.x), interp(ref.y), interp(np.unwrap(ref.psi))], axis=1)

    def build(self, ego: EgoState, ref: Reference, objects: ObjectList, road: RoadFrame) -> Tube:
        K = self.K
        traj = self.trajectory(ego, ref)
        t = np.arange(K + 1) * self.dt
        zones = []      # per car: predicted centre (s(t), d(t)), constraint half-sizes, heading vs the road
        for o in objects:
            sdot, ddot = frenet_velocity(o, road)
            zl, zw = self.keepout_half(o)
            zones.append((o.s + sdot * t, o.d + ddot * t, zl + 0.5 * self.L, zw + 0.5 * self.W,
                          float(wrap_angle(o.psi - road.heading_at(o.s)))))
        d_lo0 = min(road.lane_centers) + self.inset
        d_hi0 = max(road.lane_centers) - self.inset
        A, b = np.zeros((K + 1, 4, 2)), np.zeros((K + 1, 4))
        corners, extents = np.zeros((K + 1, 4, 2)), np.zeros((K + 1, 4))
        corner_signs = ((1, 1), (1, -1), (-1, -1), (-1, 1))
        for k in range(K + 1):
            sk, dk = traj[k, 0], traj[k, 1]
            front, back, lo, hi = self.BIG, -self.BIG, d_lo0, d_hi0
            for so, do, hl, hw, rel in zones:
                cr, sr = np.cos(rel), np.sin(rel)   # the zone's road-frame bounding box at stage k
                ss = [so[k] + cr * su * hl - sr * sw * hw for su, sw in corner_signs]
                dd = [do[k] + sr * su * hl + cr * sw * hw for su, sw in corner_signs]
                sa, sb, da, db = min(ss), max(ss), min(dd), max(dd)
                seg = (ref.s >= sa) & (ref.s <= sb)
                if np.any((ref.d[seg] > da) & (ref.d[seg] < db)):    # in path
                    if sa >= ego.s:            # ahead of the ego now: stay behind its rear face
                        front = min(front, sa - sk)
                    elif sb <= ego.s:          # behind the ego now: stay ahead of its front face
                        back = max(back, sb - sk)
                    elif ego.d >= do[k]:       # alongside now: stay on the ego's side
                        lo = max(lo, db)
                    else:
                        hi = min(hi, da)
                elif sa < sk < sb:             # level with the stage point, beside the path
                    if da >= dk:
                        hi = min(hi, da)
                    else:
                        lo = max(lo, db)
            right, left = lo - dk, hi - dk
            extents[k] = [back, front, right, left]
            h = road.heading_at(sk)
            eu, ew = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])   # along, left
            c = np.array(road.to_world(sk, dk))
            A[k] = [eu, -eu, ew, -ew]
            b[k] = [front + eu @ c, -back - eu @ c, left + ew @ c, -right - ew @ c]
            corners[k] = [c + back * eu + right * ew, c + front * eu + right * ew, c + front * eu + left * ew,
                          c + back * eu + left * ew]
        return Tube(A=A, b=b, corners=corners, traj=traj, extents=extents)


# ======================================================================================= gate
def path_blocked(ego: EgoState, waypoints: np.ndarray, objects: ObjectList, road: RoadFrame, keepout_s: float,
                 margin_d: float, max_decel: float) -> str | None:
    """Emergency-brake gate: id of the first car that (1) is in the planned path and (2) the ego closes on too fast
    to stop for; None otherwise.

    (1) The straight line (s, d) from the ego centre to the next waypoint ahead (sampled every 0.25 m) crosses the
        car's keep-out zone, and the zone starts ahead of the ego centre (cars alongside / behind: the corridor).
    (2) With closing speed dv = s_dot_ego - s_dot_car and gap = zone rear - ego front bumper, stopping needs
        a = dv^2 / (2 gap); fires when dv > 0 and (gap <= 0 or a > max_decel). Following a car at its speed (or a
        car pulling away) never fires it."""
    wps = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    ahead = wps[wps[:, 0] > ego.s]
    if len(ahead) == 0 or not objects:
        return None
    s1, d1 = ahead[0]
    n = max(int(np.hypot(s1 - ego.s, d1 - ego.d) / 0.25), 2)
    s_line, d_line = np.linspace(ego.s, s1, n), np.linspace(ego.d, d1, n)
    v_ego = frenet_velocity(ego, road)[0]
    front = ego.s + 0.5 * ego.length
    for o in objects:
        hl, hw = 0.5 * o.length + keepout_s, 0.5 * o.width + margin_d
        rear = o.s - hl
        if rear < ego.s:
            continue
        if not np.any((np.abs(s_line - o.s) < hl) & (np.abs(d_line - o.d) < hw)):
            continue
        closing = v_ego - frenet_velocity(o, road)[0]
        if closing <= 0.0:
            continue
        gap = rear - front
        if gap <= 0.0 or closing**2 / (2.0 * gap) > max_decel:
            return o.id
    return None


# ======================================================================================= method
def waypoint_count(c: float, n_slots: int) -> int:
    """Count channel c in [-1, 1] -> k = clip(1 + floor((c + 1) / 2 * n), 1, n)."""
    c = float(np.clip(c, -1.0, 1.0))
    return int(np.clip(1 + np.floor((c + 1.0) / 2.0 * n_slots), 1, n_slots))


class M3(Method):
    def __init__(self, cfg: dict, env):
        m = cfg["method"]
        self.n_slots = int(m["n_slots"])
        super().__init__(env, period=m["decision_period"], action_dim=2 * self.n_slots + 2)
        self.ds_range = tuple(float(x) for x in m["ds_range"])
        self.lateral_margin = float(m["lateral_margin"])
        lo, hi = m["speed_range_km_h"]
        self.v_range = (float(lo) / 3.6, min(float(hi) / 3.6, env.v_cap))
        self.spline_cfg = m["spline"]
        eb = m["ebrake"]
        self.gate, self.gate_decel = bool(eb["gate"]), float(eb["gate_decel"])
        self.ebrake_ends = bool(eb["ends_episode"])
        ego = cfg["sim"]["ego"]
        self.mpc = MPC(m["mpc"], ActuationMap(cfg["actuation"]["table"]), lf=ego["front_wheelbase"],
                       lr=ego["rear_wheelbase"], max_steering_rad=env.max_steering_rad, v_cap=env.v_cap)
        self.corridor = Corridor(m["corridor"], self.mpc.N, self.mpc.dt, ego["length"], ego["width"], env.v_cap)
        self.waypoints = np.zeros((0, 2))   # the k waypoints in use (s, d)
        self.tube: Tube | None = None       # this step's corridor (None while the gate brakes)

    def _d_range(self) -> tuple[float, float]:
        road = self.env.road
        return road.d_min - self.lateral_margin, road.d_max + self.lateral_margin

    def decode(self, action) -> tuple[np.ndarray, int, float]:
        """-> all 5 slots as (s, d), the count k, v* (m/s)."""
        ds = np.array([affine(action[2 * i], *self.ds_range) for i in range(self.n_slots)])
        d = np.array([affine(action[2 * i + 1], *self._d_range()) for i in range(self.n_slots)])
        s = self.env.ego.s + np.cumsum(ds)
        return np.stack([s, d], axis=1), waypoint_count(action[-1], self.n_slots), affine(action[-2], *self.v_range)

    def encode(self, ds: list[float], d: list[float], v: float) -> np.ndarray:
        """Inverse of ``decode`` (scripted waypoints): per-slot spacing ds and offsets d (k = len(ds); unused slots
        repeat the last), speed v (m/s)."""
        k = len(ds)
        a = np.zeros(2 * self.n_slots + 2)
        for i in range(self.n_slots):
            a[2 * i] = unaffine(ds[min(i, k - 1)], *self.ds_range)
            a[2 * i + 1] = unaffine(d[min(i, k - 1)], *self._d_range())
        a[-2] = unaffine(v, *self.v_range)
        a[-1] = 2 * (k - 0.5) / self.n_slots - 1
        return np.clip(a, -1, 1)

    def reset(self):
        super().reset()
        self.mpc.reset(self.env.road)
        self.waypoints = np.zeros((0, 2))
        self.tube = None

    def decide(self, action):
        slots, k, v_star = self.decode(action)
        self.waypoints = slots[:k]
        self.mpc.set_speed_limit(v_star)
        self.reference = spline_reference(self.env.ego, self.env.road, self.waypoints, v_star,
                                          ds=self.spline_cfg["ds"], extend=self.spline_cfg["extend"])
        return {"vector": np.concatenate([slots.ravel(), [v_star, k]]), "k": k, "v_target": v_star,
                "waypoints": self.waypoints, "slots": slots}

    def control(self):
        env = self.env
        ego, objects, road = env.ego, env.objects, env.road
        blocker = None
        if self.gate:
            blocker = path_blocked(ego, self.waypoints, objects, road, self.corridor.keepout_s, self.corridor.margin_d,
                                   self.gate_decel)
        if blocker is not None:
            self.tube = None
            u = self.mpc.brake(ego)
            diag = {"status": "skipped:emergency_brake", "gated_brake": True, "ref_v": ego.v,
                    "pred_xy": np.array([[ego.x, ego.y]])}
        else:
            self.tube = self.corridor.build(ego, self.reference, objects, road)
            u, diag = self.mpc.step(ego, self.tube)
            diag["tube"] = self.tube
        r = self.reference.at_s(ego.s)
        diag.update(tracking_error=abs(r["d"] - ego.d), ref_d=r["d"], ref_psi=r["psi"], blocker=blocker)
        if self.ebrake_ends and (blocker is not None or diag.get("fallback_terminal")):
            diag["ebrake_terminal"] = True     # the env ends the episode after this step (collision penalty)
        return u, diag
