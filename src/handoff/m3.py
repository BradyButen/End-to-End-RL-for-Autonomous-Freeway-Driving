"""M3 — RL + MPC. Every 10 control steps (1 s) the RL picks up to 5 waypoints and a speed; every control step the
pipeline below turns them into controls.

    action [-1, 1]^12 = [ds_1, d_1, ..., ds_5, d_5, v*, count]
      decode   k = 1..5 waypoints (s, d): ds in [5, 40] m per slot (cumulative from the ego), d in
               [right edge - 1 m, left edge + 1 m]; v* in [20, 80] km/h                          (per decision)
      check    the waypoints must be drivable: each spline segment's peak lateral acceleration
               (5.77 |dd| v^2 / ds^2) within the MPC's max_lat_accel, else the plan is infeasible  (per decision)
      spline   piecewise quintic in s through the waypoints, flat 80 m beyond the last, speed v* (per decision)
      gate     (ebrake.gate, OFF by default) a car in the planned path (the line ego -> next waypoint crosses its
               keep-out zone) closed on faster than gate_decel allows -> emergency brake, no MPC solve
      corridor one road-aligned box per MPC stage around the trajectory (spline at a ramp to v*):
               lateral faces at the exterior lane centres (+- inset), cut by the cars' keep-out zones
      MPC      mpc.py: tracks the trajectory inside the corridor (soft constraints: exact penalty, so the corridor
               holds whenever it can); v* is also its speed ceiling. Braking for cars is the MPC's job.

Failed runs (ebrake.ends_episode), with the collision penalty: the RL is charged for plans the controller cannot
carry out.
  * infeasible waypoints -> the episode ends at once as "infeasible_plan";
  * the gate fires (only with ebrake.gate on) -> "emergency_brake".
MPC solve failures never end the episode: the previous plan is replayed, then the car holds speed and lane until
a solve succeeds (mpc.py). With ends_episode false (scripts/validate.py run) nothing ends the run.
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
    is the MPC's job, through the keep-out faces. Cars ahead and alongside count; a car behind counts only while its
    front bumper is within ``rear_range`` (2 m) of the ego's rear bumper. Every face is fixed in the world, so boxes
    do not slide with the stage points:
      * lateral faces: the exterior lane centres pulled in by ``inset`` (-0.3 = 0.3 m wider: d in [-7.3, +0.3]);
      * keep-out zone of a car: its footprint + keepout_s fore / aft + margin_d sideways, predicted at constant
        velocity, grown by the ego half-size (the box constrains the ego CENTRE, so the body stays out of the zone);
      * each car's face is decided stage by stage, from where the stage point (s_k, d_k) and the car's predicted
        zone are AT THAT STAGE, with a state that starts from where the car is now (ahead / behind / alongside)
        and carries through the horizon:
          behind it (zone ahead): front face at the zone's rear while the stage point overlaps it laterally;
          level with it, beside it: lateral face on the stage point's side;
          ahead of it (zone behind): back face at the zone's front while overlapping laterally.
        Behind -> level only at a stage where the stage point is level with the zone AND laterally clear (passing
        beside); a plan that runs into or through the car stays "behind" for the rest of the horizon (front face:
        it cannot drive through, even if the plan is laterally clear once past it). Level -> ahead once the stage point
        is past the zone: a plan that overtakes a slower car and merges in front of it gets a back face there, not
        a front face (the old rule - any car ahead NOW whose zone the spline crosses -> front face at every
        stage - made M3 follow a slower car in the next lane instead of passing it).
    No front / back face = BIG. A face that cannot be reached in time makes the MPC infeasible (-> emergency brake).
    """

    BIG = 1.0e4

    def __init__(self, cfg: dict, n_stages: int, dt: float, ego_length: float, ego_width: float, v_cap: float):
        self.inset = float(cfg["inset"])
        self.rear_range = float(cfg["rear_range"])
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
        ego_rear = ego.s - 0.5 * ego.length
        for o in objects:
            if o.s + 0.5 * o.length < ego_rear - self.rear_range:   # behind the ego beyond rear_range: ignored
                continue
            sdot, ddot = frenet_velocity(o, road)
            zl, zw = self.keepout_half(o)
            zones.append((o.s + sdot * t, o.d + ddot * t, zl + 0.5 * self.L, zw + 0.5 * self.W,
                          float(wrap_angle(o.psi - road.heading_at(o.s)))))
        # Initial state of each car relative to the ego NOW (zone rear / front vs the ego's s), then advanced stage by
        # stage. (Classifying a car with its predicted zone against the ego's CURRENT s turned a slower car the ego
        # had just passed into "ahead" for the late stages; here the predicted zone is compared with the stage
        # point at the same time.)
        BEHIND, LEVEL, AHEAD = 0, 1, 2      # the stage point is behind / level with / ahead of the car's zone
        state, side = [], []
        for so, do, hl, _, _ in zones:
            rear_now, front_now = so[0] - hl, so[0] + hl
            state.append(BEHIND if rear_now >= ego.s else AHEAD if front_now <= ego.s else LEVEL)
            side.append(ego.d >= do[0])     # True: the ego is on the car's left
        d_lo0 = min(road.lane_centers) + self.inset
        d_hi0 = max(road.lane_centers) - self.inset
        A, b = np.zeros((K + 1, 4, 2)), np.zeros((K + 1, 4))
        corners, extents = np.zeros((K + 1, 4, 2)), np.zeros((K + 1, 4))
        corner_signs = ((1, 1), (1, -1), (-1, -1), (-1, 1))
        for k in range(K + 1):
            sk, dk = traj[k, 0], traj[k, 1]
            front, back, lo, hi = self.BIG, -self.BIG, d_lo0, d_hi0
            for i, (so, do, hl, hw, rel) in enumerate(zones):
                cr, sr = np.cos(rel), np.sin(rel)   # the zone's road-frame bounding box at stage k
                ss = [so[k] + cr * su * hl - sr * sw * hw for su, sw in corner_signs]
                dd = [do[k] + sr * su * hl + cr * sw * hw for su, sw in corner_signs]
                sa, sb, da, db = min(ss), max(ss), min(dd), max(dd)
                overlap = da < dk < db
                lon = BEHIND if sk < sa else AHEAD if sk > sb else LEVEL
                st = state[i]
                if st == BEHIND and lon == LEVEL and not overlap:        # drew level beside it: passing
                    st, side[i] = LEVEL, dk >= do[k]
                if st == LEVEL and lon == AHEAD:                         # past it
                    st = AHEAD
                elif st == LEVEL and lon == BEHIND:                      # fell back
                    st = BEHIND
                elif st == AHEAD and lon == LEVEL and not overlap:       # it caught up, beside the stage point
                    st, side[i] = LEVEL, dk >= do[k]
                state[i] = st
                if st == BEHIND and (overlap or lon != BEHIND):          # stay behind its rear
                    front = min(front, sa - sk)
                elif st == AHEAD and (overlap or lon != AHEAD):          # stay ahead of its front
                    back = max(back, sb - sk)
                elif st == LEVEL:                                        # stay on our side of it
                    if side[i]:
                        lo = max(lo, db)
                    else:
                        hi = min(hi, da)
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


# ======================================================================================= feasibility
def plan_lat_accel(ego: EgoState, waypoints: np.ndarray, v_star: float, brake: float) -> float:
    """Peak lateral acceleration (m/s^2) the spline through ``waypoints`` asks for. A quintic segment moving dd
    over ds (zero slope / curvature at both ends) peaks at 5.77 |dd| / ds^2 in d''(s), times v^2. The speed for a
    segment is what the car can be doing at its start: braking at ``brake`` from its current speed, but not below
    min(current speed, v*) (generous: the slower of the two is assumed reachable). The first segment's initial
    slope is ignored."""
    v0 = max(ego.v, 0.0)
    floor = min(v0, v_star)
    pts = np.vstack([[ego.s, ego.d], np.asarray(waypoints, float).reshape(-1, 2)])
    peak = 0.0
    for (s0, d0), (s1, d1) in zip(pts[:-1], pts[1:]):
        v = np.sqrt(max(v0**2 - 2.0 * brake * (s0 - ego.s), floor**2))
        peak = max(peak, 5.77 * abs(d1 - d0) * v**2 / max(s1 - s0, 1e-6) ** 2)
    return float(peak)


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
        self.lat_accel_limit = float(m["mpc"]["max_lat_accel"])
        self.waypoints = np.zeros((0, 2))   # the k waypoints in use (s, d)
        self.lat_accel = 0.0                # peak lateral acceleration the current plan asks for
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
        self.lat_accel = 0.0
        self.tube = None

    def decide(self, action):
        slots, k, v_star = self.decode(action)
        self.waypoints = slots[:k]
        self.lat_accel = plan_lat_accel(self.env.ego, self.waypoints, v_star, self.corridor.brake)
        self.mpc.set_speed_limit(v_star)
        self.reference = spline_reference(self.env.ego, self.env.road, self.waypoints, v_star,
                                          ds=self.spline_cfg["ds"], extend=self.spline_cfg["extend"])
        return {"vector": np.concatenate([slots.ravel(), [v_star, k]]), "k": k, "v_target": v_star,
                "waypoints": self.waypoints, "slots": slots, "lat_accel": self.lat_accel,
                "feasible": self.lat_accel <= self.lat_accel_limit}

    def control(self):
        env = self.env
        ego, objects, road = env.ego, env.objects, env.road
        infeasible = self.lat_accel > self.lat_accel_limit
        if infeasible and self.ebrake_ends:     # the RL asked for the undrivable: end the run now (failed run)
            self.tube = None
            r = self.reference.at_s(ego.s)
            return self.mpc.brake(ego), {"status": f"infeasible_plan:{self.lat_accel:.1f}m/s2", "ref_v": ego.v,
                                         "infeasible_terminal": True, "tracking_error": abs(r["d"] - ego.d),
                                         "ref_d": r["d"], "ref_psi": r["psi"]}
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
        diag.update(tracking_error=abs(r["d"] - ego.d), ref_d=r["d"], ref_psi=r["psi"], blocker=blocker,
                    plan_infeasible=infeasible)
        if self.ebrake_ends and (blocker is not None or diag.get("fallback_terminal")):
            diag["ebrake_terminal"] = True     # the env ends the episode after this step (collision penalty)
        return u, diag
