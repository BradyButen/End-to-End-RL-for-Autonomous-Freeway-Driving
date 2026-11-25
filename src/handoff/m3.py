"""M3: RL + MPC. Every 10 control steps (1 s) the RL picks a lane command and a speed command; every control step
the pipeline below turns them into controls. The controller owns the trajectory, the speed under the maximum and
collision avoidance: it drives toward the maximum speed as fast as the corridor allows.

    action    MultiDiscrete [3, 3] = [lane command, speed command], the same action as M2; [0, 0] = keep lane,
              hold speed (the default)
      lane    KEEP / LEFT / RIGHT moves the target lane (reference.LaneTarget: one lane at a time, saturating at the
              edges, ignored while a lane change is in progress)
      speed   HOLD / FASTER / SLOWER moves the maximum speed v* by 10 km/h (reference.SpeedTarget; starts at the
              spawn speed): the MPC's speed ceiling
    plan      a new reference.LanePlan whenever the target lane changes: a quintic d(s) from the ego to the target
              lane centre over lane_plan.duration (2.5 s) of travel, lengthened to stay within its max_lat_accel;
              flat after
    corridor  one road-aligned box per MPC stage around the trajectory (the plan at a speed ramp to v*): lateral
              faces at the outer lane centres, cut by the cars' keep-out zones (``Corridor``)
    MPC       mpc.py: tracks the trajectory inside the corridor (a soft constraint with an exact penalty, so it holds
              whenever it can) under the speed ceiling v*. Braking for cars is the MPC's job

An MPC solve failure never ends the episode: the previous plan is replayed, then the car holds speed and lane until
a solve succeeds (mpc.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from handoff.env import Method
from handoff.mpc import MPC
from handoff.reference import LanePlan, LaneTarget, Reference, SpeedTarget
from handoff.sim import ActuationMap, EgoState, ObjectList, RoadFrame, frenet_velocity, wrap_angle


# ======================================================================================= corridor
@dataclass
class Tube:
    """One box per MPC stage (stage k at t = k dt), as 4 half-spaces on the ego centre: A[k] @ p <= b[k]."""
    A: np.ndarray        # (N+1, 4, 2) normals (world): +along, -along, +left, -left
    b: np.ndarray        # (N+1, 4)
    traj: np.ndarray     # (N+1, 6) the stage points [s, d, v, x, y, psi] (what the MPC tracks)
    extents: np.ndarray  # (N+1, 4) [back, front, right, left] face offsets from the stage point (road frame, m)


class Corridor:
    """Road-aligned corridor on the ego centre, rebuilt every control step.

    The stage points follow the reference at a speed ramp toward v* (+accel / -brake m/s^2), ignoring cars:
    stopping for a car is the MPC's job, through the corridor's faces. Cars ahead and alongside count; a car behind
    counts only while its front bumper is within ``rear_range`` of the ego's rear bumper.
      * Lateral faces: the outer lane centres, pushed out by -``inset`` (inset -0.3 gives d in [-7.3, +0.3]).
      * Keep-out zone of a car: its footprint + ``keepout_s`` fore / aft + ``margin_d`` sideways, predicted at
        constant velocity and grown by half the ego's size (the box bounds the ego's centre, so this keeps the body
        out of the zone).
      * Which face a car adds is decided stage by stage, comparing the stage point with the car's predicted zone at
        that same stage. Each car carries a state through the horizon, starting from where it is now:
          BEHIND it: a front face at the zone's rear while the stage point overlaps it laterally;
          LEVEL, beside it: a lateral face on the stage point's side;
          AHEAD of it: a back face at the zone's front while overlapping laterally.
        BEHIND becomes LEVEL only at a stage where the stage point draws level while laterally clear (passing). A
        plan that runs into the car stays BEHIND for the rest of the horizon, so it cannot drive through it. LEVEL
        becomes AHEAD once the stage point is past the zone, so a plan that overtakes a slower car and merges in
        front of it gets a back face there rather than a front face.
    A missing front / back face is set to +-BIG.
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
        """Stage points [s, d, v, x, y, psi]: the reference path at a speed ramp from the ego's speed to the
        reference speed."""
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
        zones = []      # per car: predicted centre (s(t), d(t)), half-sizes on the ego centre, heading vs the road
        ego_rear = ego.s - 0.5 * ego.length
        for o in objects:
            if o.s + 0.5 * o.length < ego_rear - self.rear_range:   # behind the ego beyond rear_range: ignored
                continue
            sdot, ddot = frenet_velocity(o, road)
            zl, zw = self.keepout_half(o)
            zones.append((o.s + sdot * t, o.d + ddot * t, zl + 0.5 * self.L, zw + 0.5 * self.W,
                          float(wrap_angle(o.psi - road.heading_at(o.s)))))
        # Each car's state starts from where it is relative to the ego now, then advances stage by stage, always
        # comparing the car's predicted zone with the stage point at the same time.
        BEHIND, LEVEL, AHEAD = 0, 1, 2      # the stage point is behind / level with / ahead of the car's zone
        state, side = [], []
        for so, do, hl, _, _ in zones:
            rear_now, front_now = so[0] - hl, so[0] + hl
            state.append(BEHIND if rear_now >= ego.s else AHEAD if front_now <= ego.s else LEVEL)
            side.append(ego.d >= do[0])     # True: the ego is on the car's left
        d_lo0 = min(road.lane_centers) + self.inset
        d_hi0 = max(road.lane_centers) - self.inset
        A, b = np.zeros((K + 1, 4, 2)), np.zeros((K + 1, 4))
        extents = np.zeros((K + 1, 4))
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
        return Tube(A=A, b=b, traj=traj, extents=extents)


# ======================================================================================= method
class M3(Method):
    def __init__(self, cfg: dict, env):
        m = cfg["method"]
        super().__init__(env, period=m["decision_period"], action_space=self.ACTION_SPACE)
        self.lane = LaneTarget(cfg["lane_action"])
        self.speed = SpeedTarget(cfg["lane_action"], env.v_cap)
        self.plan_cfg = m["lane_plan"]
        self.plan: LanePlan | None = None
        self.spline_cfg = m["spline"]
        ego = cfg["sim"]["ego"]
        self.mpc = MPC(m["mpc"], ActuationMap(cfg["actuation"]["table"]), lf=ego["front_wheelbase"],
                       lr=ego["rear_wheelbase"], max_steering_rad=env.max_steering_rad, v_cap=env.v_cap)
        self.corridor = Corridor(m["corridor"], self.mpc.N, self.mpc.dt, ego["length"], ego["width"], env.v_cap)
        self.lat_accel = 0.0                # peak lateral acceleration the current plan asks for (viewer)
        self.tube: Tube | None = None       # this step's corridor (viewer)

    ACTION_SPACE = gym.spaces.MultiDiscrete([3, 3])   # [lane command, speed command]
    COMMAND_FEATURES = ("target_speed",)   # observation "cmd": max speed v* / v_max (the target lane is in "ego")

    def target_lane_offset(self) -> int:
        return self.lane.offset(self.env.ego, self.env.road)

    def command_state(self) -> list[float]:
        return [self.speed.v / self.env.v_max]

    def reset(self):
        super().reset()
        self.mpc.reset(self.env.road)
        self.lane.reset(self.env.ego, self.env.road)
        self.speed.reset(self.env.ego)
        self.plan = None
        self.lat_accel = 0.0
        self.tube = None

    def decide(self, action):
        cmd, speed_cmd = (int(x) for x in np.asarray(action).ravel())
        env = self.env
        ego, road = env.ego, env.road
        if self.lane.update(cmd, ego, road) or self.plan is None:
            self.plan = LanePlan.build(ego, road.lane_centers[self.lane.lane], self.plan_cfg)
        v_star = self.speed.update(speed_cmd)
        self.mpc.set_speed_limit(v_star)
        p = self.plan
        self.reference = p.reference(ego, road, v_star, float(self.spline_cfg["ds"]), float(self.plan_cfg["ahead"]))
        self.lat_accel = p.peak_lat_accel(ego.v)
        return {"vector": np.array([cmd, self.lane.lane, p.d1, v_star]), "lane": self.lane.lane,
                "d_target": p.d1, "v_target": v_star, "lat_accel": self.lat_accel,
                "speed_cmd": speed_cmd}

    def control(self):
        env = self.env
        ego = env.ego
        self.tube = self.corridor.build(ego, self.reference, env.objects, env.road)
        u, diag = self.mpc.step(ego, self.tube)
        r = self.reference.at_s(ego.s)
        diag.update(tracking_error=abs(r["d"] - ego.d), ref_d=r["d"], ref_psi=r["psi"])
        return u, diag
