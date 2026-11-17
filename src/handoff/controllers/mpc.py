"""M3 model-predictive controller (spec §7.3). CasADi + IPOPT, built once, warm-started every step.

Model: kinematic bicycle about the CG, state [x, y, ψ, v, δ_act], input [δ_cmd (wheel angle, rad), a (m/s²)],
β = atan(l_r/(l_f+l_r) · tan δ_act); the wheel angle follows the command through a first-order lag
δ_act' = (δ_cmd − δ_act)/τ (measured MetaDrive yaw response: ~80 % after 0.2 s, DECISIONS D-12); RK4 over dt.

Corridor (DECISIONS D-13): cars and road edges enter as HARD constraints through a convex tube — one box per
stage, 4 half-spaces on the ego centre (cars inflated by the ego footprint), built by ``CorridorBuilder``
around an initial trajectory (spline reference + speed profile that stops behind blocking cars). The tube is
rebuilt at a slower rate than the MPC (every ``corridor.rebuild_period`` control steps, and whenever the RL
reference changes); between rebuilds the MPC uses the tube's stages shifted by the elapsed steps.

Cost (stages 1..N, terminal ×terminal_factor), tracking the tube's initial trajectory:
    w_lat·e_lat² + w_head·(ψ−ψ_ref)² + w_speed·(v − v_ref)² + w_steer·δ² + w_accel·a²
    + w_steer_rate·Δδ² + w_accel_rate·Δa²
with e_lat the signed offset from the trajectory tangent at the stage's reference point and v_ref the initial
trajectory's speed (global target speed unless slowing for a blocking car).

Constraints: tube A_k·p_k ≤ b_k (hard), |δ| ≤ δ_max, a ∈ accel_bounds, |Δδ| ≤ steer_rate·dt, 0 ≤ v ≤ v_cap,
|v²·tan δ / (l_f+l_r)| ≤ a_lat_max.

Failure handling (spec §7.3): if the solve fails and the tube was not just built, rebuild the tube from the
current state and retry once; then → next input of the shifted previous plan (at most ``max_fallback_shift``
steps); none left → emergency brake (throttle_brake = −1) with a lane-holding steering law. Every fallback is
reported in the diagnostics with its reason.
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from handoff.controllers.actuation import ActuationMap
from handoff.controllers.base import Reference
from handoff.controllers.corridor import CorridorBuilder, Tube
from handoff.sim.road import RoadFrame, wrap_angle
from handoff.sim.state import EgoState, ObjectList

_SUCCESS = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}


class MPCController:
    def __init__(self, cfg: dict, road: RoadFrame | None, actuation: ActuationMap, *, lf: float, lr: float,
                 ego_length: float, ego_width: float, max_steering_rad: float, v_cap: float, v_target: float):
        self.cfg = cfg
        self.road = road
        self.actuation = actuation
        self.N = int(cfg["horizon"])
        self.dt = float(cfg["dt"])
        self.lf, self.lr = lf, lr
        self.L, self.W = ego_length, ego_width
        self.delta_max = max_steering_rad
        self.tau = float(cfg.get("steer_lag", 0.1))
        self.a_lat_max = float(cfg.get("max_lat_accel", 4.0))
        self.max_shift = int(cfg.get("max_fallback_shift", 5))
        self.NX = 5
        self.v_cap = v_cap
        self.v_target = v_target
        self.corridor = CorridorBuilder(cfg["corridor"], ego_length, ego_width, self.dt, v_target, v_cap)
        self.rebuild_period = int(cfg["corridor"]["rebuild_period"])
        if self.corridor.K < self.N + self.rebuild_period:
            raise ValueError("corridor.horizon must cover the MPC horizon plus the rebuild period")
        self._build()
        self.reset()

    # ------------------------------------------------------------------ problem construction
    def _dynamics(self, x, u):
        beta = ca.atan(self.lr / (self.lf + self.lr) * ca.tan(x[4]))
        return ca.vertcat(x[3] * ca.cos(x[2] + beta), x[3] * ca.sin(x[2] + beta), x[3] / self.lr * ca.sin(beta),
                          u[1], (u[0] - x[4]) / self.tau)

    def _rk4(self, x, u):
        dt = self.dt
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + dt / 2 * k1, u)
        k3 = self._dynamics(x + dt / 2 * k2, u)
        k4 = self._dynamics(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build(self):
        N, NX = self.N, self.NX
        w = self.cfg["weights"]
        X = ca.SX.sym("X", NX, N + 1)
        U = ca.SX.sym("U", 2, N)
        x0 = ca.SX.sym("x0", NX)
        u_prev = ca.SX.sym("u_prev", 2)
        ref = ca.SX.sym("ref", 4, N + 1)          # x_ref, y_ref, psi_ref, v_ref
        tube = ca.SX.sym("tube", 12, N + 1)       # per stage: 4 × (a_x, a_y, b)
        P = ca.vertcat(x0, u_prev, ca.vec(ref), ca.vec(tube))

        J = 0
        g, lbg, ubg = [], [], []
        g.append(X[:, 0] - x0)
        lbg += [0.0] * NX
        ubg += [0.0] * NX
        rate = self.cfg["steer_rate_limit"] * self.dt
        for k in range(N):
            g.append(X[:, k + 1] - self._rk4(X[:, k], U[:, k]))
            lbg += [0.0] * NX
            ubg += [0.0] * NX
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            g.append(du[0])
            lbg.append(-rate)
            ubg.append(rate)
            # lateral-acceleration bound on the commanded wheel angle at the stage speed
            g.append(X[3, k] ** 2 * ca.tan(U[0, k]) / (self.lf + self.lr))
            lbg.append(-self.a_lat_max)
            ubg.append(self.a_lat_max)
            J += w["steer"] * U[0, k] ** 2 + w["accel"] * U[1, k] ** 2
            J += w["steer_rate"] * du[0] ** 2 + w["accel_rate"] * du[1] ** 2

        for k in range(1, N + 1):
            fac = w["terminal_factor"] if k == N else 1.0
            xr, yr, pr, vr = ref[0, k], ref[1, k], ref[2, k], ref[3, k]
            e_lat = -ca.sin(pr) * (X[0, k] - xr) + ca.cos(pr) * (X[1, k] - yr)
            J += fac * (w["lateral"] * e_lat**2 + w["heading"] * (X[2, k] - pr) ** 2
                        + w["speed"] * (X[3, k] - vr) ** 2)
            # corridor: 4 hard half-spaces on the ego centre
            for j in range(4):
                g.append(tube[3 * j, k] * X[0, k] + tube[3 * j + 1, k] * X[1, k] - tube[3 * j + 2, k])
                lbg.append(-ca.inf)
                ubg.append(0.0)

        wvec = ca.vertcat(ca.vec(X), ca.vec(U))
        nlp = {"x": wvec, "p": P, "f": J, "g": ca.vertcat(*g)}
        s = self.cfg["solver"]
        opts = {
            "ipopt.print_level": int(s.get("print_level", 0)),
            "print_time": False,
            "ipopt.max_iter": int(s["max_iter"]),
            "ipopt.max_cpu_time": float(s["max_cpu_time"]),
            "ipopt.tol": float(s["tol"]),
            "ipopt.acceptable_tol": float(s["tol"]) * 10,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.sb": "yes",
            "expand": True,
        }
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)
        nX, nU = NX * (N + 1), 2 * N
        self._idx = (nX, nU)
        lbx = np.full(nX + nU, -np.inf)
        ubx = np.full_like(lbx, np.inf)
        lbx[3:nX:NX] = 0.0
        ubx[3:nX:NX] = self.v_cap
        lbx[3] = -np.inf  # initial state fixed by equality, avoid conflict if measured v slightly < 0
        ubx[3] = np.inf
        a_lo, a_hi = self.cfg["accel_bounds"]
        lbx[nX:nX + nU:2] = -self.delta_max
        ubx[nX:nX + nU:2] = self.delta_max
        lbx[nX + 1:nX + nU:2] = a_lo
        ubx[nX + 1:nX + nU:2] = a_hi
        self.lbx, self.ubx = lbx, ubx

    # ------------------------------------------------------------------ runtime
    def reset(self) -> None:
        self._prev_X = None
        self._prev_U = None
        self._prev_lam_g = None
        self._prev_lam_x = None
        self._shift = 0  # how many steps the stored plan has been shifted without a fresh solve
        self.u_prev = np.zeros(2)  # [delta, a] last applied
        self.delta_act = 0.0       # estimated actual wheel angle (lag-filtered commands)
        self.tube: Tube | None = None
        self._tube_age = 0
        self._tube_ref = None

    def set_road(self, road: RoadFrame) -> None:
        self.road = road

    def _rebuild(self, ego: EgoState, reference: Reference, objects: ObjectList) -> None:
        self.tube = self.corridor.build(ego, reference, objects, self.road)
        self._tube_age = 0
        self._tube_ref = reference

    def _params(self, ego: EgoState):
        N, off, t = self.N, self._tube_age, self.tube
        idx = np.minimum(np.arange(N + 1) + off, len(t) - 1)
        tr = t.traj[idx]
        ref = np.zeros((4, N + 1))
        ref[0], ref[1] = tr[:, 3], tr[:, 4]
        ref[2] = ego.psi + wrap_angle(tr[:, 5] - ego.psi)
        ref[3] = tr[:, 2]
        tube = np.concatenate([t.A[idx], t.b[idx][:, :, None]], axis=2).reshape(N + 1, 12).T  # (12, N+1)
        x0 = np.array([ego.x, ego.y, ego.psi, ego.v, self.delta_act])
        p = np.concatenate([x0, self.u_prev, ref.T.ravel(), tube.T.ravel()])
        return p, ref, idx

    def _initial_guess(self, ego: EgoState, ref: np.ndarray):
        N = self.N
        if self._prev_X is not None:
            sh = self._shift + 1  # plan may already have been advanced by fallback steps
            X = np.concatenate([self._prev_X[:, sh:], np.repeat(self._prev_X[:, -1:], sh, axis=1)], axis=1)
            U = np.concatenate([self._prev_U[:, sh:], np.repeat(self._prev_U[:, -1:], sh, axis=1)], axis=1)
        else:
            X = np.zeros((self.NX, N + 1))
            X[0], X[1], X[2], X[3] = ref[0], ref[1], ref[2], ref[3]
            U = np.zeros((2, N))
        X[:, 0] = [ego.x, ego.y, ego.psi, ego.v, self.delta_act]
        return np.concatenate([X.T.ravel(), U.T.ravel()])

    def _solve(self, x_init, p, lam=None):
        kwargs = dict(x0=x_init, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        if lam is not None:
            kwargs["lam_g0"], kwargs["lam_x0"] = lam
        try:
            sol = self.solver(**kwargs)
            stats = self.solver.stats()
            status = stats.get("return_status", "unknown")
            ok = bool(stats.get("success", False)) or status in _SUCCESS
            return sol, status, ok
        except RuntimeError as e:
            return None, f"exception:{e}"[:60], False

    def step(self, ego: EgoState, reference: Reference, objects: ObjectList) -> tuple[np.ndarray, dict]:
        N = self.N
        rebuilt = False
        if self.tube is None or reference is not self._tube_ref or self._tube_age >= self.rebuild_period:
            self._rebuild(ego, reference, objects)
            rebuilt = True
        t0 = time.perf_counter()
        p, ref, idx = self._params(ego)
        x_init = self._initial_guess(ego, ref)
        lam = (self._prev_lam_g, self._prev_lam_x) if self._prev_lam_g is not None else None
        sol, status, ok = self._solve(x_init, p, lam)
        retried = False
        if not ok and not rebuilt:  # tube may be stale: rebuild from the current state and retry once
            self._rebuild(ego, reference, objects)
            rebuilt = retried = True
            p, ref, idx = self._params(ego)
            sol, status, ok = self._solve(self._initial_guess(ego, ref), p)
        solve_time = time.perf_counter() - t0
        timeout = solve_time > float(self.cfg["solver"]["max_cpu_time"]) * (3.0 if retried else 1.5) \
            or "CpuTime" in status
        self._tube_age += 1

        r_here = reference.at_s(ego.s)
        diag = {"solve_time": solve_time, "status": status, "fallback": False, "fallback_reason": "",
                "tracking_error": float(abs(r_here["d"] - ego.d)), "ref_d": r_here["d"],
                "ref_v": float(ref[3, 1]), "ref_psi": r_here["psi"], "n_objects": len(objects),
                "tube_rebuilt": rebuilt, "tube_retry": retried,
                "tube_corners": self.tube.corners[idx].copy(),
                "tube_blocked": int(self.tube.blocked[idx].sum())}
        if rebuilt:
            diag["tube_full"] = self.tube

        if ok and not timeout:
            wopt = np.asarray(sol["x"]).ravel()
            nX, nU = self._idx
            X = wopt[:nX].reshape(N + 1, self.NX).T
            U = wopt[nX:nX + nU].reshape(N, 2).T
            self._prev_X, self._prev_U = X, U
            self._prev_lam_g = np.asarray(sol["lam_g"]).ravel()
            self._prev_lam_x = np.asarray(sol["lam_x"]).ravel()
            self._shift = 0
            delta, acc = U[:, 0]
            diag["pred_xy"] = X[:2].T.copy()
        else:
            reason = "timeout" if timeout else ("infeasible" if "Infeasible" in status else "solver_failure")
            diag["fallback"] = True
            diag["fallback_reason"] = reason
            self._prev_lam_g = None
            self._prev_lam_x = None
            if self._prev_U is not None and self._shift + 1 < min(N, self.max_shift + 1):
                self._shift += 1
                delta, acc = self._prev_U[:, self._shift]
                diag["fallback_mode"] = "shifted_previous"
                diag["pred_xy"] = self._prev_X[:2, self._shift:].T.copy()
            else:
                diag["fallback_mode"] = "emergency_brake"
                self._prev_X = self._prev_U = None
                u = self.emergency_brake(ego)
                self.u_prev = np.array([u[0] * self.delta_max, self.actuation.command_to_accel(u[1])])
                self._advance_lag(self.u_prev[0])
                diag["pred_xy"] = np.array([[ego.x, ego.y]])
                return u, diag

        self.u_prev = np.array([delta, acc])
        self._advance_lag(delta)
        steering = float(np.clip(delta / self.delta_max, -1.0, 1.0))
        throttle = float(np.clip(self.actuation.accel_to_command(acc), -1.0, 1.0))
        return np.array([steering, throttle]), diag

    def _advance_lag(self, delta_cmd: float) -> None:
        """Propagate the wheel-angle lag estimate over one control step (exact for constant input)."""
        a = np.exp(-self.dt / self.tau)
        self.delta_act = a * self.delta_act + (1 - a) * float(np.clip(delta_cmd, -self.delta_max, self.delta_max))

    def emergency_brake(self, ego: EgoState) -> np.ndarray:
        lane_d = self.road.lane_centers[self.road.nearest_lane(ego.d)]
        steer = 1.7 * float(wrap_angle(self.road.heading_at(ego.s) - ego.psi)) + 0.3 * (lane_d - ego.d)
        return np.array([float(np.clip(steer, -1, 1)), -1.0])
