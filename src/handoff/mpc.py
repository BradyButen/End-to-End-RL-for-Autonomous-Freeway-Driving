"""M3's model-predictive controller: CasADi + IPOPT, built once, warm-started every control step.

    step(ego, tube) -> [steering, throttle_brake], diagnostics

Model: kinematic bicycle about the CG, state [x, y, psi, v, delta_act], input [delta_cmd (rad), a (m/s^2)],
beta = atan(lr / (lf + lr) * tan delta_act); the wheel angle follows the command through a first-order lag
(steer_lag, measured), RK4 over dt. Horizon N stages of dt.

Cost (stages 1..N, last stage x terminal_factor), tracking the tube's trajectory (the spline at the v* ramp):
    w_lat e_lat^2 + w_head (psi - psi_ref)^2 + w_speed (v - v_ref)^2           tracking
    + w_steer delta^2 + w_accel a^2 + w_steer_rate d_delta^2 + w_accel_rate d_a^2  effort and rates
    + w_yaw_rate ((psi_k - psi_k-1) / dt)^2                                     damps heading oscillation
    + barrier on the tube's left / right faces (see ``_barrier``)
    + l1 sum(sigma) + l2 sum(sigma^2)                                          corridor violation (soft_corridor)
Constraints: tube A_k p_k <= b_k + sigma_k, sigma_k >= 0 (SOFT: 4 half-spaces per stage on the ego centre, each
with a slack; with l1 above the constraint multipliers this is an exact penalty, i.e. the hard corridor whenever
that is feasible, and the least-violating plan when it is not: car already outside a face, a cut-in inside stopping
distance), |delta| <= delta_max, a in accel_bounds, |d_delta| <= steer_rate_limit dt, 0 <= v <= v_limit(t),
|v^2 tan(delta) / L| <= max_lat_accel.
Speed ceiling: the RL's v* (``set_speed_limit``), ramped down from the current speed at ceiling_brake m/s^2 so it
never makes the problem infeasible by itself.

Failures (solve time is process CPU time; a successful solve is always used, however long it took). With the soft
corridor the problem is essentially always feasible; what remains is timeouts / numerical failures:
  * retry once in the same step from a full-braking initial guess along the planned path (the first solve after a
    car becomes unavoidable can need ~250 IPOPT iterations from the previous, cruising plan, ~35 from a braking one;
    offline: stopped car 25 m ahead at 72 km/h, 346 ms -> 49 ms);
  * replay the previous plan for up to replay_steps.timeout (.infeasible) steps;
  * then (or with no plan yet) "hold_lane": zero acceleration (the actuation table's hold-speed command), steering
    that holds the road heading and pulls to the nearest lane centre, until a solve succeeds again. Never ends the
    episode and never brakes (no emergency brake in the MPC).
Emergency brake (``brake``): full brake with the same steering law. Only M3's gate uses it (ebrake.gate, off by
default) and the last step of an infeasible_plan episode.
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from handoff.sim import ActuationMap, EgoState, RoadFrame, wrap_angle

_SUCCESS = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
NX = 5   # x, y, psi, v, delta_act


class MPC:
    def __init__(self, cfg: dict, actuation: ActuationMap, *, lf: float, lr: float, max_steering_rad: float,
                 v_cap: float):
        self.cfg = cfg
        self.actuation = actuation
        self.N = int(cfg["horizon"])
        self.dt = float(cfg["dt"])
        self.lf, self.lr = lf, lr
        self.delta_max = max_steering_rad
        self.tau = float(cfg["steer_lag"])
        self.v_cap = v_cap
        self.ceiling_brake = float(cfg["ceiling_brake"])
        self.replay = {"infeasible": int(cfg["replay_steps"]["infeasible"]),
                       "timeout": int(cfg["replay_steps"]["timeout"])}
        self.constraints_on = True   # validation switch: False replaces the tube by always-true half-spaces
        self.road: RoadFrame | None = None
        self._build()
        self.reset(None)

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

    def _barrier(self, gap):
        """Cost of the ego centre being ``gap`` m inside a lateral tube face. Two zones, each 0 with zero slope at
        its start: outer (gap < dist, 0.5 m) a gentle log, inner (gap < inner_dist, 0.25 m) growing like 1/gap^2.
        Defaults (weight 1, inner_weight 10): gap 0.40 -> 0.02, 0.25 -> 0.19, 0.15 -> 4.9, 0.10 -> 23, 0.05 -> 161.
        gap is floored at 1 mm so the terms stay finite if a solver iterate crosses the face."""
        b = self.cfg["barrier"]
        g = ca.fmax(gap, 1e-3)
        r = g / float(b["dist"])
        outer = float(b["weight"]) * ca.if_else(r < 1, -ca.log(r) + r - 1, 0)
        d_in = float(b["inner_dist"])
        inner = float(b["inner_weight"]) * ca.if_else(g < d_in, (d_in / g - 1) ** 2, 0)
        return outer + inner

    def _build(self):
        N, w = self.N, self.cfg["weights"]
        X = ca.SX.sym("X", NX, N + 1)
        U = ca.SX.sym("U", 2, N)
        x0 = ca.SX.sym("x0", NX)
        u_prev = ca.SX.sym("u_prev", 2)
        ref = ca.SX.sym("ref", 4, N + 1)          # x_ref, y_ref, psi_ref, v_ref per stage
        tube = ca.SX.sym("tube", 12, N + 1)       # per stage: 4 half-spaces x (a_x, a_y, b)
        Sig = ca.SX.sym("Sig", 4, N)              # corridor slack per stage 1..N and half-space (m, >= 0)
        P = ca.vertcat(x0, u_prev, ca.vec(ref), ca.vec(tube))
        soft = self.cfg["soft_corridor"]

        J = 0
        g, lbg, ubg = [X[:, 0] - x0], [0.0] * NX, [0.0] * NX
        rate = float(self.cfg["steer_rate_limit"]) * self.dt
        a_lat = float(self.cfg["max_lat_accel"])
        for k in range(N):
            g.append(X[:, k + 1] - self._rk4(X[:, k], U[:, k]))
            lbg += [0.0] * NX
            ubg += [0.0] * NX
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            g.append(du[0])                                                 # steering rate
            lbg.append(-rate)
            ubg.append(rate)
            g.append(X[3, k] ** 2 * ca.tan(U[0, k]) / (self.lf + self.lr))  # lateral acceleration
            lbg.append(-a_lat)
            ubg.append(a_lat)
            J += w["steer"] * U[0, k] ** 2 + w["accel"] * U[1, k] ** 2
            J += w["steer_rate"] * du[0] ** 2 + w["accel_rate"] * du[1] ** 2

        for k in range(1, N + 1):
            fac = w["terminal_factor"] if k == N else 1.0
            xr, yr, pr, vr = ref[0, k], ref[1, k], ref[2, k], ref[3, k]
            e_lat = -ca.sin(pr) * (X[0, k] - xr) + ca.cos(pr) * (X[1, k] - yr)
            J += fac * (w["lateral"] * e_lat**2 + w["heading"] * (X[2, k] - pr) ** 2 + w["speed"] * (X[3, k] - vr) ** 2)
            for j in range(4):   # <= slack (0 inside the box)
                g.append(tube[3 * j, k] * X[0, k] + tube[3 * j + 1, k] * X[1, k] - tube[3 * j + 2, k] - Sig[j, k - 1])
                lbg.append(-ca.inf)
                ubg.append(0.0)
        J += float(soft["l1"]) * ca.sum1(ca.vec(Sig)) + float(soft["l2"]) * ca.sumsqr(Sig)
        for k in range(1, N + 1):
            J += w["yaw_rate"] * ((X[2, k] - X[2, k - 1]) / self.dt) ** 2
        for k in range(1, N + 1):
            for j in (2, 3):     # left / right faces: distance of the ego centre inside the face
                J += self._barrier(tube[3 * j + 2, k] - (tube[3 * j, k] * X[0, k] + tube[3 * j + 1, k] * X[1, k]))

        s = self.cfg["solver"]
        opts = {"ipopt.print_level": int(s.get("print_level", 0)), "print_time": False,
                "ipopt.max_iter": int(s["max_iter"]), "ipopt.max_cpu_time": float(s["max_cpu_time"]),
                "ipopt.tol": float(s["tol"]), "ipopt.acceptable_tol": float(s["tol"]) * 10,
                "ipopt.warm_start_init_point": "yes", "ipopt.sb": "yes", "expand": True}
        nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(Sig)), "p": P, "f": J, "g": ca.vertcat(*g)}
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self.lbg, self.ubg = np.array(lbg, dtype=float), np.array(ubg, dtype=float)

        self.nX, self.nU, self.nS = NX * (N + 1), 2 * N, 4 * N
        lbx = np.full(self.nX + self.nU + self.nS, -np.inf)
        ubx = np.full_like(lbx, np.inf)
        lbx[3:self.nX:NX], ubx[3:self.nX:NX] = 0.0, self.v_cap
        lbx[3], ubx[3] = -np.inf, np.inf      # initial state is fixed by the equality (measured v may be < 0)
        a_lo, a_hi = self.cfg["accel_bounds"]
        lbx[self.nX:self.nX + self.nU:2], ubx[self.nX:self.nX + self.nU:2] = -self.delta_max, self.delta_max
        lbx[self.nX + 1:self.nX + self.nU:2], ubx[self.nX + 1:self.nX + self.nU:2] = a_lo, a_hi
        lbx[self.nX + self.nU:] = 0.0             # slacks >= 0
        self.lbx, self.ubx = lbx, ubx

    # ------------------------------------------------------------------ runtime
    def reset(self, road: RoadFrame | None) -> None:
        self.road = road
        self.v_limit = self.v_cap
        self.u_prev = np.zeros(2)      # [delta, a] last applied
        self.delta_act = 0.0           # estimated actual wheel angle (lag-filtered commands)
        self._clear_plan()
        self._eb_prev = None           # previous heading error of the emergency-brake law

    def _clear_plan(self) -> None:
        self._prev_X = self._prev_U = self._prev_S = None
        self._prev_lam = None          # (lam_g, lam_x) of the last successful solve
        self._shift = 0                # steps the stored plan has been replayed without a fresh solve

    def set_speed_limit(self, v: float | None) -> None:
        """Speed ceiling (m/s); None = v_cap."""
        self.v_limit = self.v_cap if v is None else float(np.clip(v, 0.0, self.v_cap))

    def step(self, ego: EgoState, tube) -> tuple[np.ndarray, dict]:
        t0, c0 = time.perf_counter(), time.process_time()
        p, ref = self._params(ego, tube)
        ubx = self._ubx(ego)
        sol, status, ok = self._solve(self._initial_guess(ego, ref), p, ubx)
        retried = False
        if not ok:                                 # second try from a braking guess (no multiplier warm start)
            self._prev_lam = None
            sol, status2, ok = self._solve(self._brake_guess(ego, tube), p, ubx)
            status, retried = (status2 if ok else status), True
        diag = {"solve_time": time.perf_counter() - t0, "cpu_time": time.process_time() - c0, "status": status,
                "fallback": not ok, "fallback_reason": "", "fallback_mode": "", "fallback_terminal": False,
                "v_limit": self.v_limit, "ref_v": float(ref[3, 1]), "corridor_violation": 0.0,
                "retried": retried}
        if ok:
            w = np.asarray(sol["x"]).ravel()
            self._prev_X = w[:self.nX].reshape(self.N + 1, NX).T
            self._prev_U = w[self.nX:self.nX + self.nU].reshape(self.N, 2).T
            self._prev_S = w[self.nX + self.nU:]
            diag["corridor_violation"] = float(self._prev_S.max())    # m, largest slack used in the plan
            self._prev_lam = (np.asarray(sol["lam_g"]).ravel(), np.asarray(sol["lam_x"]).ravel())
            self._shift = 0
            delta, acc = self._prev_U[:, 0]
            diag["pred_xy"] = self._prev_X[:2].T.copy()
            return self._apply(delta, acc), diag

        if "CpuTime" in status or "Maximum_Iterations" in status:
            reason = "timeout"
        elif "Infeasible" in status:
            reason = "infeasible"
        else:
            reason = "solver_failure"
        diag["fallback_reason"] = reason
        self._prev_lam = None
        budget = self.replay["infeasible" if reason == "infeasible" else "timeout"]
        if self._prev_U is not None and self._shift < min(budget, self.N - 1):
            self._shift += 1                     # replay the previous plan one step further
            delta, acc = self._prev_U[:, self._shift]
            diag["fallback_mode"] = "shifted_previous"
            diag["pred_xy"] = self._prev_X[:2, self._shift:].T.copy()
            return self._apply(delta, acc), diag
        diag["fallback_mode"] = "hold_lane"     # no plan left: hold speed and lane until a solve succeeds
        diag["pred_xy"] = np.array([[ego.x, ego.y]])
        return self.hold_lane(ego), diag

    def brake(self, ego: EgoState) -> np.ndarray:
        """Emergency brake: full brake with the lane-holding steering law (``_lane_steer``)."""
        return self._lane_steer(ego, -1.0)

    def hold_lane(self, ego: EgoState) -> np.ndarray:
        """Zero acceleration (hold speed) with the lane-holding steering law: the MPC fallback."""
        return self._lane_steer(ego, float(np.clip(self.actuation.accel_to_command(0.0), -1.0, 1.0)))

    def _lane_steer(self, ego: EgoState, throttle_brake: float) -> np.ndarray:
        """Steering u = k_h e_psi + k_r (e_psi - e_psi_prev) + k_l (d_lane - d), with e_psi = road heading - ego
        heading and d_lane the nearest lane centre (gains ``ebrake_steer``), and the given throttle_brake. Clears the
        warm start and keeps u_prev / the steering-lag estimate consistent for when the MPC takes over again."""
        kh, kr, kl = (float(x) for x in self.cfg["ebrake_steer"])
        road = self.road
        lane_d = road.lane_centers[road.nearest_lane(ego.d)]
        e_psi = float(wrap_angle(road.heading_at(ego.s) - ego.psi))
        de = 0.0 if self._eb_prev is None else e_psi - self._eb_prev
        self._eb_prev = e_psi
        u = np.array([float(np.clip(kh * e_psi + kr * de + kl * (lane_d - ego.d), -1.0, 1.0)), throttle_brake])
        self._clear_plan()
        self.u_prev = np.array([u[0] * self.delta_max, self.actuation.command_to_accel(u[1])])
        self._advance_lag(self.u_prev[0])
        return u

    # ------------------------------------------------------------------ helpers
    def _apply(self, delta: float, acc: float) -> np.ndarray:
        """Apply an MPC input (fresh or replayed plan)."""
        self._eb_prev = None      # the e-brake law's derivative restarts at the next emergency brake
        self.u_prev = np.array([delta, acc])
        self._advance_lag(delta)
        return np.array([float(np.clip(delta / self.delta_max, -1.0, 1.0)),
                         float(np.clip(self.actuation.accel_to_command(acc), -1.0, 1.0))])

    def _advance_lag(self, delta_cmd: float) -> None:
        """Propagate the wheel-angle lag estimate over one control step (exact for a constant input)."""
        a = np.exp(-self.dt / self.tau)
        self.delta_act = a * self.delta_act + (1 - a) * float(np.clip(delta_cmd, -self.delta_max, self.delta_max))

    def _params(self, ego: EgoState, tube):
        N = self.N
        tr = tube.traj[: N + 1]
        ref = np.zeros((4, N + 1))
        ref[0], ref[1] = tr[:, 3], tr[:, 4]
        ref[2] = ego.psi + wrap_angle(tr[:, 5] - ego.psi)
        ref[3] = tr[:, 2]
        half = np.concatenate([tube.A[: N + 1], tube.b[: N + 1, :, None]], axis=2)   # (N+1, 4, 3)
        if not self.constraints_on:
            half = np.zeros_like(half)
            half[:, :, 2] = 1e6
        x0 = np.array([ego.x, ego.y, ego.psi, ego.v, self.delta_act])
        return np.concatenate([x0, self.u_prev, ref.T.ravel(), half.ravel()]), ref

    def _initial_guess(self, ego: EgoState, ref: np.ndarray) -> np.ndarray:
        if self._prev_X is not None:
            sh = self._shift + 1      # the plan may already have been replayed by fallback steps
            X = np.concatenate([self._prev_X[:, sh:], np.repeat(self._prev_X[:, -1:], sh, axis=1)], axis=1)
            U = np.concatenate([self._prev_U[:, sh:], np.repeat(self._prev_U[:, -1:], sh, axis=1)], axis=1)
        else:
            X = np.zeros((NX, self.N + 1))
            X[:4] = ref
            U = np.zeros((2, self.N))
        X[:, 0] = [ego.x, ego.y, ego.psi, ego.v, self.delta_act]
        S = np.zeros((self.N, 4))
        if self._prev_S is not None:          # shifted like X / U (vec(Sig) is stage-major: 4 per stage)
            sh = self._shift + 1
            P = self._prev_S.reshape(self.N, 4)
            S = np.concatenate([P[sh:], np.repeat(P[-1:], sh, axis=0)], axis=0)
        return np.concatenate([X.T.ravel(), U.T.ravel(), S.ravel()])

    def _brake_guess(self, ego: EgoState, tube) -> np.ndarray:
        """Initial guess: full braking (accel_bounds[0]) along the tube's path, wheel straight."""
        a_lo = float(self.cfg["accel_bounds"][0])
        t = np.arange(self.N + 1) * self.dt
        v = np.maximum(max(ego.v, 0.0) + a_lo * t, 0.0)
        s = ego.s + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * self.dt)])
        tr = tube.traj                            # [s, d, v, x, y, psi] per stage, s increasing
        X = np.zeros((NX, self.N + 1))
        X[0], X[1] = np.interp(s, tr[:, 0], tr[:, 3]), np.interp(s, tr[:, 0], tr[:, 4])
        X[2] = ego.psi + wrap_angle(np.interp(s, tr[:, 0], tr[:, 5]) - ego.psi)
        X[3] = v
        X[:, 0] = [ego.x, ego.y, ego.psi, ego.v, self.delta_act]
        U = np.zeros((2, self.N))
        U[1] = np.where(v[:-1] > 0.0, a_lo, 0.0)
        return np.concatenate([X.T.ravel(), U.T.ravel(), np.zeros(self.nS)])

    def _ubx(self, ego: EgoState) -> np.ndarray:
        """Upper bounds with the speed ceiling ramped down from the current speed (always reachable)."""
        ub = self.ubx.copy()
        if self.v_limit < self.v_cap:
            k = np.arange(1, self.N + 1)
            ramp = max(ego.v, 0.0) - self.ceiling_brake * self.dt * k
            ub[3 + NX * k] = np.minimum(self.v_cap, np.maximum(self.v_limit, ramp))
        return ub

    def _solve(self, x_init, p, ubx):
        kwargs = dict(x0=x_init, p=p, lbx=self.lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        if self._prev_lam is not None:
            kwargs["lam_g0"], kwargs["lam_x0"] = self._prev_lam
        try:
            sol = self.solver(**kwargs)
            stats = self.solver.stats()
            status = stats.get("return_status", "unknown")
            return sol, status, bool(stats.get("success", False)) or status in _SUCCESS
        except RuntimeError as e:
            return None, f"exception:{e}"[:60], False
