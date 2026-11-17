"""Controller unit tests against a kinematic plant (no MetaDrive)."""

import numpy as np
import pytest

from handoff.config import kmh
from handoff.controllers.mpc import MPCController
from handoff.controllers.pid import PID, CascadedPIDController
from handoff.controllers.safe_set import project_waypoints, project_waypoints_detailed
from handoff.controllers.spline import build_m3_reference
from handoff.controllers.trajectory import (
    build_m2_reference,
    eval_quintic,
    quintic_coefficients,
    speed_profile,
)
from handoff.sim.state import make_tracked_object


# ----------------------------------------------------------------------------- trajectory
def test_quintic_boundary_conditions():
    c = quintic_coefficients(0.0, 0.5, -0.2, -3.5, 2.0)
    d, dd, ddd = eval_quintic(c, np.array([0.0, 2.0, 3.0]), 2.0)
    assert d[0] == pytest.approx(0.0) and dd[0] == pytest.approx(0.5) and ddd[0] == pytest.approx(-0.2)
    assert d[1] == pytest.approx(-3.5)
    assert dd[1] == pytest.approx(0.0, abs=1e-9) and ddd[1] == pytest.approx(0, abs=1e-9)
    assert d[2] == pytest.approx(-3.5)  # held after T


def test_speed_profile_limits():
    t = np.arange(0, 6, 0.1)
    v, a, s = speed_profile(10.0, 20.0, t, 2.0, 4.0)
    assert v[-1] == pytest.approx(20.0) and np.all(np.diff(v) <= 2.0 * 0.1 + 1e-9)
    v2, _, _ = speed_profile(20.0, 0.0, t, 2.0, 4.0)
    assert v2[-1] == pytest.approx(0.0) and np.all(np.diff(v2) >= -4.0 * 0.1 - 1e-9)
    assert s[0] == 0.0 and np.all(np.diff(s) > 0)


def test_m2_reference_continuity(road, plant_cls, actuation):
    p = plant_cls(actuation, x=50, y=-3.5, v=18)
    ego = p.ego(road)
    ref = build_m2_reference(ego, road, 0.0, kmh(70), 2.0, 0.1, 2.0, 4.0)
    assert ref.d[0] == pytest.approx(ego.d) and ref.d[-1] == pytest.approx(0.0)
    assert ref.v[0] == pytest.approx(ego.v)
    assert ref.s[0] == pytest.approx(ego.s)


# ----------------------------------------------------------------------------- PID (spec §17.3 in miniature)
def test_pid_basic():
    pid = PID(1.0, 0.1, 0.0, integral_limit=1.0)
    assert pid(2.0) == pytest.approx(2.0 + 0.1 * 1.0)  # integral clamped to 1.0


def test_cascaded_pid_lane_change_and_speed(road, plant_cls, actuation, cfg_m2):
    m = cfg_m2["method"]
    ctrl = CascadedPIDController.from_config(m["pid"], actuation, 0.1)
    p = plant_cls(actuation, x=50, y=-3.5, v=15)
    max_cte = 0.0
    ref = None
    for k in range(120):
        ego = p.ego(road, t=k * 0.1)
        if k % m["decision_period"] == 0:
            lat = None if ref is None else (ref.at_time(0.5)["d_dot"], ref.at_time(0.5)["d_ddot"])
            ref = build_m2_reference(ego, road, 0.0, kmh(60), 2.0, 0.1, 2.0, 4.0, lateral_init=lat,
                                     min_duration=0.6)
            ctrl.new_reference()
        u, diag = ctrl.step(ego, ref, [])
        max_cte = max(max_cte, diag["tracking_error"])
        p.step(u)
    ego = p.ego(road)
    assert abs(ego.d - 0.0) < 0.2, ego.d
    assert abs(ego.v - kmh(60)) < 0.5, ego.v
    assert max_cte < 0.5


# ----------------------------------------------------------------------------- spline
def test_m3_spline_passes_waypoints_without_overshoot(road, plant_cls, actuation):
    ego = plant_cls(actuation, x=50, y=-3.5, v=18).ego(road)
    wps = np.array([[80.0, 0.0], [120.0, 0.0], [160.0, -3.5]])
    ref = build_m3_reference(ego, road, wps, kmh(70), ds=1.0, extend=50.0)
    for s, d in wps:
        assert ref.at_s(s)["d"] == pytest.approx(d, abs=1e-6)
    assert ref.d.max() <= 0.0 + 1e-9 and ref.d.min() >= -3.5 - 1e-9
    assert ref.s[-1] == pytest.approx(210.0)


# ----------------------------------------------------------------------------- safe set (spec §17.5)
def _safe_cfg(cfg_m3):
    c = dict(cfg_m3["method"]["safe_set"])
    c["v_cap"] = kmh(80)
    return c


def test_safe_set_inside_unchanged(road, plant_cls, actuation, cfg_m3):
    ego = plant_cls(actuation, x=50, y=-3.5, v=18).ego(road)
    wps = np.array([[70.0, -3.5], [100.0, -3.5]])
    proj, dist = project_waypoints(wps, ego, [], road, _safe_cfg(cfg_m3))
    np.testing.assert_allclose(proj, wps)
    np.testing.assert_allclose(dist, 0.0)


def test_safe_set_off_road_projected_inside(road, plant_cls, actuation, cfg_m3):
    ego = plant_cls(actuation, x=50, y=-7.0, v=18).ego(road)
    c = _safe_cfg(cfg_m3)
    wps = np.array([[100.0, -10.0], [150.0, 3.0]])
    res = project_waypoints_detailed(wps, ego, [], road, c, c["v_cap"])
    lo = road.d_min + ego.width / 2 + c["edge_margin"]
    hi = road.d_max - ego.width / 2 - c["edge_margin"]
    assert res.valid.all()
    assert lo - 1e-6 <= res.projected[0, 1] <= hi + 1e-6 and res.projected[0, 1] == pytest.approx(lo)
    assert res.projected[1, 1] <= hi + 1e-6
    assert res.distances[0] == pytest.approx(lo - (-10.0))


def test_safe_set_object_footprint(road, plant_cls, actuation, cfg_m3):
    ego = plant_cls(actuation, x=50, y=-3.5, v=18).ego(road)
    c = _safe_cfg(cfg_m3)
    stopped = make_tracked_object("x", 80.0, -3.5, 0.0, 0.0, 0.0, 4.5, 1.85, "bg", road)
    wps = np.array([[80.0, -3.5]])  # right on top of a stopped car
    res = project_waypoints_detailed(wps, ego, [stopped], road, c, c["v_cap"])
    assert res.valid.all()
    p = res.projected[0]
    box = res.boxes[0][0]
    assert not (box[0] < p[0] < box[1] and box[2] < p[1] < box[3])
    assert res.distances[0] > 0
    # a lateral exit (2.35 m) would be nearer, but it is not reachable within 30 m at 18 m/s under the
    # lateral-acceleration limit, so the projection exits longitudinally (before the box)
    half_l = 0.5 * (4.5 + ego.length) + c["obj_margin_s"]
    assert p[0] == pytest.approx(80.0 - half_l, abs=0.01)
    assert p[1] == pytest.approx(-3.5)
    # with lateral reach available (slow ego), the nearer lateral exit is chosen
    slow = plant_cls(actuation, x=50, y=-3.5, v=5).ego(road)
    res2 = project_waypoints_detailed(wps, slow, [stopped], road, c, c["v_cap"])
    half_w = 0.5 * (1.85 + ego.width) + c["obj_margin_d"]
    assert res2.projected[0, 0] == pytest.approx(80.0)
    assert abs(res2.projected[0, 1] - (-3.5)) == pytest.approx(half_w, abs=0.01)


def test_safe_set_infeasible_drops_rest(road, plant_cls, actuation, cfg_m3):
    ego = plant_cls(actuation, x=50, y=-3.5, v=18).ego(road)
    c = _safe_cfg(cfg_m3)
    c["max_slope"] = 0.0       # no lateral motion allowed
    c["min_ds"] = 40.0         # first waypoint must be >= 90 m: stopped car blocks the whole reachable lane
    wall = make_tracked_object("x", 100.0, -3.5, 0.0, 0.0, 0.0, 400.0, 1.85, "bg", road)
    res = project_waypoints_detailed(np.array([[95.0, -3.5], [130.0, -3.5]]), ego, [wall], road, c, c["v_cap"])
    assert res.all_infeasible
    assert np.all(np.isnan(res.projected))
    np.testing.assert_allclose(res.distances, c["infeasible_penalty_distance"])


# ----------------------------------------------------------------------------- MPC (spec §17.4 in miniature)
def _mpc(cfg_m3, road, actuation):
    m = cfg_m3["method"]["mpc"]
    return MPCController(m, road, actuation, lf=1.05234, lr=1.4166, ego_length=4.515, ego_width=1.852,
                         max_steering_rad=np.deg2rad(40), v_cap=kmh(80), v_target=kmh(70))


def test_mpc_passes_slow_lead(road, plant_cls, actuation, cfg_m3):
    mpc = _mpc(cfg_m3, road, actuation)
    p = plant_cls(actuation, x=50, y=-3.5, v=18)
    lead = [90.0, -3.5, 10.0]
    min_gap, fallbacks = np.inf, 0
    from handoff.sim.state import frenet_box_gap

    for k in range(120):
        ego = p.ego(road, t=k * 0.1)
        obj = make_tracked_object("lead", lead[0], lead[1], 0.0, lead[2], 0.0, 5.74, 2.3, "lead", road)
        if k % 10 == 0:
            ref = build_m3_reference(ego, road, np.array([[ego.s + 25, 0.0], [ego.s + 80, 0.0]]), kmh(70))
        u, diag = mpc.step(ego, ref, [obj])
        fallbacks += diag["fallback"]
        min_gap = min(min_gap, frenet_box_gap(ego, obj))
        p.step(u)
        lead[0] += lead[2] * 0.1
    assert p.x > lead[0] + 10, "ego should have passed the lead"
    assert min_gap > 0.3
    assert abs(p.y - 0.0) < 0.3
    assert fallbacks == 0


def test_mpc_avoids_stopped_vehicle_in_path(road, plant_cls, actuation, cfg_m3):
    """Waypoints drive straight through a stopped car; the MPC's collision constraint must prevent contact."""
    mpc = _mpc(cfg_m3, road, actuation)
    p = plant_cls(actuation, x=50, y=-3.5, v=15)
    from handoff.sim.state import frenet_box_gap

    stopped = make_tracked_object("s", 110.0, -3.5, 0.0, 0.0, 0.0, 4.5, 1.85, "bg", road)
    min_gap = np.inf
    for k in range(150):
        ego = p.ego(road, t=k * 0.1)
        if k % 10 == 0:
            ref = build_m3_reference(ego, road, np.array([[ego.s + 30, -3.5]]), kmh(70))
        u, _ = mpc.step(ego, ref, [stopped])
        min_gap = min(min_gap, frenet_box_gap(ego, stopped))
        p.step(u)
    assert min_gap > 0.2, min_gap


def test_mpc_fallback_chain(road, plant_cls, actuation, cfg_m3):
    mpc = _mpc(cfg_m3, road, actuation)
    p = plant_cls(actuation, x=50, y=-3.5, v=18)
    ego = p.ego(road)
    ref = build_m3_reference(ego, road, np.array([[ego.s + 30, -3.5]]), kmh(70))
    u, diag = mpc.step(ego, ref, [])
    assert not diag["fallback"]
    # force failures: zero iterations allowed
    real_solver = mpc.solver

    def broken(**kw):
        raise RuntimeError("forced failure")

    mpc.solver = broken
    u1, d1 = mpc.step(ego, ref, [])
    assert d1["fallback"] and d1["fallback_mode"] == "shifted_previous"
    for _ in range(mpc.N):
        u2, d2 = mpc.step(ego, ref, [])
    assert d2["fallback_mode"] == "emergency_brake" and u2[1] == -1.0
    mpc.solver = real_solver
    u3, d3 = mpc.step(ego, ref, [])
    assert not d3["fallback"]
