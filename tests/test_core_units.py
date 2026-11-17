"""Unit tests without the simulator: config, RoadFrame, state helpers, scenario sampling, metrics, actuation."""

import numpy as np
import pytest

from handoff.config import deep_merge, load_config, parse_override
from handoff.env.decision import affine
from handoff.env.factory import method_gamma
from handoff.eval.metrics import count_lane_changes, episode_metrics, termination_reason
from handoff.methods.common.actuation import ActuationMap
from handoff.methods.m3.wrapper import waypoint_count
from handoff.sim.road import RoadFrame
from handoff.sim.scenario import VEHICLE_LENGTHS, FreewayPassScenario
from handoff.sim.state import frenet_box_gap, make_ego_state, make_tracked_object, time_to_collision


# ----------------------------------------------------------------------------- config
def test_deep_merge_and_overrides():
    a = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    b = {"x": {"y": 2}, "n": 3}
    m = deep_merge(a, b)
    assert m == {"x": {"y": 2, "z": [1, 2]}, "k": 1, "n": 3}
    assert a["x"]["y"] == 1  # not mutated
    assert parse_override("a.b.c=[1, 2]") == {"a": {"b": {"c": [1, 2]}}}
    assert parse_override("ppo.learning_rate=1e-4") == {"ppo": {"learning_rate": 1e-4}}


def test_load_config_merge_order():
    cfg = load_config("m3", "r0", overrides=["method.decision_period=7"])
    assert cfg["experiment"]["method"] == "m3"
    assert cfg["experiment"]["reward_variant"] == "r0"
    assert cfg["method"]["decision_period"] == 7
    assert cfg["sim"]["ego"]["max_engine_force"] == 800
    assert set(cfg["scenario"]["seeds"]["eval"]).isdisjoint(range(cfg["scenario"]["seeds"]["train_low"],
                                                                  cfg["scenario"]["seeds"]["train_high"]))


def test_discount_scaling():
    for m, period in (("m1", 1), ("m2", 5), ("m3", 10)):
        cfg = load_config(m)
        assert cfg["method"]["decision_period"] == period
        assert method_gamma(cfg) == pytest.approx(0.99 ** period)


# ----------------------------------------------------------------------------- road frame
def test_roadframe_roundtrip(road):
    rng = np.random.default_rng(0)
    for _ in range(200):
        s, d = rng.uniform(0, 990), rng.uniform(-9, 2)
        x, y = road.to_world(s, d)
        s2, d2 = road.to_frenet(x, y)
        assert s2 == pytest.approx(s, abs=1e-6)
        assert d2 == pytest.approx(d, abs=1e-6)


def test_roadframe_curved_roundtrip():
    th = np.linspace(0, np.pi / 2, 400)
    pts = np.stack([100 * np.sin(th), 100 * (1 - np.cos(th))], axis=1)  # left-turning arc
    rf = RoadFrame(pts, [0.0, -3.5], d_min=-5.25, d_max=1.75, lane_width=3.5)
    for s in (5.0, 50.0, 120.0):
        for d in (-4.0, 0.0, 1.5):
            x, y = rf.to_world(s, d)
            s2, d2 = rf.to_frenet(x, y)
            assert abs(s2 - s) < 0.05 and abs(d2 - d) < 0.02


def test_roadframe_sign_convention(road):
    # heading +x, left = +y => positive d
    assert road.to_frenet(10.0, 1.0)[1] == pytest.approx(1.0)
    assert road.lane_centers == [0.0, -3.5, -7.0]
    assert road.d_max == pytest.approx(1.75) and road.d_min == pytest.approx(-8.75)
    assert road.nearest_lane(-3.0) == 1


def test_ego_state_and_gap(road):
    ego = make_ego_state(0, 100, -3.5, 0.05, 20, 0, 0, 4.5, 1.85, road)
    assert ego.lane_index == 1 and ego.psi_err == pytest.approx(0.05)
    lead = make_tracked_object("a", 120, -3.5, 0.0, 10, 0, 5.74, 2.3, "lead", road)
    gap = frenet_box_gap(ego, lead)
    assert gap == pytest.approx(20 - (4.5 + 5.74) / 2)
    ttc = time_to_collision(ego, lead, road)
    closing = 20 * np.cos(0.05) - 10
    assert ttc == pytest.approx(gap / closing, rel=1e-3)
    side = make_tracked_object("b", 100, 0.0, 0.0, 25, 0, 4.5, 1.85, "bg", road)
    assert np.isinf(time_to_collision(ego, side, road))


# ----------------------------------------------------------------------------- scenario
def test_scenario_sampling(cfg_m3):
    sc = FreewayPassScenario(cfg_m3["scenario"], lane_num=3)
    for seed in range(50):
        spec = sc.sample(np.random.default_rng(seed), seed)
        c = cfg_m3["scenario"]
        assert spec.lead.lane == spec.ego.lane
        gap = spec.lead.s - spec.ego.s - 0.5 * (VEHICLE_LENGTHS["default"] + VEHICLE_LENGTHS["xl"])
        assert c["lead"]["gap"][0] - 1e-9 <= gap <= c["lead"]["gap"][1] + 1e-9
        assert c["lead"]["speed_km_h"][0] / 3.6 - 1e-9 <= spec.lead.speed <= c["lead"]["speed_km_h"][1] / 3.6 + 1e-9
        assert len(spec.background) <= c["background"]["count"][1]
        if c["background"].get("lanes", "other") == "other":
            for b in spec.background:
                assert b.lane != spec.ego.lane
        # deterministic per seed
        assert sc.sample(np.random.default_rng(seed), seed).as_dict() == spec.as_dict()


# ----------------------------------------------------------------------------- decoding
def test_waypoint_count_mapping():
    assert waypoint_count(-1.0) == 1
    assert waypoint_count(1.0) == 5
    assert waypoint_count(0.0) == 3
    assert [waypoint_count(c) for c in np.linspace(-1, 1, 11)] == sorted(waypoint_count(c)
                                                                         for c in np.linspace(-1, 1, 11))
    assert affine(-1, 2, 4) == 2 and affine(1, 2, 4) == 4 and affine(0, 2, 4) == 3


# ----------------------------------------------------------------------------- actuation
def test_actuation_map_monotone_inverse(actuation):
    for a in np.linspace(actuation.a_min, actuation.a_max, 25):
        assert actuation.command_to_accel(actuation.accel_to_command(a)) == pytest.approx(a, abs=1e-6)
    assert actuation.accel_to_command(100) == pytest.approx(1.0)
    assert actuation.accel_to_command(-100) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        ActuationMap([[0, 1], [1, 0]])


# ----------------------------------------------------------------------------- metrics
def test_termination_reason_priority():
    assert termination_reason({"arrive_dest": True, "crash_vehicle": True}, False) == "success"
    assert termination_reason({"crash_vehicle": True, "out_of_road": True}, False) == "collision"
    assert termination_reason({"out_of_road": True}, False) == "off_road"
    assert termination_reason({}, True) == "timeout"


def test_lane_change_counting():
    d = np.concatenate([np.zeros(10), np.linspace(0, -3.5, 20), np.full(10, -3.5), np.linspace(-3.5, 0, 20)])
    assert count_lane_changes(d, [0.0, -3.5, -7.0], 3.5) == 2
    jitter = np.array([0.0, -1.7, -1.8, -1.7, -1.8, 0.0])  # hovering on the line, never settles
    assert count_lane_changes(jitter, [0.0, -3.5, -7.0], 3.5) == 0


def test_episode_metrics_basic():
    n = 50
    t = np.arange(n) * 0.1
    steps = {"t": t, "v": np.full(n, 10.0), "s": 10.0 * t, "d": np.zeros(n), "psi": np.zeros(n),
             "steering": np.zeros(n), "min_gap": np.linspace(30, 5, n), "min_ttc": np.full(n, np.inf),
             "tracking_error": np.full(n, 0.1), "solve_time": np.full(n, 0.01), "fallback": np.zeros(n)}
    decs = {"k": np.array([1, 2, 2, 5]), "projection_distance_sum": np.array([0.0, 1.0, 0.0, 2.0])}
    m = episode_metrics(steps, decs, "success", 0.1, [0.0, -3.5, -7.0], 3.5, np.deg2rad(40))
    assert m["success"] == 1.0 and m["time_to_complete"] == pytest.approx(4.9)
    assert m["mean_speed"] == pytest.approx(10.0) and m["rms_long_jerk"] == pytest.approx(0.0)
    assert m["min_gap"] == pytest.approx(5.0) and m["k2_count"] == 2 and m["k5_count"] == 1
    assert m["projection_distance_mean"] == pytest.approx(0.75)
    assert m["solve_time_p95"] == pytest.approx(0.01)
