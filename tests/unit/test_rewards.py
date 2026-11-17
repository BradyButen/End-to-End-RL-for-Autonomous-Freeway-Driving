import numpy as np
import pytest

from handoff.config import load_config
from handoff.rl.env.rewards import TERM_NAMES, RewardComposer
from handoff.simulator.state import make_ego_state


def _ego(road, s=100.0, d=0.0, v=20.0, steering=0.0, t=0.0):
    x, y = road.to_world(s, d)
    return make_ego_state(t, x, y, 0.0, v, steering, 0.0, 4.5, 1.85, road)


def _composer(variant, road):
    cfg = load_config("m1", variant)
    ctx = {"dt": 0.1, "v_max": 80 / 3.6, "v_target": 70 / 3.6, "road": road}
    return RewardComposer.from_config(cfg["reward"], ctx)


def test_all_terms_always_computed(road):
    comp = _composer("r0", road)
    _, raw = comp.compute(_ego(road), _ego(road, s=102.0), [], {}, None)
    assert set(raw) == set(TERM_NAMES)


def test_r0_matches_metadrive_semantics(road):
    comp = _composer("r0", road)
    prev, curr = _ego(road, s=100.0, v=20.0), _ego(road, s=102.0, v=20.0)
    total, raw = comp.compute(prev, curr, [], {}, None)
    assert raw["progress"] == pytest.approx(2.0)
    assert total == pytest.approx(1.0 * 2.0 + 0.1 * 20.0 / (80 / 3.6))
    # terminal replaces the step reward; priority success > off_road > collision
    assert comp.compute(prev, curr, [], {"crash_vehicle": True}, None)[0] == pytest.approx(-5.0)
    assert comp.compute(prev, curr, [], {"out_of_road": True}, None)[0] == pytest.approx(-5.0)
    assert comp.compute(prev, curr, [], {"arrive_dest": True, "crash_vehicle": True}, None)[0] == pytest.approx(10.0)
    w = comp.weighted(comp.compute(prev, curr, [], {"crash_vehicle": True}, None)[1])
    assert w["collision"] == pytest.approx(-5.0) and w["progress"] == 0.0


def test_add_mode_and_comfort_terms(road):
    comp = _composer("r2_dense_comfort", road)
    prev = _ego(road, s=100.0, d=-0.5, v=20.0, steering=0.0)
    curr = _ego(road, s=102.0, d=-0.875, v=21.0, steering=0.1)
    _, raw = comp.compute(prev, curr, [], {}, None)
    assert raw["lane_center"] == pytest.approx(-0.875 / 1.75)
    assert raw["steer_rate"] == pytest.approx(-1.0)
    assert raw["speed_tracking"] == pytest.approx(-abs(21.0 - 70 / 3.6) / (70 / 3.6))
    assert raw["jerk"] == 0.0  # first step
    _, raw2 = comp.compute(curr, _ego(road, s=104.0, v=21.0), [], {}, None)
    assert raw2["jerk"] == pytest.approx(-100.0)  # accel 10 -> 0 m/s^2 in 0.1 s


def test_projection_penalty_once_per_decision(road):
    comp = _composer("r2_dense_comfort", road)
    prev, curr = _ego(road), _ego(road, s=102.0)
    dec = {"first_step": True, "projection_distances": np.array([1.0, 0.5])}
    assert comp.compute(prev, curr, [], {}, dec)[1]["projection_penalty"] == pytest.approx(-1.5)
    dec2 = {**dec, "first_step": False}
    assert comp.compute(prev, curr, [], {}, dec2)[1]["projection_penalty"] == 0.0


def test_unknown_weight_rejected(road):
    cfg = load_config("m1", "r0")
    cfg["reward"]["weights"]["nonexistent"] = 1.0
    with pytest.raises(ValueError):
        RewardComposer.from_config(cfg["reward"], {"dt": 0.1, "v_max": 1, "v_target": 1, "road": road})
