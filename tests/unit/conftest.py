import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from handoff.config import kmh, load_config  # noqa: E402
from handoff.rl.methods.common import ActuationMap  # noqa: E402
from handoff.simulator.road import RoadFrame  # noqa: E402


@pytest.fixture
def road():
    return RoadFrame.straight(1000.0, lane_num=3, lane_width=3.5)


@pytest.fixture
def cfg_m2():
    return load_config("m2")


@pytest.fixture
def cfg_m3():
    return load_config("m3")


@pytest.fixture
def actuation(cfg_m3):
    return ActuationMap(cfg_m3["actuation"]["table"])


class KinematicPlant:
    """Kinematic bicycle with a first-order steering lag and the actuation table — a cheap stand-in for
    MetaDrive in controller unit tests."""

    def __init__(self, actuation, x=0.0, y=0.0, psi=0.0, v=15.0, lf=1.05234, lr=1.4166, tau=0.1,
                 max_steer=np.deg2rad(40), dt=0.1, v_cap=kmh(80)):  # noqa: B008
        self.x, self.y, self.psi, self.v = x, y, psi, v
        self.lf, self.lr, self.tau, self.max_steer, self.dt, self.v_cap = lf, lr, tau, max_steer, dt, v_cap
        self.delta = 0.0
        self.act = actuation
        self.steering = 0.0
        self.throttle = 0.0

    def step(self, u, substeps=5):
        self.steering, self.throttle = float(u[0]), float(u[1])
        cmd = u[0] * self.max_steer
        a = self.act.command_to_accel(u[1])
        h = self.dt / substeps
        for _ in range(substeps):
            self.delta += (cmd - self.delta) * h / self.tau
            beta = np.arctan(self.lr / (self.lf + self.lr) * np.tan(self.delta))
            self.x += self.v * np.cos(self.psi + beta) * h
            self.y += self.v * np.sin(self.psi + beta) * h
            self.psi += self.v / self.lr * np.sin(beta) * h
            self.v = float(np.clip(self.v + a * h, 0.0, self.v_cap))

    def ego(self, road, t=0.0):
        from handoff.simulator.state import make_ego_state

        return make_ego_state(t, self.x, self.y, self.psi, self.v, self.steering, self.throttle, 4.515, 1.852, road)


@pytest.fixture
def plant_cls():
    return KinematicPlant
