import sys
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
from conftest import KinematicPlant

from handoff.config import kmh, load_config
from handoff.methods.common.actuation import ActuationMap
from handoff.methods.m3.mpc import MPCController
from handoff.methods.m3.spline import build_m3_reference
from handoff.sim.road import RoadFrame
from handoff.sim.state import make_tracked_object

cfg = load_config("m3", overrides=sys.argv[1:])
road = RoadFrame.straight(1000); act = ActuationMap(cfg["actuation"]["table"])
mpc = MPCController(cfg["method"]["mpc"], road, act, lf=1.05234, lr=1.4166, ego_length=4.515, ego_width=1.852, max_steering_rad=np.deg2rad(40), v_cap=kmh(80), v_target=kmh(70))
p = KinematicPlant(act, x=50, y=-3.5, v=18); lead=[90.0,-3.5,10.0]
for k in range(120):
    ego = p.ego(road, t=k*0.1)
    obj = make_tracked_object("lead", lead[0], lead[1], 0.0, lead[2], 0.0, 5.74, 2.3, "lead", road)
    if k % 10 == 0:
        ref = build_m3_reference(ego, road, np.array([[ego.s + 25, 0.0], [ego.s + 80, 0.0]]), kmh(70))
    u, d = mpc.step(ego, ref, [obj])
    if k % 5 == 0:
        ext = mpc.tube.extents[min(mpc._tube_age, len(mpc.tube) - 1)]
        print(k, f"x={p.x:.1f} y={p.y:.2f} v={p.v:.1f} u={u.round(2)} lead={lead[0]:.1f} "
              f"refd@ego={ref.at_s(ego.s)['d']:.2f} box[u-,u+,w-,w+]={np.round(ext, 1)} "
              f"blocked={d['tube_blocked']} {d['status']}")
    p.step(u); lead[0] += lead[2]*0.1
