import numpy as np

from handoff.config import load_config
from handoff.simulator.metadrive_env import FreewayMetaDriveEnv, make_metadrive_config
from handoff.simulator.scenario import ScenarioSpec, VehicleSpec

cfg = load_config("m1")
env = FreewayMetaDriveEnv(make_metadrive_config(cfg))
L = 1.05234 + 1.4166
for v0 in [10.0, 15.0, 20.0]:
  for st in [0.05, 0.1]:
    spec = ScenarioSpec(seed=0, ego=VehicleSpec(lane=1, s=60, speed=v0, role='ego'), lead=VehicleSpec(lane=0, s=900, speed=5, model='xl', role='lead'), background=[])
    env.reset_with_spec(spec)
    a = env.agent
    # hold speed for a few steps
    for i in range(5): env.step([0, 0.12])
    psis=[a.heading_theta]; vs=[a.speed]
    for i in range(8):
        env.step([st, 0.12]); psis.append(a.heading_theta); vs.append(a.speed)
    r = np.diff(psis)/0.1
    delta = st*np.deg2rad(40)
    model = np.mean(vs)*np.tan(delta)/L
    print(f"v={np.mean(vs):.1f} st={st}: yaw rates", np.round(r,3), " kinematic ss", round(model,3), "ratio_ss", round(r[-1]/model,3))
env.close()
