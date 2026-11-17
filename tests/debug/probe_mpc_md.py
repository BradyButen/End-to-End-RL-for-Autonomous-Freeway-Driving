import sys
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validate import _all_traffic, _env_with_spec, _spec

from handoff.config import kmh

which = sys.argv[1]
if which == "lead":
    spec = _spec(ego_lane=1, ego_v=kmh(60), lead_lane=1, lead_s=130.0, lead_v=kmh(35)); ov=sys.argv[2:]
else:
    spec = _spec(ego_lane=1, ego_v=kmh(50), lead_lane=1, lead_s=170.0, lead_v=0.0, lead_model="default"); ov=["method.safe_set.enabled=false"]+sys.argv[2:]
env, cfg = _env_with_spec("m3", spec, overrides=ov)
env.reset(options={"episode_seed": 0})
core = env.core; L = core.road.lane_centers
for k in range(40):
    lead = _all_traffic(core, "lead")
    if which == "lead":
        ahead = (lead[0].s - core.ego.s) if lead else -999
        d_tgt = L[0] if ahead > -15 else L[1]
        a = env.encode([40.0, 40.0], [d_tgt, d_tgt])
    else:
        a = env.encode([25.0, 25.0], [L[1], L[1]])
    obs, r, te, tr, info = env.step(a)
    st = core.recorder.steps
    for j in range(-env.period, 0, 2):
        print(f"dec {k} s={st['s'][j]:.1f} d={st['d'][j]:.2f} psi={st['psi'][j]:.3f} v={st['v'][j]:.1f} cmd=({st['cmd_steering'][j]:.3f},{st['cmd_throttle'][j]:.2f}) refd={st['ref_d'][j]:.2f} blocked={st['tube_blocked'][j]:.0f} gap={st['min_gap'][j]:.1f} lead_s={lead[0].s if lead else 0:.1f} nobj={len(core.objects)} st={core.recorder.status[j]}")
    if te or tr or k == 39:
        print('END', k, 'min_gap', min(core.recorder.steps['min_gap']), 'fallbacks', int(sum(core.recorder.steps['fallback'])), 'solve p95', np.percentile(core.recorder.steps['solve_time'],95)*1e3)
    if te or tr:
        print(info['termination'], {k: info[k] for k in ['crash_vehicle','out_of_road']}, core.md.agent.on_white_continuous_line, core.md.agent.on_yellow_continuous_line, core.md.agent.crash_sidewalk); break
env.close()
