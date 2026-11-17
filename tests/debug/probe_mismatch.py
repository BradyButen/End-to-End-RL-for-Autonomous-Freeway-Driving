import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validate import _all_traffic, _env_with_spec, _spec

from handoff.config import kmh

spec = _spec(ego_lane=1, ego_v=kmh(50), lead_lane=1, lead_s=170.0, lead_v=0.0, lead_model="default")
env, cfg = _env_with_spec("m3", spec, overrides=["method.safe_set.enabled=false"]+sys.argv[1:])
env.reset(options={"episode_seed": 0})
core = env.core; L = core.road.lane_centers
mpc = env.mpc
orig = mpc.step
log = []
def step(ego, ref, objs):
    u, d = orig(ego, ref, objs)
    X = mpc._prev_X
    log.append((ego.x, ego.y, ego.v, ego.psi, u.copy(), None if X is None else X[:, 1].copy(), d["status"] + f" nobj={len(objs)} objs={[(round(o.x,1), round(o.y,1)) for o in objs]} traffic={[(round(t.x,1),round(t.y,1)) for t in _all_traffic(core)]}", mpc.u_prev.copy()))
    return u, d
env.mpc.step = step
for k in range(40):
    obs, r, te, tr, info = env.step(env.encode([25.0, 25.0], [L[1], L[1]]))
    if te or tr: break
for i in range(len(log)-1):
    x,y,v,psi,u,pred,st,uprev = log[i]
    nx,ny,nv,npsi = log[i+1][:4]
    if pred is not None and i > len(log)-25:
        print(i, f"v={v:.2f} a_cmd={uprev[1]:+.2f} u={u.round(3)} pred_v={pred[3]:.2f} act_v={nv:.2f} | pred_y={pred[1]:.3f} act_y={ny:.3f} pred_psi={pred[2]:.3f} act_psi={npsi:.3f} {st}")
print(info.get('termination'))
env.close()
