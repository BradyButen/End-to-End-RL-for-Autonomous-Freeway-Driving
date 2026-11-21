# Design

How the study is built. Training settings and their rationale are in [TRAINING.md](TRAINING.md); the original
course specification is [SPEC_v2.md](SPEC_v2.md) (historical, not updated since 2026-11-16).

## 1. The question
Where should learning hand off to control? Four drivers on the same freeway, same observation, same reward:

| method | RL decides (every) | what turns it into steering/throttle |
|---|---|---|
| **M1** end-to-end | `[steering, target accel]` (0.1 s; accel 0 = hold speed, `method.accel_range`) | actuation table → throttle_brake |
| **M2** RL + PID | target lateral offset d\*, target speed v\* (0.5 s) | quintic lateral profile + accel-limited speed profile (2 s), cascaded PID |
| **M3** RL + MPC | up to 5 Frenet waypoints (Δs, d), target/max speed v\*, waypoint count (1 s) | spline → road corridor → hard-constrained MPC (+ emergency-brake gate) |
| **B0** rule-based | — | MetaDrive's IDM at the 70 km/h target speed (evaluation only) |

## 2. Simulator and scenario
* **MetaDrive** `TopDownMetaDrive`, stock — configured, not patched (`sim.py`).
* **Road**: 17 straight PG blocks (lengths drawn from the seed) → 852–1144 m route, 3 lanes × 3.5 m, stock
  two-way map; crossing the solid centre line or the road edge ends the episode (off-road).
* **Traffic**: MetaDrive's `PGTrafficManager`, density 0.15 (~36 cars per episode), placed randomly over every block
  but the first (from the seed). All cars drive from t = 0 (`release_all_at_reset`; MetaDrive's Trigger mode would
  hold each block's cars until the ego reaches it), starting at and cruising at their lane's speed under IDM: **left 45, middle 40,
  right 35 km/h** (re-evaluated every step from the car's current lane; set per car's policy in
  `sim.TrafficSpeeds`; IDM's built-in is 30).
* **Ego**: `static_default` vehicle — fixed dynamics (engine 800, brake 150, friction 0.9, 40° steering). The stock
  `default` model resamples engine/brake every episode. Spawns in the first block, random lane, 40 km/h.
* **Seeds**: training draws map+traffic seeds from [0, 1000); evaluation uses 40 fixed seeds 1000–1039.
* **Step**: physics 0.02 s, control step 0.1 s; episode cap 1500 control steps (150 s).
* `force_destroy: true` (D-11): objects are destroyed on reset, so re-simulation of an episode is bit-exact.

## 3. Shared Gym env (`env.py`)
* `DrivingEnv` — one MetaDrive instance and the loop all methods share: one RL action → `method.decide()` → up to
  N control steps of `method.control()`; each control step applies `[steering, throttle_brake]`, extracts the ego
  and the cars within R = 50 m in the road (Frenet) frame, computes and logs every reward term. The decision's
  reward is the undiscounted sum over its control steps.
* A method (`m1.py`, `m2.py`, `m3.py`) is a small class with `period`, `action_space`, `decide(action)` and
  `control()`; it reads the world from the env (`env.ego`, `env.objects`, `env.road`).
* **Observation** (identical for all methods): `bev` 5×84×84 top-down (road, past positions, 3 traffic frames;
  ±50 m) + `ego` [v/v_max, d (road-centred, ÷ half-width), heading error, steering, throttle_brake].
* **Frenet frame** (`sim.RoadFrame`): s along the route, d positive to the left; lane centres d = 0, −3.5, −7.
* **Termination reasons**: success > collision > emergency_brake (M3) > off_road > timeout.

## 4. Methods
### M2 (`m2.py`)
Action → d\* ∈ [right edge − 1 m, left edge + 1 m] (not clipped to the road), v\* ∈ [0, 80] km/h. Each decision
builds a quintic from the current lateral state (continuity with the previous reference) to d\* over 2 s and a
speed ramp (+2 / −4 m/s²). PID: heading [1.0, 0.005, 0], lateral [0.06, 0.0005, 0], speed [0.25, 0.0008, 0] with
acceleration feed-forward through the actuation table (gains tuned 2026-11-19, `scripts/matlab/m2_pid_tuning.m`).
No safety layer.

### M3 (`m3.py`, `mpc.py`)
Action (12-D) = `[Δs₁, d₁, …, Δs₅, d₅, v*, count]`: Δs ∈ [5, 40] m per slot, d as in M2, v\* ∈ [20, 80] km/h,
k = 1–5 waypoints. Pipeline (also driven by `scripts/validate.py` with hand-placed waypoints):
1. **Spline** — piecewise quintic in s through the waypoints (flat 80 m beyond the last), speed v\*.
2. **Road corridor** (`m3.Corridor`, D-13) — one road-aligned box per MPC stage on the ego centre, rebuilt every
   control step: lateral faces fixed at the exterior lane centres widened by 0.3 m; cars' keep-out zones (car +
   1 m fore/aft + 0.3 m sideways, grown by the ego half-size) add a front face (car ahead in the path), back face
   (behind) or lateral face (alongside / level). Initial trajectory = reference at a ramp to v\* (cars ignored:
   stopping is the MPC's job).
3. **MPC** — kinematic bicycle + 0.1 s steering lag, horizon 50 × 0.1 s, CasADi/IPOPT, hard corridor constraints,
   costs on tracking, effort, rates, yaw rate and a barrier near the lateral faces. **v\* is also a hard speed
   ceiling** (ramped down at 4 m/s² from the current speed so it is always feasible).
4. **Emergency-brake gate** — a car in the planned path (the line ego → next waypoint crosses its keep-out zone)
   that the ego closes on faster than it can stop for (Δv² / 2·gap > 6 m/s²) → full brake with a heading-first
   steering law instead of the MPC. Following a car at its speed never fires it.
5. **Failures** — solve time counted in process CPU time; successful solves always used. Timeout / numerical
   failure → replay the previous plan up to 5 steps (`mpc.replay_steps.timeout`); infeasible → no replay
   (`mpc.replay_steps.infeasible: 0`); then emergency brake.

**Failed runs** (`method.ebrake.ends_episode`, 2026-11-21): the RL is charged for plans the controller cannot carry
out, instead of the car stopping dead on the freeway. All end the episode with the collision penalty:
* `infeasible_plan` — checked once per decision: a waypoint segment of the spline needs more peak lateral
  acceleration (5.77·|Δd|·v²/Δs², at the speed reachable by that segment braking at 4 m/s², floored at
  min(v, v\*)) than the MPC's `max_lat_accel` (4 m/s²). Ends on the decision's first control step.
* `emergency_brake` — the gate fires, or the MPC still fails after the replay.

Only M3 has these; report them next to collisions (predicted crashes / undrivable plans). In
`scripts/validate.py run` (ends_episode false) none of them end the run: the car brakes and you see it.

## 5. Logging (`logs.py`, `evaluate.py`)
Every episode → one row in `episodes.csv` (termination, return and per-term returns, speed, jerk, gaps, solver
stats, `gated_brake_count`, lane changes, …). Every eval episode and every 50th training episode per env → full
per-step `.npz` (ego, objects, reward terms, controller diagnostics; M3 corridor every 10th step), replayable with
`scripts/view.py replay`; `scripts/plot.py episode <npz>` draws its overview plot and writes a per-step CSV.

## 6. Decision IDs referenced in code
| ID | decision |
|---|---|
| D-04 | Decision periods: M2 0.5 s, M3 1 s (fixed); M3 Δs ∈ [5, 40] m, lateral margin 1 m. |
| D-06 | Safe-set projection of M3 waypoints: disabled 2026-11-19, removed 2026-11-21 (in git history before then). |
| D-07 | Budgets, eval and checkpoint cadences counted in control steps (= simulated time), so methods with longer decision periods are compared at equal experience. Best checkpoint = highest eval success, ties by return. |
| D-11 | `force_destroy` for bit-exact re-simulation. |
| D-12 | Actuation map a ↔ throttle_brake: measured lookup table (brake is strongly nonlinear); MPC steering lag 0.1 s and lateral-accel bound 4 m/s². |
| D-13 | M3 collision avoidance = hard convex corridor (no slack); now the road-aligned `RoadCorridor`. |
