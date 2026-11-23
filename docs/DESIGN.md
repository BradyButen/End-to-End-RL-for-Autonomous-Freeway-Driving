# Design

How the study is built. Training settings and their rationale are in [TRAINING.md](TRAINING.md); the original
course specification is [SPEC_v2.md](SPEC_v2.md) (historical, not updated since 2026-11-16).

## 1. The question
Where should learning hand off to control? Four drivers on the same freeway, same observation, same reward (`configs/r5.yaml`):

| method | RL decides (every) | what the controller owns | what turns it into steering/throttle |
|---|---|---|---|
| **M1** end-to-end | `[lateral accel, target accel]` (0.1 s; 0 = straight / hold speed) | nothing | kinematic bicycle → wheel angle; actuation table → throttle_brake |
| **M2** RL + PID | lane command keep / left / right + speed level 80…0 km/h (0.5 s) | tracking | quintic to the target lane + accel-limited speed profile (2 s), cascaded PID |
| **M3** RL + MPC | lane command keep / left / right (1 s) | tracking, speed (toward 80 km/h), braking for cars | lane plan → road corridor → MPC with a soft (exact-penalty) corridor |
| **B0** rule-based | — | everything | MetaDrive's IDM at the 70 km/h target speed (evaluation only) |

Batches 1–5 (until 2026-11-23) used continuous targets for M2 (d\*, v\*) and 12 waypoint numbers for M3, and a
top-down image observation; both converged on following traffic at ~45 km/h (`TRAINING.md` §5).

## 2. Simulator and scenario
* **MetaDrive** `MetaDriveEnv`, stock — configured, not patched (`sim.py`); its own observation is unused.
* **Road**: 17 straight PG blocks (lengths drawn from the seed) → 852–1144 m route, 3 lanes × 3.5 m, stock
  two-way map; crossing the solid centre line or the road edge ends the episode (off-road).
* **Traffic**: MetaDrive's `PGTrafficManager`, density 0.15 (~36 cars per episode), placed randomly over every block
  but the first (from the seed). All cars drive from t = 0 (`release_all_at_reset`; MetaDrive's Trigger mode would
  hold each block's cars until the ego reaches it), starting at and cruising at their lane's speed under IDM: **left 45, middle 40,
  right 35 km/h** (re-evaluated every step from the car's current lane; set per car's policy in
  `sim.TrafficSpeeds`; IDM's built-in is 30).
* **Ego**: `static_default` vehicle — fixed dynamics (engine 800, brake 150, friction 0.9, 40° steering). The stock
  `default` model resamples engine/brake every episode. Spawns in the first block, random lane, 40 km/h (80 km/h was tried 2026-11-23: every episode crashed within ~6 s).
* **Seeds**: training draws map+traffic seeds from [0, 1000); evaluation uses 40 fixed seeds 1000–1039.
* **Step**: physics 0.02 s, control step 0.1 s; episode cap 1500 control steps (150 s).
* `force_destroy: true` (D-11): objects are destroyed on reset, so re-simulation of an episode is bit-exact.

## 3. Shared Gym env (`env.py`)
* `DrivingEnv` — one MetaDrive instance and the loop all methods share: one RL action → `method.decide()` → up to
  N control steps of `method.control()`; each control step applies `[steering, throttle_brake]`, extracts the ego
  and the cars within R = 50 m in the road (Frenet) frame, computes and logs every reward term. The decision's
  reward is the undiscounted sum over its control steps.
* A method (`m1.py`, `m2.py`, `m3.py`) is a small class with `period`, `action_space`, `decide(action)` and
  `control()`; it reads the world from the env (`env.ego`, `env.objects` within 50 m, `env.road`).
* **Observation** (identical for all methods; ground truth, what a tracking stack would hand a planner):
  * `ego` (8): v/v_max, d (road-centred, ÷ half-width), offset from the nearest lane centre (÷ lane width), heading
    error, last steering, last throttle_brake, target lane − current lane (M2 / M3 mid lane change; 0 for M1),
    time left ÷ the 150 s cap.
  * `lanes` (33): for the lane left of the ego, its own lane and the lane right of it: exists, then the nearest car
    **ahead** and **behind** in that lane, each [present, bumper gap ÷ 100 m, lateral offset ÷ lane width, relative
    speed ÷ v_max, closing speed ÷ gap (1/TTC, ≤ 5)], out to `observation.range` = 100 m. A car occupies every lane
    it reaches 0.3 m into, so a car mid lane change shows up in both.
* **Frenet frame** (`sim.RoadFrame`): s along the route, d positive to the left; lane centres d = 0, −3.5, −7.
* **Termination reasons**: success > collision > emergency_brake (M3) > off_road > timeout.

## 4. Methods
### Lane command (M2, M3; `reference.LaneTarget`)
Discrete: 0 KEEP (the default), 1 LEFT, 2 RIGHT. LEFT / RIGHT move the target lane by one (saturating at the edge
lanes). A lane change runs to completion: while the ego is more than `lane_action.settle` (0.5 m) from the target
lane's centre, commands are ignored. Training starts with KEEP at probability 0.9–0.95 (`training.init_default_prob`).

### M2 (`m2.py`)
Action `MultiDiscrete [3, 8]` = [lane command, speed level], levels 80, 70, …, 20, 0 km/h (`method.speeds_km_h`;
level 0 = 80 km/h, the default). d\* = the target lane's centre. Each decision builds a quintic from the current
lateral state (continuity with the previous reference) to d\* over 2 s and a speed ramp to v\* (+2 / −4 m/s²).
PID: heading [1.0, 0.005, 0], lateral [0.06, 0.0005, 0], speed [0.25, 0.0008, 0] with acceleration feed-forward
through the actuation table (gains tuned 2026-11-19, `scripts/matlab/m2_pid_tuning.m`). No safety layer: the RL
picks the speed, so braking for a car ahead is its job.

### M3 (`m3.py`, `mpc.py`)
Action `Discrete(3)` = lane command. Speed is the controller's: it drives toward v_cap (80 km/h) and brakes for cars.
1. **Lane plan** (`reference.LanePlan`) — built only when the target lane changes: a quintic d(s) from the ego (its
   d and heading) to the target lane centre over `lane_plan.duration` (2.5 s) of travel at the current speed,
   lengthened until its peak lateral acceleration is within 5 m/s²; flat after. Reference at v_cap.
   (`scripts/validate.py` flies hand-placed waypoints instead: `M3.plan_waypoints`, a piecewise-quintic spline.)
2. **Road corridor** (`m3.Corridor`, D-13) — one road-aligned box per MPC stage on the ego centre, rebuilt every
   control step: lateral faces fixed at the exterior lane centres widened by 0.3 m; cars' keep-out zones (car +
   1 m fore/aft + 0.3 m sideways, grown by the ego half-size) add a front face, back face or lateral face, decided
   **stage by stage** from where the stage point and the car's predicted zone are at that stage: behind it and
   overlapping laterally → front face; level beside it → lateral face on that side; past it → back face. A plan that
   draws level while laterally clear is passing (front → lateral → back: overtake and merge in front); one that runs
   into the car keeps the front face. Initial trajectory = reference at a ramp to v_cap (cars ignored: stopping is
   the MPC's job).
3. **MPC** — kinematic bicycle + 0.1 s steering lag, horizon 50 × 0.1 s, CasADi/IPOPT, costs on tracking, effort,
   rates, yaw rate and a barrier near the lateral faces. **Soft corridor**: each stage's 4 half-spaces get a slack
   σ ≥ 0 with cost 1000·Σσ + 1000·Σσ² (`mpc.soft_corridor`), an exact penalty: the plan stays inside the corridor
   whenever that is possible, and otherwise is the least-violating one instead of an infeasible problem. The largest
   slack is reported as `corridor_violation`. Speed ceiling v_cap (ramped down at 4 m/s² from the current speed so
   it is always feasible). Braking for cars is the MPC's job (accel down to −8 m/s²).
4. **Emergency-brake gate** — off (`ebrake.gate: false`). When on, a car in the planned path closed on faster than
   6 m/s² allows → full brake instead of the MPC; it pre-empted MPC braking it could do (−8 m/s²) and ended runs.
5. **Failures** — solve time counted in process CPU time; successful solves always used. A failed solve is retried
   once in the same step from a full-braking initial guess. Still failing → replay the previous plan (up to 10
   steps = 1 s, `mpc.replay_steps`), then hold speed and lane until a solve succeeds. Never brakes, never ends the
   episode.

**Failed runs** (`method.ebrake.ends_episode`): only the gate (`ebrake.gate`, off by default) ends an episode, as
`emergency_brake` with the collision penalty. In `scripts/validate.py run` (ends_episode false) it brakes instead.

## 5. Logging (`logs.py`, `evaluate.py`)
Every episode → one row in `episodes.csv` (termination, return and per-term returns, speed, jerk, gaps, solver
stats, `gated_brake_count`, lane changes, …). Every eval episode and every 50th training episode per env → full
per-step `.npz` (ego, objects, reward terms, controller diagnostics; M3 corridor every 10th step), replayable with
`scripts/view.py replay`; `scripts/plot.py episode <npz>` draws its overview plot and writes a per-step CSV.

## 6. Decision IDs referenced in code
| ID | decision |
|---|---|
| D-04 | Decision periods: M2 0.5 s, M3 1 s (fixed). |
| D-06 | Safe-set projection of M3 waypoints: disabled 2026-11-19, removed 2026-11-21 (in git history before then). |
| D-07 | Budgets, eval and checkpoint cadences counted in control steps (= simulated time), so methods with longer decision periods are compared at equal experience. Best checkpoint = highest eval success, ties by return. |
| D-11 | `force_destroy` for bit-exact re-simulation. |
| D-12 | Actuation map a ↔ throttle_brake: measured lookup table (brake is strongly nonlinear); MPC steering lag 0.1 s and lateral-accel bound 9.3 m/s² (was 4; the human demos' peak). |
| D-13 | M3 collision avoidance = hard convex corridor (no slack); now the road-aligned `RoadCorridor`. |
