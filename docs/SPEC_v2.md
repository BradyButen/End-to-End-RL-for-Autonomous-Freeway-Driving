# Architectural Specification — Where Should Learning Hand Off to Control?

**Nav 565, Team 16** · Simulated freeway driving in MetaDrive · **Spec v2.0 (2026-11-16)** — supersedes v1.0

> **How to read this version.** Text changed or added relative to v1.0 is marked **[v2]** (with the decision ID from
> [DECISIONS.md](DECISIONS.md) where one exists). A consolidated change list is at the top; v1.0 sections that are
> unchanged are kept verbatim or lightly condensed. Implementation details live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Summary of changes from v1.0

| § | Change | ID |
|---|---|---|
| 4.1 | Python **3.12**; MetaDrive from GitHub commit `fe54b03` (PyPI 0.4.3 does not install on 3.12) | D-01 |
| 4.3 | Map pinned to 50 m + 4 × 240 m = **1010 m**; **one-way** road (no oncoming lanes) | D-02 |
| 4.4 | Dynamics pinned via `VaryingDynamicsEnv` constant ranges; stopped IDM vehicles supported; background lane changes off | D-03, D-08 |
| 4.6 | Second scenario **`freeway_dense`** (22 cars, all lanes, whole route, 30 km/h) | D-17 |
| 4.7 | Off-road determined geometrically (MetaDrive's lane raycast is unreliable) | D-10 |
| 5.1 | Frenet sign convention fixed: reference = lane-0 centreline, d positive left | D-05 |
| 6 | BEV window 100 m side (±50 m), 5 channels (no navigation channel), channels-first | D-09 |
| 7.2 | M2 open items resolved (N2 = 5, T_traj = 2 s, PID details) | D-04, D-16 |
| 7.3 | M3: N3 = 10; safe-set reachability uses a lateral-acceleration budget; infeasibility rule; **MPC collision avoidance is a hard convex corridor** (was: soft constraints with slack); MPC steering-lag model + lateral-accel bound; speed reference from the corridor trajectory | D-04, D-06, D-12, D-13 |
| 7.3 | Actuation mapping = measured static lookup table | D-12 |
| 10 | Budgets and eval/checkpoint cadence counted in control steps; training on **Great Lakes** with GPU support | D-07, D-18 |
| 12 | `force_destroy` for bit-exact re-simulation; logs also store corridor tubes | D-11 |
| 19 | Most open decisions resolved (see §19) | — |
| 20 | New MetaDrive pitfalls #9–#12 | D-10, D-11, D-12 |

---

## 1. Purpose

### 1.1 Research question
How much of the driving task should a reinforcement-learning (RL) policy control, and how much should a model-based
controller control? We study this along a **ladder of RL authority** in a simulated freeway scenario.

### 1.2 The methods

| ID | RL decides | Controller executes | Safety responsibility |
|---|---|---|---|
| **M1** End-to-end | Steering + throttle/brake, every control step | — | RL (learned via reward) |
| **M2** RL + PID | A short trajectory: target lateral offset + target speed | Quintic trajectory generator + cascaded PID | RL (learned via reward); **no safety layer** |
| **M3** RL + MPC | Sparse, long-horizon waypoints (up to 5) — *where* to maneuver | MPC plans and executes the path inside a **hard collision-free corridor [v2]** | MPC (+ safe-set projection of waypoints) |
| **B0** Baseline | — | MetaDrive rule-based `IDMPolicy` | Rule-based |

### 1.3 Framing constraints
- M2 and M3 do **not** share an action interface; results describe how outcomes change as authority moves from RL to
  controller, not a PID-vs-MPC effect.
- Intended finding: in M2 the RL must learn safety; in M3 it does not.
- "Deployment" = loading frozen weights and running evaluation episodes in simulation.

## 2. Scope
**Core:** M1, M2, M3, B0 on the fixed freeway; PPO training; configurable rewards; evaluation harness; custom logging;
MetaDrive-native visualisation. **[v2]** Two scenarios: `freeway_pass` (spec scenario) and `freeway_dense`.
**Reach goals:** §18. **Out of scope:** hardware, sys-ID, imitation warm starts, MetaDrive's `ExpertPolicy`, external
experiment trackers.

## 3. System overview
Unchanged from v1.0 (policy → method wrapper → MetaDrive → state extraction → reward/logging/visualisation), with
**[v2]** M3's controller block reading: *waypoints → safe-set projection → spline → corridor (initial trajectory +
convex boxes, rebuilt every 1 s) → MPC with hard corridor constraints (every 0.1 s)*.

Integration pattern unchanged: Gymnasium wrappers (`M1Wrapper`, `PIDWrapper`, `MPCWrapper`, `B0Wrapper` around a shared
`CoreEnv`); controllers are plain Python without MetaDrive/SB3 dependencies; MetaDrive source is not modified.

## 4. Simulation environment & scenario

### 4.1 Platform
- MetaDrive (`metadrive-simulator`). **[v2]** Python **3.12**; MetaDrive installed from GitHub commit
  `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1` (reports 0.4.3) because the PyPI 0.4.3 wheel is capped below 3.12. (D-01)
- One MetaDrive instance per process; parallel environments use one subprocess each.

### 4.2 Timing
Physics step 0.02 s; control step 0.1 s (`decision_repeat = 5`); M1 decision = 1 control step;
**[v2]** M2 decision = **N2 = 5** control steps (0.5 s); M3 decision = **N3 = 10** control steps (1 s). (D-04)

### 4.3 Map
- **[v2]** Fixed straight freeway built from an explicit block list: first block 50 m + four straight blocks of 240 m =
  **1010 m** route. 3 lanes × 3.5 m. **One-way**: no oncoming carriageway. (D-02)
- Same map for training and evaluation.

### 4.4 Actors
- **Ego:** `DefaultVehicle` geometry; dynamics pinned to engine 800, brake 150, steering 40°, friction 0.9, cap
  80 km/h. **[v2]** Pinned through MetaDrive's `VaryingDynamicsEnv` with equal min/max ranges. (D-03)
- **Lead ("semi" stand-in):** `XLVehicle`, ego lane, IDM, no lane changes, slow target speed.
- **Background traffic:** IDM vehicles with per-vehicle cruise speeds. **[v2]** Lane changes disabled; vehicles with a
  target speed below 1 km/h hold full brake (stationary obstacles). (D-08)
- Spawned by a custom scenario manager.

### 4.5 Speeds
Global target speed 70 km/h; speed cap 80 km/h (M2 action range, M3 MPC constraint).

### 4.6 Per-episode randomisation and scenarios
Fixed map; seeded randomisation of lead speed, ego–lead gap, ego lane and speed, background traffic. Disjoint training
seeds `[0, 900000)` and 20 fixed evaluation seeds `1000000–1000019`.

| | `freeway_pass` (spec scenario) | **[v2]** `freeway_dense` (D-17) |
|---|---|---|
| ego | s₀ = 80 m, any lane, 40–70 km/h | same |
| lead | XL, ego lane, 30–45 km/h, gap 25–60 m | XL, 30 km/h, gap 25–60 m |
| background | 0–4 cars, other lanes, −60…+100 m around the ego, 55–75 km/h, 15 m spacing | **22** cars, **all lanes**, **whole route**, **30 km/h**, 10 m spacing |

### 4.7 Termination
Route completion, collision, off-road, timeout (1200 control steps = 120 s, reported as truncated). **[v2]** Off-road
= vehicle centre outside the road (Frenet bounds) or beyond the route ends, or contact with continuous lines /
sidewalk; MetaDrive's raycast-based `on_lane` is not used (pitfall #9). (D-10)

## 5. Shared state representations
- **RoadFrame**: **[v2]** reference line = centreline of MetaDrive lane 0 (leftmost); `s` = arc length from the route
  start; `d` positive to the left; lane centres d = 0, −3.5, −7.0; road edges −8.75 / +1.75. (D-05)
- **EgoState**, **TrackedObject / ObjectList** (vehicles within R = 50 m, ground truth) — unchanged.

## 6. RL observation (all methods identical)
`Dict{bev, ego}`:
- `bev`: MetaDrive multi-channel top-down image, 84 × 84, floats in [0, 1]. **[v2]** Window side length 100 m
  (MetaDrive's `distance` is the side length, so ±50 m = R); **5 channels** (road network, past positions, 3 stacked
  traffic frames — MetaDrive's multi-channel observation has no navigation channel); delivered **channels-first
  (5, 84, 84)** for SB3. (D-09)
- `ego`: `v/v_max`, `d` (relative to the road centre, ÷ road half-width), `ψ_err`, previous steering, previous
  throttle_brake.

## 7. Method specifications

### 7.1 M1 — End-to-end
Unchanged: `Box([-1,-1],[1,1])` = `[steering, throttle_brake]` every control step.

### 7.2 M2 — RL + PID
- Action → `d*` ∈ [d_min − 1 m, d_max + 1 m] (not clipped), `v*` ∈ [0, 80 km/h].
- Reference: quintic lateral profile to `d*` with zero end rate/accel over **T_traj = 2 s [v2]**, speed ramp
  (+2 / −4 m/s²), re-planned every **N2 = 5 [v2]** control steps. **[v2]** Replans start from the previous
  reference's lateral rate and acceleration (continuity). (D-16)
- Tracking: heading PID + lateral PID (MetaDrive LaneChangePolicy gains) → steering; **[v2]** speed PID
  (0.6, 0.02, 0) + feed-forward of the reference acceleration through the actuation table → throttle_brake. (D-16)
- No safety layer.

### 7.3 M3 — RL + MPC
- **Action:** 5 × (Δs ∈ [5, 40] m, d ∈ [d_min − 1, d_max + 1]) + count channel → k ∈ 1…5 (logged). **[v2]** (D-04)
- **Safe-set projection** (once per decision): drivable band (ego half-width + 0.2 m), outside inflated
  constant-velocity footprints at the estimated arrival time (±0.5 s), reachable. **[v2]** Reachability: ≥ 2 m after
  the previous waypoint, ≤ v_cap·10 s ahead, lateral shift ≤ min(0.16·Δs, a_lat·Δs²/(5.77·v²)) with a_lat = 3 m/s².
  **[v2]** Infeasible waypoint → it and all later ones are dropped, each charged 10 m projection distance; none left →
  hold the current lane. (D-06)
- **Reference:** piecewise-quintic spline from the ego pose through the projected waypoints (zero slope/curvature at
  waypoints), extended 80 m.
- **[v2] Corridor** (D-13): an initial trajectory along the spline (speed ramping to the target, slowing to stop 2 m
  behind any car whose inflated, predicted footprint blocks the path) and one convex box per stage over 3 s, grown
  from the trajectory point to the nearest inflated car / road edge (lateral faces first, then longitudinal; caps
  5.25 / 20 / 10 m). Boxes are road-aligned, or rotated to the trajectory heading during lane changes. Cars are
  inflated by the ego footprint so the box constrains the ego centre. Rebuilt every 10 control steps and on every new
  RL reference.
- **MPC** (every control step): kinematic bicycle **[v2]** with a 0.1 s first-order steering lag; N = 20, dt = 0.1 s;
  cost = path tracking to the corridor trajectory + **[v2] speed error to the corridor trajectory's speed** (equals the
  global target unless slowing for a blocking car) + input magnitude and rate penalties. Constraints: |δ| ≤ 40°,
  a ∈ [−6, 2.5] m/s², steering-rate bound, 0 ≤ v ≤ v_cap, **[v2]** |v²·tan δ / L| ≤ 4 m/s², and **[v2] the corridor
  as hard constraints (4 half-spaces per stage); no slack variables** (v1.0: soft collision constraints with slack).
- Output: steering = δ / 40°; a → throttle_brake through **[v2] a measured static lookup table** (D-12).
- Solver: CasADi + IPOPT, warm-started.
- **Failure handling [v2]:** failed solve → rebuild the corridor from the current state and retry once → replay the
  previous plan for ≤ 5 steps → emergency brake with lane hold. Every fallback logged with its reason.
- Solve time is first-class (mean, p95, max logged).

### 7.4 B0
MetaDrive `IDMPolicy` with lane changes, target speed 70 km/h, evaluation only.

## 8. Wrapper step semantics
Unchanged: one RL transition spans up to N control steps; reward summed within a decision;
`γ_method = γ_ref^(T_decision / 0.1 s)` with γ_ref = 0.99. **[v2]** GAE λ is not scaled. (D-15)

## 9. Reward framework
Unchanged design (weighted named terms, all computed and logged every step; MetaDrive's reward ignored).
Catalog as v1.0. **[v2]** R0 terminal replacement priority success > off_road > collision; example variants
`r1_sparse`, `r2_dense_comfort` exist but are untuned; `projection_penalty` weight 0 in R0. (D-14)

## 10. RL training
PPO (SB3, `MultiInputPolicy`, `normalize_images=False`) for M1–M3; SubprocVecEnv; ≥ 3 seeds per (method, variant);
periodic and best-on-eval checkpoints; CSV logger.
**[v2]** Training budget, evaluation cadence and checkpoint cadence are counted in **control steps** (simulated
experience), with an evaluation at step 0 (D-07). **[v2]** Training runs on **Great Lakes** (GPU for PPO updates when
available; simulation and MPC on CPU); the dev machine is for development and validation only (D-18). Default
budget 2M control steps per run is a placeholder pending learning curves.

## 11. Evaluation
Unchanged protocol (20 fixed held-out seeds, deterministic actions, method-agnostic harness) and metric set.
**[v2]** Min gap and TTC use road-aligned boxes (D-19). Learning-curve x-axis still open (all three counters logged).

## 12. Logging
Three levels (control step, decision, episode) and run layout as v1.0, plus `device.txt`. Full logs for all
evaluation episodes and every 50th training episode per env. **[v2]** Full logs also contain each corridor rebuild
(boxes + initial trajectory) and MPC predictions. **[v2]** `force_destroy` makes re-simulation bit-exact, which replay
relies on (D-11).

## 13. Visualisation
MetaDrive's own viewer, live and replay (replay = deterministic re-simulation from seed + checkpoint, overlays from
logs). **[v2]** Overlays include corridor boxes. Analysis plots: learning curves, comparison table, episode time series.

## 14. Configuration
Unchanged (YAML merge base → scenario → method → reward → CLI `--set`; merged config saved per run).

## 15. Repository layout
As v1.0 with package `handoff`, plus **[v2]** `methods/m3/corridor.py`, `methods/common/actuation.py`,
`methods/m3/spline.py`, `scripts/validate.py`, `scripts/profile_timing.py`, `scripts/calibrate_actuation.py`,
`scripts/debug/`, `slurm/`, `docs/`.

## 16. Interface contracts
As v1.0. **[v2]** Additions: `CorridorBuilder.build(ego, reference, objects, road) -> Tube` (half-spaces `A`, `b`,
corners, initial trajectory per stage); `project_waypoints` keeps the v1.0 signature (a detailed variant also returns
validity and footprints).

## 17. Validation requirements (before any RL training)
Unchanged list; implemented in `scripts/validate.py` (+ unit tests). Current results: [STATUS.md](STATUS.md).

## 18. Reach goals
Unchanged: adaptive MPC under dynamics mismatch (now a config change via D-03), robustness across traffic scenarios,
custom semi-truck model.

## 19. Open decisions
| Item | v2 status |
|---|---|
| M2 `T_traj`, `N2` | **Resolved**: 2 s, 5 (D-04) |
| M3 decision trigger | **Resolved**: fixed N3 = 10 (D-04) |
| M3 Δs ranges, lateral margin | **Resolved**: [5, 40] m, 1 m (D-04) |
| MPC horizon, dt, rate, solver | **Resolved for now**: N = 20, 0.1 s, every control step, IPOPT |
| Actuation mapping | **Resolved**: measured lookup table (D-12) |
| Safe-set infeasibility | **Resolved** (D-06) |
| Ego vector contents | **Resolved**: §6 baseline |
| Reward variants beyond R0 | Open (examples r1, r2 exist) |
| Learning-efficiency axis | Open (all counters logged) |
| PPO hyperparameters, budget | Open — set from cluster profiling |
| Visualisation | **Resolved**: live 3D + top-down, replay by re-simulation |
| Scenario numeric ranges | **Resolved** for both scenarios (§4.6) |
| **[v2]** PID / MPC / corridor tuning | Open — see STATUS.md |

## 20. Known MetaDrive pitfalls
1–8 as v1.0 (speed cap, dynamics randomisation, slow default traffic, `WaypointPolicy` teleports, one instance per
process, top-down rendering cost, continuous-line termination, Python 3.13). **[v2]** Additionally:
9. **`on_lane` raycast misses at lane-mesh seams** → spurious off-road terminations and deleted traffic; use geometry (D-10).
10. **Object reuse breaks determinism** across resets (~1e-5 m noise amplified by closed loops); use `force_destroy` (D-11).
11. **Brake response is strongly nonlinear** (≈ −8 m/s² at −0.3, −15.8 m/s² at −1) and steering acts with ~0.1 s lag (D-12).
12. **Stock IDM** divides by its target speed (crashes at 0 km/h) and its lane changes overshoot at 70 km/h (B0 off-road ~15 %).
13. **PyPI wheel vs Python 3.12** (see D-01). Under WSL, set `GALLIUM_DRIVER=d3d12` for GPU-accelerated 3D viewing.

## References
As v1.0.
