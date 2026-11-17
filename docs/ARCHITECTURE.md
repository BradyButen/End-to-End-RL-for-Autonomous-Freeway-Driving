# Architecture — how the code works

Package `handoff` (in `src/handoff/`). This document follows one RL decision from the policy down to MetaDrive and
back, then describes each module. Design rationale and deviations from the spec are in [DECISIONS.md](DECISIONS.md);
what works and what doesn't is in [STATUS.md](STATUS.md).

## 1. The big picture

```
SB3 PPO (MultiInputPolicy: CNN on bev + MLP on ego)
   │  action (method-specific), once per decision
   ▼
Method wrapper  (wrappers/methods.py)            M1Wrapper | PIDWrapper (M2) | MPCWrapper (M3) | B0Wrapper
   decide(action)  -> decision + Reference       (M3: decode waypoints -> safe-set projection -> spline)
   for j in range(N):                            N = 1 (M1, B0), 5 (M2), 10 (M3) control steps
       u = control(j)                            M2: trajectory + cascaded PID; M3: corridor + MPC
       core.control_step(u)  ───────────────►  CoreEnv (wrappers/core.py)
   core.end_decision(...)                          md.step(u)  -> FreewayMetaDriveEnv (sim/metadrive_env.py)
   return obs, ΣR, terminated, truncated, info     extract EgoState / ObjectList (sim/state.py, RoadFrame)
                                                   reward terms (rewards/), recorder (logging/recorder.py)
```

* One control step = one MetaDrive `env.step` = 0.1 s (5 physics steps of 0.02 s).
* Reward is summed over the control steps of a decision; PPO's discount is scaled so every method has the same
  lookahead in seconds: `γ_method = 0.99 ** (T_decision / 0.1 s)` (`wrappers/factory.py: method_gamma`).
* Every env runs in its own process (MetaDrive allows one instance per process): SB3 `SubprocVecEnv` for training,
  separate worker processes for evaluation (`eval/harness.py`).

## 2. Simulation layer (`sim/`)

### `metadrive_env.py` — MetaDrive integration (MetaDrive source is not modified)
* **`FreewayMetaDriveEnv`** subclasses MetaDrive's `VaryingDynamicsEnv`:
  * **Map**: built from an explicit block list (`MapGenerateMethod.PG_MAP_FILE`) — first block (50 m) + four straight
    blocks of 240 m = 1010 m, 3 lanes × 3.5 m. **One-way** (`OneWayPGMap` passes MetaDrive's `remove_negative_lanes`
    to every block; `sim.map.one_way`).
  * **Ego dynamics pinned** (engine 800, brake 150, friction 0.9, steering 40°) by giving `VaryingDynamicsEnv`'s
    `random_dynamics` ranges with equal bounds. Changing them to real ranges is reach goal 1.
  * **Observation**: MetaDrive's `TopDownMultiChannel` (84×84, 5 channels: road, past positions, 3 stacked traffic
    frames), window 100 m wide (±50 m = perception range R).
  * **Out-of-road** uses a geometric test (vehicle centre outside the road in Frenet coordinates) plus MetaDrive's
    continuous-line/sidewalk contacts. MetaDrive's own `on_lane` raycast randomly misses at lane-mesh seams.
* **`ScenarioTrafficManager`** spawns exactly the lead + background vehicles of the episode's `ScenarioSpec` with
  per-vehicle IDM cruise speed and lane-change switch (`ScenarioIDMPolicy`). Vehicles are removed only when they
  geometrically leave the road.
* **`ScenarioAgentManager`** places the ego (lane, s, speed) from the spec.
* **`EgoIDMPolicy`** is the B0 baseline (stock IDM + lane changes at the global target speed).
* The spec for the upcoming reset is handed to the managers through a module-level holder (`_EPISODE`), set right
  before `reset` — safe because there is one MetaDrive per process.

### `road.py` — `RoadFrame` (Frenet frame)
* `s` = arc length from the route start along the centreline of lane 0; `d` = lateral offset, **positive to the left**.
* Lanes (MetaDrive index 0 = leftmost) are at `d = 0, -3.5, -7.0`; road edges `d_max = +1.75`, `d_min = -8.75`.
* Pure numpy (`to_frenet`, `to_world`, `heading_at`, `nearest_lane`); `RoadFrame.straight(...)` builds a synthetic
  road for tests without MetaDrive.

### `state.py`
`EgoState`, `TrackedObject` (= `ObjectList` entries: all vehicles within R = 50 m, ground truth), gap and
time-to-collision helpers used by metrics.

### `scenario.py`
`FreewayPassScenario.sample(rng)` → `ScenarioSpec` (ego, lead, background). Fully determined by the episode seed.
Options: background lanes `other` (not the ego lane) or `all`; placement relative to the ego (`s_offset`) or along the
whole route (`s_range`).

## 3. Core env (`wrappers/core.py`)

`CoreEnv` is the Gymnasium env shared by every method:
* **reset**: picks the episode seed (training: drawn from `[train_low, train_high)` by a per-env RNG; evaluation:
  passed explicitly as `options={"episode_seed": s}` from the fixed eval list), samples the scenario, resets MetaDrive.
* **observation** `Dict{bev: (5, 84, 84) float [0,1] channels-first, ego: [v/v_max, d (normalised to the road),
  ψ_err, steering, throttle_brake]}`.
* **control_step(u)**: steps MetaDrive once, re-extracts state, computes **all** reward terms, records a control-step
  log row, calls the optional viewer hook.
* **end_decision(...)**: records the decision-level row; on termination computes the episode summary
  (`info["episode_summary"]`) and writes the full npz log if this episode is logged in full.

## 4. Methods (`wrappers/methods.py`, `controllers/`)

| | action | decode | controller (every 0.1 s) |
|---|---|---|---|
| **M1** | `[steering, throttle_brake]` | identity | — |
| **M2** | `[d*, v*]` ∈ [-1,1]² | `d*` ∈ [d_min−1, d_max+1] (not clipped), `v*` ∈ [0, 80 km/h] | `trajectory.py` quintic lateral profile + accel-limited speed profile (re-planned every 5 steps), `pid.py` heading+lateral PID → steering, speed PID + feed-forward → throttle |
| **M3** | 11 numbers | 5 × (Δs ∈ [5,40] m, d) + count k ∈ 1..5 | see below |
| **B0** | ignored | — | MetaDrive `IDMPolicy` drives the ego |

### M3 pipeline (once per decision = 10 control steps)
1. **Decode** to k Frenet waypoints `(s_i, d_i)`, increasing in s.
2. **Safe-set projection** (`controllers/safe_set.py`): each waypoint is moved to the nearest point that is
   (a) on the road with room for the ego, (b) outside every car's inflated footprint predicted at the ego's arrival time,
   (c) reachable (≥ 2 m after the previous waypoint, lateral shift within a lateral-acceleration budget). Infeasible
   waypoints are dropped (10 m penalty each); none left → "hold current lane". Distances feed `projection_penalty`.
3. **Spline** (`controllers/spline.py`): piecewise-quintic path from the ego pose through the waypoints (zero slope
   and curvature at each waypoint → no overshoot), extended 80 m.
4. **Corridor** (`controllers/corridor.py`, rebuilt every 10 steps and on each new reference):
   * initial trajectory along the spline, speed ramping to 70 km/h, slowing to stop 2 m behind any car blocking the path;
   * one box per stage (3 s, 30 stages) around it, grown face by face until an inflated car or the road edge;
     road-aligned, or rotated to the trajectory heading while lane-changing; returned as 4 half-spaces `A p ≤ b`.
5. **MPC** (`controllers/mpc.py`, every control step): CasADi + IPOPT, N = 20 × 0.1 s, kinematic bicycle + 0.1 s
   steering lag, tracking the initial trajectory; **hard** tube constraints on the ego centre, input/rate/speed bounds,
   lateral-acceleration bound. Output → steering (÷40°) and throttle/brake via the measured lookup table
   (`controllers/actuation.py`, `actuation.table` in `configs/base.yaml`).
6. **Fallbacks**: failed solve → rebuild tube + retry once → replay previous plan (≤ 5 steps) → emergency brake with
   lane hold. Every fallback is logged with its reason.

## 5. Rewards (`rewards/`)
* `terms.py`: `progress`, `speed`, `speed_tracking`, `success`, `collision`, `off_road`, `lane_center`, `jerk`,
  `steer_rate`, `time`, `projection_penalty` — each with a docstring definition and units.
* `composer.py`: weighted sum from `configs/reward/<variant>.yaml`. `terminal_mode: replace` reproduces MetaDrive's
  rule (on termination the step reward is replaced by the first active terminal term, priority success > off_road >
  collision). Every term is computed and logged every control step even at weight 0, so runs can be re-scored offline.
* Variants: `r0` (MetaDrive default), `r1_sparse`, `r2_dense_comfort` (examples, untuned).

## 6. Training, evaluation, logging
* `rl/train.py`: SB3 PPO, `MultiInputPolicy`, `normalize_images=False`, γ scaled per method, device `auto` (CUDA when
  available). Budget, eval cadence and checkpoint cadence are counted in **control steps**.
* `rl/callbacks.py`: writes `episodes.csv`, logs counters (control steps / decisions / simulated time), evaluates at
  step 0 and every `eval.freq_control_steps` on the 20 fixed seeds (deterministic actions) in separate processes,
  saves `best.zip` (by success rate, then return), periodic checkpoints, `final.zip`.
* `eval/harness.py`: method-agnostic episode runner + parallel evaluation workers; `eval/metrics.py`: episode metrics
  (success/collision/off-road/timeout, time to complete, mean speed, min gap, min TTC, RMS jerk / lateral accel /
  steering rate, tracking error, MPC solve time & fallbacks, projection distance, k histogram, lane changes).
* `logging/recorder.py`: control-step, decision and episode records; full logs (`episodes/<id>.npz`) for every
  evaluation episode and every 50th training episode per env. `load_episode(path)` reads them back.
* Run directory: `runs/<method>/<reward>/seed_<n>/<timestamp>/{config.yaml, git_commit.txt, device.txt,
  train_progress.csv, episodes.csv, episodes/, checkpoints/}`.

## 7. Visualisation (`viz/`)
* `viewer.py` + `scripts/view.py`: live runs in MetaDrive's 3D window (or `--topdown` 2D) with overlays drawn by
  `overlays.py` — reference (yellow), MPC prediction (cyan), raw/projected waypoints (red/green), safe-set footprints
  (magenta), corridor boxes (blue), status text. Replay re-simulates a logged episode from its seed + checkpoint
  (bit-exact) and redraws overlays from the log.
* `plots.py` + `scripts/plot.py`: learning curves (x = control steps / decisions / sim time), comparison table,
  episode time series.

## 8. Configuration
Plain YAML merged in order `configs/base.yaml → scenario/<s> → method/<m> → reward/<r> → --set key.sub=value`
(`config.py`). The merged config is saved with every run. Main knobs:

| file | what |
|---|---|
| `base.yaml` | timing, map, ego dynamics, speeds, BEV, actuation table, logging, eval, training, PPO |
| `scenario/freeway_pass.yaml` | sparse: lead + 0–4 background cars around the ego, other lanes only |
| `scenario/freeway_dense.yaml` | dense: 22 cars along the whole route, all lanes, 30 km/h |
| `method/m2.yaml` | decision period, trajectory horizon/limits, PID gains |
| `method/m3.yaml` | waypoint ranges, safe set, spline, MPC weights/limits, corridor |
| `reward/*.yaml` | reward weights and terminal mode |

## 9. Tests and tools
* `tests/` (pytest, no simulator): config, RoadFrame, scenario, metrics, rewards, trajectory, PID, spline, safe set,
  MPC + fallback chain on a kinematic plant.
* `scripts/validate.py`: spec §17 checks against MetaDrive (`--check env|pid|mpc|smoke|b0|train_plumbing`, `--all`).
* `scripts/benchmark_throughput.py`, `scripts/profile_timing.py`: throughput / time breakdown.
* `scripts/calibrate_actuation.py`: re-measures the throttle/brake table and steering response.
* `scripts/debug/`: per-step printouts of MPC scenarios (stopped car, pass lead, prediction vs MetaDrive).
