# Project status — 2026-11-16

**Phase:** implementation complete, validation in progress, **no training run yet**. Next: controller review
(PID gains, MPC/corridor behaviour) by the team, then cluster profiling and the first training runs.

## 1. What exists
* Full pipeline per the spec (v2): MetaDrive env with fixed one-way 1 km freeway and two scenarios, M1/M2/M3/B0
  wrappers, PID + quintic trajectory, safe-set projection, spline, **corridor MPC (hard constraints)**, configurable
  rewards (R0 + 2 example variants), PPO training with control-step cadences, parallel evaluation, three-level logging,
  3D/top-down viewer with overlays, replay by bit-exact re-simulation, plots, Great Lakes job scripts.
* 31 unit tests (no simulator) and a 6-check validation suite against MetaDrive.

## 2. Validation results (dev machine, WSL2, CPU)

| check (spec §17) | result | notes |
|---|---|---|
| unit tests (`pytest`) | **pass** (31/31) | incl. MPC on a kinematic plant: passes slow lead, stops for a stopped car, full fallback chain |
| lint (`ruff`) | **pass** | |
| 1 env (`validate --check env`) | **pass** | spawns match the spec (< 5 cm, < 0.05 m/s), dynamics pinned every reset, lead holds its speed (< 0.08 m/s), RoadFrame round-trip exact, forced off-road terminates, re-simulation bit-exact |
| 2 throughput | measured | see §4 |
| 3 PID standalone (`--check pid`) | **FAIL** | lane-change cross-track max **0.59 m** (tolerance 0.30), settled **0.27 m** (0.10); speed hold fine (0.07 m/s). Gains untuned (MetaDrive LaneChangePolicy). |
| 4 MPC standalone (`--check mpc`) | **partial** | (a) pass slow lead: **pass** (min gap 1.4 m, solves 3.3 ms mean / 4.8 ms p95, no fallbacks); (b) stopped car in path, safe set disabled: **FAIL — collision at 3.9 m/s**; (c) fallback chain: **pass** (shifted plan → emergency brake → recovery) |
| 5 safe-set unit tests | **pass** | |
| 6 full-stack smoke (`--check smoke`) | **pass** | M1/M2/M3 random-action episodes complete; logs consistent (terms, decisions, returns) |
| 7 B0 through the harness (`--check b0`) | **pass** | 20 eval seeds, `freeway_pass`: success 85 %, off-road 15 %, collision 0 %, mean 63.7 km/h, 50.7 s to complete |
| train plumbing (`--check train_plumbing`) | **pass** | `train.py` end-to-end for M1/M2/M3 (vec envs, callbacks, eval subprocess, checkpoints, CSVs) — a few hundred steps, no real learning |

Reports: `runs/validation/*.json` (regenerate with `python scripts/validate.py --all`).

## 3. Open issues for review (not tuned — left for the team)
1. **M3 hits a stopped car** in the MPC standalone check (waypoints straight through it, projection off): the corridor's
   initial trajectory should stop the ego 2 m behind the inflated car and the hard tube should forbid contact, yet it
   collides at 3.9 m/s. Also one M3 random-action smoke episode ended in a collision. With hard constraints this
   points at a corridor/MPC issue (candidates: the car entering the 50 m perception range late relative to the
   4 m/s² planning decel, stage-0/rebuild infeasibility → fallback, box growth when the trajectory point is blocked).
   Debug with `scripts/debug/probe_mpc_md.py stopped`.
2. **PID tracking** exceeds the stated tolerances during lane changes (§2).
3. **MPC weights** are first guesses; no systematic tuning.
4. **B0 off-road 15 %**: MetaDrive's IDM lane changes into lane 0 overshoot at 70 km/h (heading 8–11°) and touch the
   solid left edge line. Baseline behaviour, not a harness bug (IDM's lateral gains are tuned for 30 km/h).
5. **Determinism under load**: IPOPT's 0.1 s CPU limit makes M3 outcomes machine-load dependent when hit.
6. Metrics `min_gap`/`min_ttc` are road-aligned-box approximations (under-report gaps when the ego is strongly yawed).
7. **Intermittent native crash**: once, two M3 env worker processes segfaulted (signal 11, inside the Python
   interpreter — a native-library crash) during a benchmark with 8 parallel envs; the main process then died with a
   broken pipe and the other workers hung. Not reproduced in 8 retries (64 env start-ups). A worker crash kills a
   training job, and **resuming from a checkpoint is not implemented** — worth adding before long cluster runs.

## 4. Performance (dev machine, random actions, BEV on)

`scripts/benchmark_throughput.py` (control steps per wall second / simulated seconds per wall second):

| method | 1 env | 4 envs | 8 envs |
|---|---|---|---|
| M1 | 317 / 31.7 | 438 / 43.8 | 576 / 57.6 |
| M2 | 290 / 29.0 | 606 / 60.6 | 795 / 79.5 |
| M3 | 110 / 11.0 | 205 / 20.5 | 198 / 19.8 |

MPC solve time: 5.0 ms mean / 7.2 ms p95 (1 env), 8.1 / 24.6 ms (4 envs), 13.0 / 38.9 ms (8 envs).
**Scaling is sub-linear on this machine** (4 envs ≈ 1.4–2×; M3 gains nothing from 4 → 8 envs while its solves slow
down — possibly per-worker thread oversubscription in the numerical libraries; not investigated. The SLURM scripts set
`OMP_NUM_THREADS=1`; local runs do not). Random actions end episodes within seconds, so resets
(costlier with `force_destroy`) are over-represented; trained policies will run longer episodes. Cluster scaling must
be measured there before setting budgets — the earlier per-run estimates (M1 ≈ 1 h, M2 ≈ 0.4 h, M3 ≈ 0.75 h on a
16-core GPU node) assumed linear scaling and are optimistic.

`scripts/profile_timing.py` (1 env): ms per control step M1 5.7, M2 3.7, M3 9.2; one PPO update on a 256-decision
buffer ≈ 1.6 s on one CPU thread (≈ 6.3 ms per decision — dominates M1 on CPU; the GPU removes most of it).

## 5. Untested
* Anything on Great Lakes (`slurm/` scripts, CUDA PyTorch, asset download on the cluster).
* Training beyond the plumbing check; PPO hyperparameters; reward variants r1/r2.
* The 3D overlays for the corridor boxes in the live viewer; replay mode end-to-end on a trained checkpoint.
* `freeway_dense` beyond B0 / viewer runs (not part of the validation suite).
* Resuming training from a checkpoint (not implemented).

## 6. Next steps (proposed)
1. Team review of the controllers (issues 1–3).
2. Set up Great Lakes (`slurm/setup_greatlakes.sh`), run `profile_timing.py` and the benchmark there.
3. Pick the training budget and PPO overrides from those numbers; short pilot runs (1 seed per method) to see
   learning curves; then the 3-seed sweep per reward variant.
4. Push to the team GitHub repo once it exists.
