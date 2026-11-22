# Training

RL setup, why each setting is what it is, what the pilot taught us, and how to run on Great Lakes.
System design: [DESIGN.md](DESIGN.md).

## 1. Algorithm
PPO (Stable-Baselines3, `MultiInputPolicy`: NatureCNN on the BEV + MLP on the ego vector, shared by actor and
critic), identical for M1/M2/M3. Budgets and cadences are counted in **control steps** (0.1 s of simulated driving),
summed over parallel envs, so every method gets the same simulated experience (D-07).

## 2. Reward: `r4_straight` (the only reward)
Per control step, summed within a decision; terminal terms added on the last step. Raw terms: `src/handoff/reward.py`;
weights and the full arithmetic: `configs/r4_straight.yaml`. Tuned 2026-11-21 by driving the env by hand
(`scripts/drive.py`) and on M3 scenarios (`scripts/validate.py run`), watching the return live.

| term | weight | why |
|---|---|---|
| progress (Δs, m) | +0.2 /m | dense signal: tells an untrained policy which way is good (~+200 for a full route, whatever the speed) |
| time | −0.3 /step (−3 /s) | arriving sooner is directly better |
| success | +500 | finishing beats every non-finish; mostly felt near the end (the ~10 s discount horizon) |
| collision | −4000 | also charged for an M3 gate stop (`emergency_brake`, gate off by default). Until 2026-11-22 also M3 `infeasible_plan` (no longer a termination) |
| off_road | −4000 | sets the speed-vs-safety trade-off: a pass pays while its crash risk is below ~26 % (−2000 would allow ~46 %) |
| speed_gap | −6.0 × ((v − 80 km/h)/80 km/h)² per step | pushes toward the cap, quadratically: −15/s at 40 km/h, −3.75/s at 60, −0.94/s at 70 |
| heading | −2.7 per rad of heading error beyond 0.5°, per step | damps oscillation (weaving ±0.5 m for 100 s ≈ −110); summed over a lane change ≈ lateral distance / speed (−4.0 to −7.9) |
| lane_keep | −0.2 per step outside ±0.57 m of a lane centre (body ~25 cm inside the lines) | no riding the lines (−2/s); 0 inside the lane rather than an in-lane bonus, which would cancel the time cost |
| proximity | −5 × (G − gap)/G per step for a car overlapping sideways; G = 5 m to a car ahead, 2 m to a car behind | steep near-miss shaping before the crash (−30/s at 2 m behind a car, −50/s at contact); cars beside the ego never count |
| lane_change | 0 (logged only) | the `lane_changes` metric; passes are paid for by heading + lane_keep (≈ −5.5 to −11 per lane change) |

Weights raised 2026-11-22 to push for speed (time, speed_gap ×3, success ×10, crash ×2). Expected episode scores:
fast finish at 80 km/h ≈ +565 > finish at 70 km/h ≈ +483 > finish behind 40 km/h traffic ≈ −920 > crash ≈ −3935
(late) to −4000 (early) > crawl to the timeout at 20 km/h ≈ −5350 > standing still for 150 s ≈ −9450. Known:
crawling and standing still score below a crash, so a policy that cannot drive yet (early M1) loses less by
crashing; if that shows up, charge a crash the speed penalty it skips. Earlier runs (round 3, first warm batch) used
time 0.1, speed_gap 2, success 50, crash −2000: compare returns only within one weight set. Full arithmetic: `configs/r4_straight.yaml`.
History: r0 (MetaDrive's default, pilot round 1: a finish earned ~+1040 and a crash only −5) and r3_fast (round 2:
progress 0.1, time, terminals ±50 / −500, lane_change −2) were removed; their weights are in each run's config.yaml.

## 3. PPO settings and why
Base (`configs/base.yaml`) unless a method overrides it (`configs/m1.yaml`, `m2.yaml`, `m3.yaml`).

| setting | value | why |
|---|---|---|
| γ | 0.99 per 0.1 s → M1 0.99, M2 0.951, M3 0.904 per decision | same ~10 s lookahead in seconds for every method |
| GAE λ | 0.995 per 0.1 s → M1 0.995, M2 0.975, M3 0.951 per decision | **new.** A fixed λ = 0.95 per *decision* gave M1 a ~2 s credit window and M3 ~20 s. Scaled like γ, all methods get the same ~7 s window (1/(1−γλ) in seconds). |
| rollout `n_steps` | M1 1280, M2 256, M3 128 per env | **new for M1.** = 128 s of driving per env per rollout for every method, so updates per simulated hour are equal (with 14 envs: one update per ~30 min of driving). Pilot: M1 70, M2 14, M3 7 updates in 2M steps. |
| `batch_size` / `n_epochs` / `target_kl` | M1 512 / 4 / 0.03; M2, M3 256 / 4 / 0.03 | **Update guard.** Each PPO update makes `n_epochs` passes over the last rollout; approx_kl measures how far that moves the policy's action distribution. Small moves keep the rollout data valid for the new policy; large ones step into untested territory, the policy gets worse, collects worse data, and spirals. `target_kl` stops an update's passes early once approx_kl exceeds it; fewer epochs and (M2) a smaller learning rate keep each step small. Healthy: approx_kl < ~0.02, clip fraction 0.1–0.2. Evidence: M1 pilot approx_kl 0.01 → 0.23 (guard added; round 3 stayed ~0.01); M2 round 3 approx_kl 0.006 → 0.55, clip fraction 0.66, eval success 1.0 → 0 after 1.2M steps (M2 also collapsed in round 2); M3 round 2 ~0.05 / 0.4–0.49 without collapsing. |
| `log_std_init` | −0.7 (initial std 0.5) | **new.** SB3's default std 1 on a [−1, 1] action box clips ~32 % of samples to the limits. Pilot: M3's std stayed 0.98 for the whole run (near-random rollouts: 50 % training collisions vs 0 % in deterministic eval); M2's first third was 100 % off-road within 8 s; M1 steered randomly at speed. Std 0.5 clips ~5 %. |
| `ent_coef` | 0 | exploration is set by `log_std_init`; an entropy bonus would push std back up |
| reward normalisation | on (`VecNormalize`, rewards only), clip ±100 | **new.** Returns span −500…+105; normalising by the running return std keeps the value loss from swamping the policy loss (pilot explained variance ≈ 0.5). Clip raised from SB3's 10 so the crash penalty is not clipped early in training. Logs and eval stay raw. |
| `learning_rate`, `clip_range` | 3e-4 (M2 1e-4), 0.2 | SB3 defaults; M2 lowered with the update guard |
| eval | 40 seeds (1000–1039), deterministic, every 200k control steps | 2.5 % resolution on rates (20 seeds = 5 %); best checkpoint = highest success, ties by return |
| episode cap | 150 s | longest route stuck behind the slowest lane (35 km/h) takes 118 s; 120 s left no margin |

## 4. Run robustness
* A crashed env or eval worker (e.g. a native segfault) ends the job: `FAILED.txt` (traceback),
  `checkpoints/crash.zip`, all child processes killed, non-zero exit.
* Watchdog: no progress for 15 min (`training.watchdog_minutes`; env start-up, PPO steps, each finished eval
  episode and the final shutdown all count) → kill workers, exit code 3 (hung worker). Env shutdown waits at most 2 min.
* `train.sbatch` sets `PYTHONUNBUFFERED=1` (live log) and `PYTHONFAULTHANDLER=1` (traceback on a segfault).
* `checkpoints/vecnormalize_*.pkl` is saved with every checkpoint (needed only to resume; resume is not implemented).

## 5. Pilot round 1 (2026-11-18/19, reward r0, 2M control steps, seed 0)
Run folders were in `11_19_data/` (removed 2026-11-21; in git history at commit `62bc7b3`).
| | M1 | M2 | M3 |
|---|---|---|---|
| final eval success / collision | 0 % / 85 % | 65 % / 35 % | 100 % / 0 % |
| mean speed | 13.5 m/s | 10.8 m/s | 10.0 m/s |
| PPO updates | ~70 | ~14 | ~7 |

What it showed (and what changed because of it):
* **Reward** made crashing late nearly as good as finishing → `r3_fast`.
* **Too few updates** for M2/M3 → equal 128 s rollouts. M2 only started working after ~1.2M steps and was still
  improving.
* **Exploration too wide** → `log_std_init −0.7`.
* **M3 never passed**: lane changes were exactly 1 on off-centre spawns (it moved to the middle lane and stayed),
  0 otherwise; its policy barely changed (std 0.98) and it followed 30 km/h traffic. M3 also could not choose its
  speed → v\* channel (target and ceiling).
* **M3 off-road in ~30 % of training episodes** → road corridor with fixed lateral faces.
* **Randomised ego dynamics**: the pilot trained on MetaDrive's `default` vehicle (engine/brake resampled every
  episode), the controllers were tuned on `static_default` → fixed.
* No episode crashed in its first 3 s (spawns are clean).

## 6. Known risks for round 2
* **M3 plans over the lateral-acceleration limit no longer end episodes** (2026-11-22; DESIGN §4): they are flown
  and counted (`infeasible_count`). The `infeasible_plan` termination column stays in `episodes.csv` (0 for new runs)
  so earlier runs, where it was a crash, still compare. `emergency_brake` only with the gate on.
* **M3 decides once per second**: a car cutting in mid-decision is the corridor/MPC's problem. The corridor ignores
  cars more than 2 m behind the ego (`corridor.rear_range`); keeping clear of faster traffic from behind is the RL's.
* **Spurious off-road**: MetaDrive's lane raycast can miss at road-piece seams (an `off_road` end near a lane
  centre).
* **No time-left feature** in the observation (value estimates slightly off near the 150 s cap).
* **20 → 40 eval seeds** doubles eval cost; M3 eval is the slowest.

## 7. Great Lakes workflow
One-time setup (login node, needs internet):
```bash
git clone https://github.com/BradyButen/learning-driving-control.git handoff && cd handoff
PY_MODULE=python/3.12.1 bash slurm/setup_greatlakes.sh            # CUDA PyTorch (gpu partitions)
PY_MODULE=python/3.12.1 CPU_ONLY=1 bash slurm/setup_greatlakes.sh # CPU-only PyTorch (standard partition)
```
Before launching, run the validation suite on a compute node, then read throughput off a short run: training prints
a status line every `training.progress_seconds` (60 s) with control steps/s.
```bash
srun --account=rob535_navarch565_f26_class --partition=standard --cpus-per-task=20 --time=01:00:00 --pty bash
source .venv/bin/activate && export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 && unset PYTHONPATH
python scripts/validate.py check
python scripts/train.py --method m3 --set training.n_envs=14 --set eval.n_workers=4 --set eval.freq_control_steps=0
```
Budget for a 4–5 h run ≈ control steps/s × 4.5 h × 3600 × 0.8 (eval + PPO updates).

Sizing from round 2 (14 training envs + 4 eval workers, 20 cores, 4 GB/core, 40 eval episodes every 200k steps;
control steps/s including eval): M1 ~175 on `gpu`, M2 ~195 and M3 ~93 on `standard`. All three methods get the
same budget (equal simulated experience, D-07); only the wall time differs. Round 2 lost all three runs: M1 and M3
hit the time limit, M2 was OOM-killed at 80 GB after 3.9 h. Cause (found round 3): MetaDrive's map cache
(`store_map`, default true) keeps every visited seed's map in each process, ~10 MB each, up to 1040 seeds: memory
grew ~10 MB per episode (a laptop warm-start trial reached 38 GB in 28 min). `sim.store_map: false` (base.yaml)
rebuilds the seed's map on every reset instead (same map, so replays and evals are unchanged). The progress line
prints `mem <GB>` (main + all workers): it should now level off; check `sacct -j <id> --format=MaxRSS` afterwards.

Jobs are capped at 8 h. `training.max_wall_minutes` stops the training loop in time for the final eval and
final.zip: 450 min for M1 / M2 (final eval ~10-15 min), 420 min for M3 (final eval up to ~40 min: 40 episodes of
up to 150 s on 4 workers), plus ~5 min start-up. Round 3 (2.2M each) finished M1 in 2.5 h (~240
steps/s) and M2 in 3.3 h, so the budgets now live in each method's config and use most of the 7.5 h; M2's is the
first one after the memory fix (it was 72 GB at 2.2M before), so watch its `mem` line. Compare methods at equal experience from the eval every 200k
steps and the checkpoint every 100k.

| | partition | budget (`configs/<m>.yaml`) | `max_wall_minutes` | expected training time | memory at end |
|---|---|---|---|---|---|
| M1 | `gpu` | 6.0M | 450 | ~7 h | ~115 GB |
| M2 | `standard` | 4.5M | 450 | ~6.8 h | flat (was 72 GB at 2.2M before the fix) |
| M3 | `standard` | 2.3M | 420 | ~7 h | |

M1 on `gpu`: its rollouts are 18k samples × 4 epochs through the CNN, which would eat much of its time on 2 CPU
threads; M2 / M3 updates are small (3.6k / 1.8k samples), so `standard` is fine. Memory: ask for 7 GB/core
(140 GB for 20 cores) until the `mem` line shows the real growth rate (check the node's total with
`scontrol show node <node>`).

Launch a batch (`r4_straight` is the default reward; every job gets `--time=08:00:00`). Per seed, each method
from scratch (`m1 m2 m3`) and warm-started from its behaviour-cloning clone (`m1_warm m2_warm m3_warm`); M1 on
`gpu`, M2 / M3 on `standard`. The class account runs 2 jobs at a time (`AssocMaxJobsLimit`), in submission order:
```bash
DRY=1 bash slurm/launch_batch.sh                            # print the sbatch commands (default: all 6)
JOBS="m1_warm m2_warm m3_warm" bash slurm/launch_batch.sh   # warm starts only
SEEDS="0 1 2" JOBS="m3_warm m3" bash slurm/launch_batch.sh
```
Warm start (docs/human-data-pipeline.md): `training.init_policy=data/bc/<m>_clone.zip`, `init_log_std=-2.0`,
`critic_warmup_control_steps=100000`, output under `runs/warm/`, Slurm log `slurm-handoff-warm-<m>-<id>.out`
(from scratch: `slurm-handoff-<m>-<id>.out`).
`train.sbatch`: 20 cores → 14 training envs + 4 eval workers + main; `OMP_NUM_THREADS=1`. Each run writes
`runs/<method>/r4_straight/seed_<n>/<timestamp>/`. Check `sacct -j <id> --format=MaxRSS` and lower `--mem-per-cpu`
if far below 80 GB.

Analysis:
```bash
python scripts/plot.py curves runs/m*/r4_straight/seed_*/* --metric success --x control_steps --out curves.png
python scripts/evaluate.py --checkpoint <run>/checkpoints/best.zip --n-workers 4
python scripts/view.py live --checkpoint <run>/checkpoints/best.zip --seeds 1017   # needs a display
python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip
```
Always `unset PYTHONPATH` (ROS) and `export SDL_VIDEODRIVER=dummy` on headless nodes.

## 8. Warm-start batch (2026-11-21/22, old r4 weights) and what changed for the next one

Jobs 66584139–41 (`m1_warm m2_warm m3_warm`, seed 0), runs in `runs/warm/<m>/r4_straight/seed_0/`. 40 eval seeds,
deterministic. Reward then: time 0.1, speed_gap 2, success +50, crash −2000.

| | step 0 (= the clone) | best eval round | final | from scratch, round 3 (final) |
|---|---|---|---|---|
| M1 | 0 %, −2012, 60 km/h | 25 %, −1483 (3.0M) | 12.5 %, 75 km/h, 75–85 % crash | 0 % (one 48 % spike at 0.8M) |
| M2 | 22 %, 47 km/h | 100 %, −174 (3.4M) | 100 %, 45 km/h | 0 % (collapsed) |
| M3 | 100 %, 44 km/h | 100 %, −172 (1.6M) | 100 %, 45 km/h | ~90 %, 25 km/h (old MPC) |

Mechanics all worked: clone loaded (step-0 eval = clone), actor unfrozen at ~107k steps with no drop, approx_kl
~0.01 (M2) / ~0.03 (M1, M3), memory flat at 16–19 GB (the `store_map` fix held). M1 hit the 450-min wall at 3.58M of
6M steps (133 steps/s); M2 (279 min) and M3 (231 min) finished their step budgets with hours to spare.

Findings:
* **M2 and M3 learned to follow traffic, never pass**: ~44 km/h, 0.75 lane changes per episode (the spawn merge)
  in every eval round, speed term ≈ −335 of the ≈ −175 return. Passing was worth ~+340 and risked −2000.
* **M3's only training failures were `infeasible_plan`** (101 of 2817 episodes, none in eval): exploration noise
  (0.8 m per waypoint) pushed lane changes ~10 % over the lateral-acceleration limit, each charged as a crash. Sideways
  exploration was what got punished (`docs/m3-mpc-lessons.md` §8).
* **M1 never brakes**: the clone has no braking-for-traffic data and the low noise (std 0.13) does not discover it.
  The reward is not its bottleneck; corrective drives (DAgger) are.

Changes for the next batch (2026-11-22):
1. M3 infeasible plans are flown and counted (`infeasible_count`), no longer a termination (DESIGN §4).
2. Reward weights for speed (§2): time 0.3, speed_gap 6, success +500, crash −4000. Passing is worth ~+1180 and pays
   while its crash risk is below ~26 % (was 16 %; −2000 with the new speed terms would have allowed ~46 %).
3. The clones are unchanged: behaviour cloning never sees the reward or the termination rule; the M3 clone never
   asked for an infeasible plan in its (100 %) selection drives. Only the PPO stage reruns.
4. From-scratch baselines rerun too (`JOBS="m3_warm m2_warm m1_warm m3 m2 m1"`): returns from different weight
   sets do not compare; compare success, speed and crash rate across batches.
