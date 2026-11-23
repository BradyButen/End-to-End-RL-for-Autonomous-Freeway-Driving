# Training

RL setup, why each setting is what it is, what the batches taught us, and how to run on Great Lakes.
System design: [DESIGN.md](DESIGN.md).

## 1. Algorithm
PPO (Stable-Baselines3, `MultiInputPolicy`: the `ego` and `lanes` observation vectors concatenated into separate
256-256 policy and value MLPs), identical for M1/M2/M3. M1's action is continuous (Gaussian policy); M2's and M3's
are discrete (categorical: lane command and a relative speed command). Budgets and cadences are counted in **control steps** (0.1 s of simulated driving),
summed over parallel envs, so every method gets the same simulated experience (D-07).

## 2. Reward: `r5` (the only reward)
Goal: reach the finish line as fast as possible, like an aggressive driver but not a stupid one (no line riding, no
rapid oscillation, no bumper-touching). Per control step, summed within a decision; terminal terms added on the last
step. Raw terms: `src/handoff/reward.py`; weights and the arithmetic: `configs/r5.yaml`. Prices per second:

| term | price | why |
|---|---|---|
| progress | +0.0225 per m (+0.5 /s at 80 km/h) | a little for every metre down the road |
| speed | +4 /s × (v / 80 km/h)² | the main reward: driving fast |
| speed_gap | −42 /s × ((v − 80 km/h)/80 km/h)² | sets the break-even at 60 km/h: anything slower costs, steepest at a standstill |
| success | +150 | you still have to make it |
| collision / off_road | −1200 | also an M3 gate stop (gate off). Must stay worse than the worst finish (stuck behind the 35 km/h lane ≈ −1120), or crashing on purpose pays whenever the car is slow |
| lane_keep | −1 /s × (offset from the nearest lane centre / 1.75 m)² | smooth: centred is free, a lane line costs 1 /s, a lane change ≈ 0.7 |
| heading | −9 /s per rad of heading error beyond 0.5° | slow weaving (±0.5 m in lane for 100 s ≈ −40) |
| steer_rate | −0.6 /s per m/s² of change, per step, of the lateral acceleration the steering asks for (v² tan δ / L) | rapid oscillation / jitter, speed-aware; a smooth lane change ≈ −0.7. In training it also charges the exploration noise, which deterministic evals do not pay |
| headway | −5 /s × (1 − headway / 0.5 s) for a car overlapping sideways ahead (gap / ego speed) or behind (gap / its speed: a cut-in) | tailgating closer than 0.5 s (11 m at 80 km/h) |
| time, lane_change | 0 (logged only) | `lane_change` feeds the `lane_changes` metric |

Net driving reward per second: +4.5 at 80 km/h, +2.8 at 70, **0 at 60 km/h**, −6.5 at 45, −12.3 at 35, −42 stopped.
Following 35–45 km/h traffic costs every second (the push to pass); the crash penalty is set so that crashing is
still worse than any finish, so quitting never pays. Episode scores: finish at 80 km/h ≈ +350 > 70 ≈ +295 > 60 ≈
+150 > behind 45 km/h traffic ≈ −370 > behind the 35 km/h lane ≈ −1120 > crash ≈ −1200 > crawl to the timeout
≈ −5700. Passing instead of following the 45 km/h lane (+720) pays while it adds less than ~45 % crash risk.

History (2026-11-23, laptop M1 trials): r4_straight (batches 1–6a; in git history and each old run's `config.yaml`:
time cost, quadratic speed penalty, −4000 crash, an on / off lane term beyond 0.57 m, 5 m proximity) flipped between
crawling and crashing with a ~10 s discount, and its lane term gave no signal toward the centre. A positive
(v / v_max)² alone is flat near 0 and crawled at 6 km/h. A time cost + tripled speed penalty with a −100 crash made
every second alive a loss once the policy could not finish, and it learned to drive off the road within ~5 s
(`rl-lessons.md` §12).
Returns from r4 and r5 do not compare; compare success, speed and crash rate.

## 3. PPO settings and why
Base (`configs/base.yaml`) unless a method overrides it (`configs/m1.yaml`, `m2.yaml`, `m3.yaml`).

| setting | value | why |
|---|---|---|
| γ | 0.999 per 0.1 s → M1 0.999, M2 0.995, M3 0.990 per decision | same ~100 s lookahead in seconds for every method: most of an episode, so the +150 finish is felt from the start (0.999^450 ≈ 0.64). Was 0.99 (~10 s): crawling then looked cheaper than a crash risk within the horizon (`rl-lessons.md` §11). The ego feature `t_left` (time left / 150 s) lets the critic see how much episode is left, which a horizon this long makes matter |
| GAE λ | 0.995 per 0.1 s → M1 0.995, M2 0.975, M3 0.951 per decision | **new.** A fixed λ = 0.95 per *decision* gave M1 a ~2 s credit window and M3 ~20 s. Scaled like γ, all methods get the same ~7 s window (1/(1−γλ) in seconds). |
| rollout `n_steps` | M1 1280, M2 256, M3 128 per env | **new for M1.** = 128 s of driving per env per rollout for every method, so updates per simulated hour are equal (with 14 envs: one update per ~30 min of driving). Pilot: M1 70, M2 14, M3 7 updates in 2M steps. |
| `batch_size` / `n_epochs` / `target_kl` | M1 512 / 4 / 0.03; M2, M3 256 / 4 / 0.03 | **Update guard.** Each PPO update makes `n_epochs` passes over the last rollout; approx_kl measures how far that moves the policy's action distribution. Small moves keep the rollout data valid for the new policy; large ones step into untested territory, the policy gets worse, collects worse data, and spirals. `target_kl` stops an update's passes early once approx_kl exceeds it; fewer epochs and (M2) a smaller learning rate keep each step small. Healthy: approx_kl < ~0.02, clip fraction 0.1–0.2. Evidence: M1 pilot approx_kl 0.01 → 0.23 (guard added; round 3 stayed ~0.01); M2 round 3 approx_kl 0.006 → 0.55, clip fraction 0.66, eval success 1.0 → 0 after 1.2M steps (M2 also collapsed in round 2); M3 round 2 ~0.05 / 0.4–0.49 without collapsing. |
| network | `net_arch` [256, 256], separate actor / critic | batches 1–5 shared a NatureCNN on the BEV between actor and critic; M1's critic never learned (explained variance ~0 for 6M steps) |
| M1 exploration | `init_log_std` [−1.9, −0.7] → std [0.15, 0.5] = [1.2 m/s² lateral, ~1.5 m/s² accel] | per dimension, in physical units (`rl-lessons.md` §13). Warm M1 at one std 0.135 never braked (0.4 % of steps) and its steering noise asked for ~19 m/s² at 80 km/h. SB3's default std 1 clips ~32 % of samples. |
| M2 / M3 exploration | `init_default_prob` KEEP 0.95 (M2) / 0.9 (M3), HOLD speed 0.8 | categorical policies start uniform; that would change lanes on 2 of 3 decisions and move the target speed every decision. The bias starts them at "keep lane, hold speed" with occasional alternatives (faster and slower equally likely, so no drift) |
| `ent_coef` | M1 0; M2, M3 0.01 | M1: exploration is set by its std; an entropy bonus would push it back up. Discrete: a small bonus keeps lane changes in play (warm batches collapsed to a constant plan) |
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

## 5. Results so far (what to build on)
Returns from different reward weights do not compare; compare success, speed and crash rate. Run folders: `runs/`.

| batch | reward | M1 | M2 | M3 |
|---|---|---|---|---|
| pilot round 1 (11-18/19, 2M) | r0 (MetaDrive default) | 0 %, 85 % crash | 65 % | 100 %, ~36 km/h, never passed |
| round 3, scratch (2.2M) | r4, first weights | best 48 % at 0.8M, then every eval a crash | collapsed (best = untrained step 0) | ~90 %, 25 km/h (old MPC) |
| warm batch 1 (11-21/22) | r4, first weights | 25 % best, 75 % at ~75 km/h crash, never brakes | 22 → 100 %, ~45 km/h | 100 %, ~45 km/h |
| batch 5, scratch + warm (11-22/23) | r4, current weights | still crashes | ~100 %, slow | ~100 %, slow |

What changed because of them: pilot → r0 replaced (a late crash scored nearly as well as finishing), equal 128 s
rollouts, `log_std_init` −0.7, M3 road corridor + v\* channel, fixed ego dynamics (`static_default`); round 3 →
update guard (`target_kl`), `store_map: false` (memory); warm batch 1 → M3 infeasible plans no longer end episodes,
reward weights pushed for speed.

Open problems for the next attempt:
* **M2 / M3 follow traffic and never pass.** Every batch: ~45 km/h, ~1 lane change per episode (the spawn merge).
  Low-noise warm starts converge to the nearest safe behaviour; heavier speed penalties (batch 5) did not change it.
  The M3 waypoint action is 12 numbers the corridor / MPC mostly override, so the reward barely shapes them.
* **M1 crashes / never brakes.** The clone has no braking data; from scratch, its exploration noise at speed throws
  the car off the road, and the noisy policy learns to crawl (`docs/rl-lessons.md` §11–13).
* **The crawl escape.** With r4 (negative per-step reward) and a ~10 s discount, crawling beats risking a crash
  (`docs/rl-lessons.md` §11–12).

### Batch 6 changes (2026-11-23): fixes 1–3, bundled, then reward r5 + γ 0.998
1. **M2 / M3 decide lanes and a speed** (DESIGN §4): the same discrete action, [keep / left / right,
   hold / +10 / −10 km/h on a sticky target speed]; a lane change runs to completion. M2's PID tracks the speed; for M3 it is the MPC's ceiling and the
   MPC owns the trajectory and braking. Exploring a pass in M3 no longer risks a crash; the default
   action is "keep lane, 80 km/h", not 40 / 50 km/h. When fed human lane plans the controllers already drove
   70–77 km/h, so the bottleneck was the decisions.
2. **Ground-truth lane observation** out to 100 m (DESIGN §3), separate actor / critic MLPs.
3. **M1 in physical units**: steering = target lateral acceleration (±8 m/s²), per-dimension noise (braking gets
   sampled). The old M1 clone is incompatible (observation and steering label changed): rebuild it
   (`CLONE_ONLY=1 scripts/laptop_warm_trial.sh`). M2 / M3 warm starts are dropped (their labels were for the old
   actions, and the low-noise warm starts pinned them to following).
4. **Reward r5, γ 0.998, `t_left`, smooth lane term** (§2, §3), after two laptop M1 trials under r4 crawled and
   crashed in turn and never centred in a lane.

What to watch: lane changes per episode in eval (> 1 means passing), mean speed, and for M2 collisions (it owns
braking). Old runs (batches 1–5) no longer load with this code: compare them through their `episodes.csv`.

Known issues: MetaDrive's lane raycast can report a spurious `off_road` at road-piece seams; M3 decides once per
second (a mid-decision cut-in is the corridor / MPC's problem; cars > 2 m behind are ignored by the corridor); no
time-left feature in the observation.

## 6. Great Lakes workflow
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
control steps/s including eval, with the BEV observation): M1 ~175 on `gpu`, M2 ~195 and M3 ~93 on `standard`.
Without the BEV rendering (batch 6) every method should be faster; not measured yet. All three methods get the
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

| | partition | budget (`configs/<m>.yaml`) | `max_wall_minutes` |
|---|---|---|---|
| M1 | `standard` | 10M | 450 |
| M2 | `standard` | 8M | 450 |
| M3 | `standard` | 4M | 420 |

Budgets are set above what 7.5 h should reach, so the wall clock is the stop; compare methods at equal control
steps from the evals. All on `standard`: the policies are small MLPs (no CNN), so the GPU buys nothing. Memory: ask for 7 GB/core
(140 GB for 20 cores) until the `mem` line shows the real growth rate (check the node's total with
`scontrol show node <node>`).

Launch a batch (`r5` is the default reward; every job gets `--time=08:00:00`). Per seed, each method
from scratch (`m1 m2 m3`) and M1 warm-started from its behaviour-cloning clone (`m1_warm`). The class account runs
2 jobs at a time (`AssocMaxJobsLimit`), in submission order (default `m3 m2 m1_warm m1`):
```bash
DRY=1 bash slurm/launch_batch.sh                            # print the sbatch commands (default: all 4)
SEEDS="0 1 2" JOBS="m3 m2" bash slurm/launch_batch.sh
```
Warm start (docs/warm-start.md): `training.init_policy=data/bc/m1_clone.zip`, `critic_warmup_control_steps=100000`
(noise from `configs/m1.yaml`), output under `runs/warm/`, Slurm log `slurm-handoff-warm-m1-<id>.out` (from scratch:
`slurm-handoff-<m>-<id>.out`).
`train.sbatch`: 20 cores → 14 training envs + 4 eval workers + main; `OMP_NUM_THREADS=1`. Each run writes
`runs/<method>/r5/seed_<n>/<timestamp>/`. Check `sacct -j <id> --format=MaxRSS` and lower `--mem-per-cpu`
if far below 80 GB.

Analysis:
```bash
python scripts/plot.py curves runs/m*/r5/seed_*/* --metric success --x control_steps --out curves.png
python scripts/evaluate.py --checkpoint <run>/checkpoints/best.zip --n-workers 4
python scripts/view.py live --checkpoint <run>/checkpoints/best.zip --seeds 1017   # needs a display
python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip
```
Always `unset PYTHONPATH` (ROS) and `export SDL_VIDEODRIVER=dummy` on headless nodes.
