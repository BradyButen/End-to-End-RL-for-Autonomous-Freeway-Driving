# Human driving data → imitation learning (M1)

**Goal:** check whether human demonstrations help the RL. We record people driving the freeway, train a copy of
PPO's policy network to imitate them (behaviour cloning, BC), and later start PPO from that copy instead of from
random weights. The comparison for the report is **M1 PPO from scratch vs. M1 PPO warm-started from BC**, measured by
learning curves and final eval metrics.

Branch: `human-demos` (from Brady's `New_RL_Formulations`, with `main` merged in). Owner: Kevin.

## Status (2026-11-21)

| step | what | status |
|---|---|---|
| 0 | record human drives (`scripts/drive.py`) | ✅ 30 episodes driven, 17 kept |
| 1 | turn the logs into a training dataset (`scripts/build_demos.py`) | ✅ 8,220 samples, checked |
| 2 | behaviour cloning (`scripts/bc.py`) | ✅ code works; first clone **does not drive yet** (0/40) |
| 2b | make the clone drive: regularisation | ✅ tried, **does not help** (data problem, see below) |
| 2c | make the clone drive: corrective data (DAgger / targeted drives) | ⏳ next, or skip to step 3 |
| 3 | PPO warm start option in `scripts/train.py` | ✅ done, tested locally (off by default) |
| 3b | 1 h laptop pilot of the warm start (`scripts/laptop_warm_trial.sh`) | ✅ first positive M1 returns, see below |
| 4 | from-scratch vs warm-start runs on Great Lakes (`slurm/launch_batch.sh`) | ⏳ launching |
| 5 | M2 / M3 clones: the same drives relabelled in their actions (`build_demos.py --method m2/m3`) | ✅ code ready, offline-checked; clones to build (`METHOD=m3 CLONE_ONLY=1 scripts/laptop_warm_trial.sh`) |

**M1's action changed (2026-11-21)** from raw `throttle_brake` to a target acceleration (`method.accel_range`
[−4, 2.9] m/s², 0 = hold speed; `src/handoff/m1.py`): raw throttle_brake is ~5× stronger on the brake side, so
PPO's zero-mean noise netted out to hard braking. `build_demos.py` now replays the logs with raw commands
(`method.accel_range=null`, so they reproduce the drive) and converts the labels through the actuation table
(`action`; the logged command is kept as `action_raw`). Throttle labels change by ≤ 0.003 (+1 = full throttle in
both), only the 1.2 % braking samples differ, so `data/bc/m1_clone.zip` (trained on raw labels) is used as is;
rebuild + retrain (`REBUILD=1 scripts/laptop_warm_trial.sh`) for a clone on exactly matching labels.

**Laptop pilot** (`runs/warm_trial/m1/r4_straight/seed_0/20261121-165111`; 12 envs, 60k-step critic warm-up, std
0.135; ~500k control steps before an OOM, since fixed with `sim.store_map: false`): training distance per episode
106 → 324 m, off-road 66 → ~18 %, never fell back to crawling, first successes after ~300k steps (10 in training,
~77 km/h, return up to +79; from-scratch M1 never had a positive return). Eval at 400k: 5 % success (seed 1000,
+53.4). Collisions stay ~80 %: it passes traffic but does not brake when a pass fails.

## The pipeline

```
 you drive            curate               replay in sim              fit PPO's network          drive the clone
 scripts/drive.py ──▶ data/human_demos/ ──▶ scripts/build_demos.py ──▶ scripts/bc.py ──────────▶ scripts/evaluate.py
 runs/drive/<batch>/  (git-tracked .npz)    runs/demos/m1.npz         runs/bc/m1/<ts>/            <run>/final_eval/
                                                                       checkpoints/best.zip   ──▶ (step 3) PPO warm start
```

### Commands

Every new terminal (conda's Python otherwise shadows the project's `.venv`, and MetaDrive is only in `.venv`):
```bash
conda deactivate; source .venv/bin/activate; unset PYTHONPATH
which python          # must be .../RL-MPC-DrivingControl/.venv/bin/python; if not, call .venv/bin/python directly
```

```bash
# 0. drive (needs a display; one episode per seed, back to back). Use TRAINING seeds 0-999 only.
python scripts/drive.py --seeds 11 87 142 203 266 --out runs/drive/<batch_name>     # --controller xbox for a gamepad

# 0b. copy the good episodes (see "Curation") into the tracked folder
cp runs/drive/<batch_name>/episodes/drive_eval_r0_e*_s<seed>.npz data/human_demos/

# 1. logs -> dataset (~100 s for 17 episodes)
python scripts/build_demos.py                          # data/human_demos/*.npz -> runs/demos/m1.npz

# 2. behaviour cloning (~2 s per epoch on a laptop CPU)
python scripts/bc.py                                   # -> runs/bc/m1/<timestamp>/checkpoints/best.zip

# 2. drive the clone on the 40 evaluation seeds (never seen in the demos)
python scripts/evaluate.py --checkpoint runs/bc/m1/<timestamp>/checkpoints/best.zip --n-workers 6
python scripts/view.py live --checkpoint runs/bc/m1/<timestamp> --seeds 1003 1027     # watch it (display)
```

## Step 0 — recording (`scripts/drive.py`, Brady's)

The same env the RL trains in (map, traffic, termination, reward, logging); only the controls come from a keyboard /
gamepad through MetaDrive's `ManualControlPolicy`. Each episode is ~1 km of freeway, ~45 s at 80 km/h, 150 s cap.

* Keys: W throttle, S brake, A / D steer. **X ends the episode and saves it.** Esc quits *without* saving.
  Avoid **T** (MetaDrive's autopilot takeover: its driving would be logged as yours) and **R** (reset).
* Output: `runs/drive/<batch>/episodes/<id>.npz` (full per-step log, ~0.1 MB) + `<id>_reward.png`.
* **Seeds 0–999 only.** Seeds 1000+ are the fixed evaluation seeds; demos on them would leak the test maps.
* `runs/` is git-ignored. Keepers are copied to `data/human_demos/`, which `.gitignore` explicitly re-includes
  (`!data/human_demos/*.npz`).

### Curation (what we keep)

Kept: `success`, no contact, no sustained tailgating. Dropped: every collision / off-road, plus successes that
touched a car (min gap 0.0 m) or rode bumpers (proximity return ≲ −70) — the clone copies everything, including
tailgating.

| batch | seeds driven | kept | dropped |
|---|---|---|---|
| `demos_clean1` | 11 87 142 203 266 318 391 455 512 579 | 11 87 318 391 512 579 | 142 203 (collision), 266 455 (off road) |
| `demos_clean2` | 604 648 701 733 789 822 856 903 947 988 | 822 856 903 | 604 789 988 (collision), 648 733 (off road), 701 (touched a car), 947 (tailgating) |
| `demos_recovery` | 25 61 174 239 347 428 563 677 765 871 | 25 61 239 428 563 677 765 871 | 174 (off road), 347 (tailgating) |

17 episodes, ~8,200 control steps (~14 min of driving), mean speed ~73–78 km/h, 2–10 lane changes each. Despite the
name, the "recovery" batch has few real recoveries (its heading / lane-keep penalties look like the clean runs).
All driven on a keyboard.

## Step 1 — dataset (`scripts/build_demos.py`)

The logs store the ego / car states and the human's controls, **not** the observation the policy sees (the 5×84×84
top-down BEV + 5-number ego vector). Redrawing the BEV from logged states would be fragile (MetaDrive's renderer keeps
its own history of past positions and traffic frames), so each episode is **replayed**: the M1 env is reset on the
same seed and fed the logged `[steering, throttle_brake]` step by step, headless, recording the real observation
before every action.

Checks (all passed):
* **Replay = original drive.** MetaDrive is deterministic: max ego position error 0.0 m with a fresh env, ≤ 6 mm
  when one env is reused across episodes (a BEV pixel is 100 m / 84 ≈ 1.2 m). Every replay ends at the same step
  with the same termination. Tolerance in the script: 5 cm.
* **No off-by-one.** Log row *t* is the state after control step *t*, and its steering / throttle are the action
  applied at step *t*. Sample *t* = (observation before step *t*, action at step *t*). Verified: the ego vector at
  *t+1* contains exactly the action label at *t* (difference 0.0 on all 8,220 samples).
* **Images look right.** Sample frames (cruise, hardest left, hardest right, braking) show road, past positions and
  three traffic frames in the expected places.

Output `runs/demos/m1.npz` (6 MB, git-ignored, rebuild any time): `bev` uint8 (×255), `ego`, `action`, `episode`,
`step`, `files`, `seeds`.

Label statistics: steering is 0 on 88 % of steps (keyboard: either nothing or a few fixed values), |steering| ≤ 0.5;
throttle is full on 80–97 % of steps; **braking on ~1 %**.

## Step 2 — behaviour cloning (`scripts/bc.py`)

* **Same network as PPO.** Builds the exact model `scripts/train.py` builds (SB3 `PPO("MultiInputPolicy", ...)` with
  the M1 `ppo` kwargs: NatureCNN on the BEV + ego vector → 64×64 MLP → action mean), without starting MetaDrive.
  Saved as a normal SB3 PPO `.zip` in a normal run folder, so `evaluate.py` / `view.py` work on it unchanged and
  step 3 can load it straight into PPO.
* **Trained:** CNN + policy MLP + action head, weighted MSE on the action mean. **Not trained:** the value network
  and `log_std` (still 0.5 std). Step 3 has to handle both.
* **Copycat guard.** The ego vector includes the *previous* step's steering / throttle, which almost equals the
  label (consecutive human actions differ by ~0.02). A net with those inputs learns "repeat the last action" and
  ignores the road — it looks great on validation loss and fails when it drives. They are zeroed in training and
  their first-layer weights are held at 0, so the clone ignores them (verified: randomising them changes the output
  by exactly 0). PPO can still learn to use them after the warm start. `--keep-prev-action` turns this off.
* **Imbalance.** Samples that steer (12 %) are up-weighted ×7.2, braking samples (0.9 %) ×10 (inverse frequency,
  capped). Each action dimension's error is divided by its label variance.
* **Validation** on 3 whole held-out episodes (seeds 391, 856, 903).
* **Checkpoint selection by driving** (`--select drive`, default). Validation loss turned out not to predict
  driving, so after every epoch the clone drives 10 training-range seeds that are not in the demos (100–109) and the
  epoch with the furthest mean distance is kept. The 40 evaluation seeds stay untouched. Adds ~5 s per epoch.
  `--select val` = lowest validation loss instead. Early stop after `--patience` epochs without improvement.
* **Weight decay** (`--weight-decay`, AdamW) and the steer / brake weight cap (`--max-weight`) are options.

### First result: `runs/bc/m1/20261121-144523` — the clone does not drive yet

Offline: val loss 6.76 at epoch 4 vs 8.89 for "zero steering, mean throttle" (24 % better); steering sign correct on
78 % of held-out steering samples. **Heavy overfitting**: train loss falls to 0.36 while val loss rises after epoch
4 (6,700 samples vs ~1.7 M network parameters).

Closed loop, 40 evaluation seeds: **0 % success, 77.5 % collision, 22.5 % off road**, mean 227 m of ~1 km
(best 667 m, seed 1017, 10 lane changes). Two failure modes:
1. **Off road in 1–3 s (9/40).** On seeds where the ego starts in an edge lane, it drifts over the solid edge line
   (touching it ends the episode). The clone has not learned to hold its lane from the BEV.
2. **Rear-ends traffic after 6–30 s (31/40).** Full throttle, almost never brakes, changes lane too rarely / late.
   Braking is 1 % of the data; the human avoided slow cars mostly by changing lanes early.

This is a common first BC result with ~15 min of keyboard data; the pipeline itself is verified end to end.

### Regularisation sweep — does not fix it

Each run: 20 epochs, checkpoint picked by driving seeds 100–109 (mean distance, ~1 km = finished).

| run (`runs/bc/m1/...`) | settings | best drive (mean m) | success |
|---|---|---|---|
| `sweep_base` | defaults | 276 (epoch 20) | 0 % |
| `sweep_wd` | weight decay 0.05 | 229 (epoch 6) | 0 % |
| `sweep_wd_w3` | weight decay 0.05, steer / brake weight cap 3 | 277 (epoch 17) | 0 % |

Driving distance stays at ~150–280 m in every epoch of every run, whether the net is barely trained (epoch 1) or
heavily overfit (epoch 20): offline loss and driving are essentially unrelated here. (A 4th run, lr 1e-4, was killed
for memory: three parallel runs × 5 simulator workers is too much for the laptop. Run sweeps one at a time.)

### Diagnosis — a data-coverage problem, not a training problem

Checked on `sweep_wd_w3`:
* **No pipeline bug.** On the recorded frames the clone reproduces the human's actions closely (seed 11, first
  2.5 s: steering within ~0.02, throttle ramp matched).
* **It fails even on the maps it was trained on** (12 demo seeds: 0 % success, 4 off road within 2–3 s). Seed 11:
  the human starts in the right edge lane and does not touch the controls for ~1 s (steering exactly 0). The clone
  outputs a small "average" steering of −0.01 to −0.04 (a keyboard's exact zeros, blurred by regression), drifts
  0.6 m right in 2 s, and touches the edge line, which ends the episode. Forcing |steering| < 0.05 to 0 delays the
  drift but does not remove it (22 seeds: 0 % success with and without; mean 209 vs 210 m): once slightly off the
  human's path, the clone has never seen a correction. Keyboard driving with exact-zero steering never drifts, so
  the demos contain no "drifting → steer back" examples.
* **Collisions (31/40 eval seeds):** 90 % hit a car **ahead**, at about one car length, often clipping its corner
  (1–2 m lateral overlap), mostly at ~80 km/h behind 35–40 km/h traffic; **0 % braked in the last 2 s**. The clone
  never learned "slow car ahead → change lane / brake" (the human made ~120 passes in total, too few to learn that
  from pixels; braking is 1 % of samples).

This is the textbook behaviour-cloning failure (compounding error / covariate shift): small mistakes take the car
to states the human never visited, and the clone has no data there. Regularisation and loss tweaks cannot add that
data.

## Next steps (proposed)

1. **DAgger-style corrections (best fix).** The clone drives; you take over (hold a key / button) when it starts to
   go wrong; the states it reached are labelled with your corrections and added to the dataset; retrain; repeat 2–3
   rounds. Directly targets both failure modes. Needs a `drive.py` variant where the clone drives and the human can
   override (~1 evening of work), then ~20–30 min of driving per round.
2. **Targeted plain drives (cheaper, weaker).** Deliberate drift-and-correct in the edge lanes (let the car wander
   toward the edge line, then steer back), and many early passes / brakes behind slow cars. A gamepad gives
   proportional steering (no exact zeros, more corrections). Target 25–35 kept episodes.
3. **Go to the warm start anyway (step 3, now built).** The research question is whether demos speed up PPO, not whether BC alone
   drives. The clone already holds throttle, follows the road for ~200 m and changes lanes; PPO may fix the rest
   quickly. Load `best.zip` into `train.py`; pre-train the value net on the demo returns (or briefly train
   critic-only) so the first PPO updates do not wreck the actor; lower `log_std` (the human's steering std is ~0.06,
   PPO's initial 0.5 would swamp it). Same reward as the from-scratch runs. Touches Brady's `train.py`: coordinate.
   Can run in parallel with 1 or 2, and a better clone can be dropped in later.

## Step 3 — PPO warm start (`src/handoff/train.py`, `configs/base.yaml`)

Three `training.*` settings, all off by default (a run without them is unchanged):

| setting | what it does | suggested |
|---|---|---|
| `init_policy` | load the clone's **actor** (CNN, policy MLP, action head) into PPO before training; the critic and `log_std` stay freshly initialised. Errors if the network shapes do not match. | `data/bc/m1_clone.zip` |
| `init_log_std` | initial log action std. PPO's default −0.7 (std 0.5) would bury the clone's precise actions (human steering std ~0.06) in noise. | −2.0 (std 0.135) |
| `critic_warmup_control_steps` | train **only the critic** (actor + CNN frozen) for this many control steps first, so the first real PPO updates use a sensible advantage estimate instead of a random critic's, and do not wreck the actor. | 100,000 (~5–6 PPO updates on Great Lakes) |

A warm-started run writes `warm_start.txt` in its run folder and logs `train/actor_frozen` in `train_progress.csv`.
Its step-0 evaluation is the clone itself, so the learning curve starts from the clone's performance.

The clone used: `data/bc/m1_clone.zip` (git-tracked, 4 MB) = `runs/bc/m1/sweep_wd_w3` (weight decay 0.05, steer /
brake weight cap 3, epoch 17; settings in `data/bc/m1_clone_summary.json`).

Local tests (tiny runs, 3,000 control steps): with the warm-up longer than the run, the final actor is bit-for-bit
the clone (max change 0.0) while the critic trained; with a 1,500-step warm-up the actor is frozen for the first two
updates, unfrozen at 1,536 steps, and changes after; a run without `init_policy` is unchanged.

## Step 5 — converting human drives into M2 / M3 actions

The human drove with steering and throttle. M2 and M3 never output those: M2 picks a lateral target and speed
(d*, v*) every 0.5 s and its quintic + cascaded PID fly it; M3 picks up to 5 waypoints and a speed v* every 1 s and
its spline + corridor + MPC fly it. So the drives are relabelled in **hindsight**: at each decision, the label is
what the human's next few seconds of driving show they were going for, written in the method's own action.

**1. Observations — reused, not re-rendered.** Every method sees the same observation (5×84×84 BEV + 5-number ego
vector) and MetaDrive builds the BEV every 0.1 s control step regardless of the method. The replay of step 1 (logged
`[steering, throttle_brake]` fed back on the same seed, checked to ≤ 5 cm) therefore already produces the exact
observation M2 / M3 would see at each control step; the dataset keeps the ones at decision times (every 5th control
step for M2, every 10th for M3). The replay also records the ego state (s, d, v) before every step: the human's
trajectory in road coordinates, from which the labels are read.

**2. M2 labels — the reference trajectory the human travelled.** At decision t the label is the human's lateral
offset and speed 1 s later: d* = d(t + 1 s), v* = v(t + 1 s) (clipped to 0–80 km/h; `--m2-ahead-s`), with M2's
quintic horizon `trajectory.horizon` = 1.5 s; the PID (unchanged gains) tracks it. Why these: M2 replans every
0.5 s and flies only the start of each quintic, so the path it flies is a chain of quintic starts. Chaining them
offline against the 17 human paths (perfect tracking; the PID tracks its reference to ~0.03 m rms in the sim):

| lookahead L | horizon T | rms error | max | timing vs human |
|---|---|---|---|---|
| 2.0 s | 2.0 s (before) | 0.60 m | 3.0 m | 0.7 s early |
| 1.0 s | 2.0 s | 0.31 m | 1.6 m | 0.3 s late |
| **1.0 s** | **1.5 s** | **0.09 m** | **0.49 m** | **0.0 s** |
| 0.75 s | 1.0 s | 0.04 m | 0.25 m | 0.0 s |

The flown path matches the human's when L ≈ 0.7 T. T 1.0 fits slightly better but lets the RL command a lane change
at ~20 m/s² lateral; 1.5 s peaks at ~9 m/s² (M3's limit is 9.3). With 2 s / 2 s M2 changed lanes ~1 s before the
human and sideswiped the cars the human had waited for (sim: 1 of 8 drives succeeded). Encoded with `M2.encode`
(d* over [right edge − 1 m, left edge + 1 m], v* over [0, v_cap]). 17 drives → 1,652 samples, mean v* 75 km/h.

**3. M3 labels — the human path, sampled into waypoints.** At decision t:
* spacing ds = human speed × 2 s, clipped to `ds_range` [5, 40] m (≈ 36–40 m at 77 km/h; `--m3-spacing-s`),
* 5 waypoints at s(t) + ds, + 2 ds, … + 5 ds (≈ 180 m, ~8 s ahead), each at the human's lateral offset where they
  passed that s (held at the last offset beyond the end of the drive, as M3's spline extends flat), all 5 slots used,
* v* = the human's highest speed over the next 3 s (`--m3-speed-ahead-s`; clipped to 20–80 km/h). v* is M3's
  speed target AND hard ceiling: the mean over the whole ~8 s waypoint stretch made M3 slow ~175 m before the human;
  the next-second speed capped it at the human's speed, so it could never catch up after falling behind.

**Feasibility.** M3 ends an episode as `infeasible_plan` when a plan's peak lateral acceleration exceeds 4 m/s²
(`m3.plan_lat_accel`: a quintic per segment, flat at every waypoint, so 5.77·|Δd|·v²/Δs²). Keyboard lane changes take
~40 m at 77 km/h, and every plan whose lookahead contains one fails: **47–54 % of the human plans** at any spacing.
Dropping them (`--m3-infeasible drop`) would remove every lane change from the data. Default
`--m3-infeasible smooth`: the waypoint offsets are rate-limited to the feasible |Δd| per segment (95 % of the limit),
so the plan heads for the same lane but spreads the move over more distance; every smoothed plan is re-checked. With
2 s spacing the smoothed plans stay close to the human path (max offset: median 0.0 m, 90 % within 1.8 m; 391 plans
contain a lane change) — 1 s spacing (~20 m) lags more (90 % within 2.7 m, 205 lane-change plans), because the
allowed |Δd| grows with Δs². 829 samples, none dropped.

**4. Weighting.** Most decisions are "keep the lane, keep the speed". Each sample carries per-dimension `flags` for
the rare events — a lateral move > 0.5 m (M2's d*; M3's waypoint offsets) and slowing > 1 m/s (v*) — and `bc.py`
up-weights them by their inverse frequency (capped at `--max-weight`), the same scheme as M1's steering / braking.

**5. Cloning and warm start** are unchanged: the same PPO network for that method (only the action head's size
differs: M2 2, M3 12 outputs), the previous-action ego features masked, checkpoint picked by driving seeds 100–109
(M2 / M3: by mean return, since their controllers finish most drives even untrained), then
`training.init_policy=data/bc/<m>_clone.zip`.

```bash
METHOD=m2 CLONE_ONLY=1 scripts/laptop_warm_trial.sh   # runs/demos/m2.npz -> runs/bc/m2/laptop -> data/bc/m2_clone.zip
METHOD=m3 CLONE_ONLY=1 scripts/laptop_warm_trial.sh   # same for M3 (BC 20 epochs: MPC driving tests are slow)
METHOD=m3 scripts/laptop_warm_trial.sh                # + a 60 min warm-start trial
python scripts/view.py live --checkpoint data/bc/m3_clone.zip --seeds 1003   # watch a clone drive
```

### Launching on Great Lakes (step 4)

Same method, reward, budget and seeds as the from-scratch runs; only the warm-start settings differ, and a separate
`runs_root` (`runs/warm/`) keeps the two groups apart. `slurm/launch_batch.sh` submits both kinds for every method
(jobs `m1 m2 m3 m1_warm m2_warm m3_warm`; a warm job needs its `data/bc/<m>_clone.zip` committed):
```bash
DRY=1 bash slurm/launch_batch.sh                            # print the 6 sbatch commands
JOBS="m1_warm m2_warm m3_warm" bash slurm/launch_batch.sh   # warm starts only
```
Repeat for SEED=0,1,2 (at least). Compare with `scripts/plot.py curves runs/m1/r4_straight/seed_*/* runs/warm/m1/r4_straight/seed_*/*`
and `scripts/compare.py`. Optional ablation: warm start with the default `log_std` (drop `init_log_std`) to separate
"good initial actor" from "less exploration noise".

## Files

| path | what |
|---|---|
| `scripts/drive.py` | drive it yourself (Brady) |
| `data/human_demos/*.npz` | the 17 curated episodes (git-tracked) |
| `scripts/build_demos.py` | logs → `runs/demos/<m>.npz` by deterministic replay (`--method m1/m2/m3` labels) |
| `scripts/bc.py` | behaviour cloning → `runs/bc/<m>/<ts>/` (PPO-compatible checkpoint) |
| `data/bc/<m>_clone.zip` | the clones used for the warm starts (git-tracked) |
| `scripts/laptop_warm_trial.sh` | dataset → clone → `data/bc/<m>_clone.zip` → laptop warm-start trial |
| `slurm/launch_batch.sh` | Great Lakes batch: from-scratch and warm-start jobs per method and seed |
| `src/handoff/train.py`, `configs/base.yaml` | warm-start settings `training.init_policy`, `init_log_std`, `critic_warmup_control_steps` |
| `docs/human-data-pipeline.md` | this file |
