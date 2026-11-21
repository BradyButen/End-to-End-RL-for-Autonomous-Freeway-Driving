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
| 4 | from-scratch vs warm-start runs on Great Lakes | ⏳ ready to launch (Brady) |

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

### Launching on Great Lakes (step 4)

Same method, reward, budget and seeds as the from-scratch M1 runs; only the warm-start settings differ. A separate
`runs_root` keeps the two groups apart:
```bash
# from scratch (baseline) - Brady's normal M1 runs
sbatch --export=ALL,METHOD=m1,REWARD=r4_straight,SCENARIO=freeway,SEED=0 slurm/train.sbatch

# warm start from the clone
sbatch --export=ALL,METHOD=m1,REWARD=r4_straight,SCENARIO=freeway,SEED=0,EXTRA="--set training.init_policy=data/bc/m1_clone.zip --set training.init_log_std=-2.0 --set training.critic_warmup_control_steps=100000 --set experiment.runs_root=runs/warm" slurm/train.sbatch
```
Repeat for SEED=0,1,2 (at least). Compare with `scripts/plot.py curves runs/m1/r4_straight/seed_*/* runs/warm/m1/r4_straight/seed_*/*`
and `scripts/compare.py`. Optional ablation: warm start with the default `log_std` (drop `init_log_std`) to separate
"good initial actor" from "less exploration noise".

## Files

| path | what |
|---|---|
| `scripts/drive.py` | drive it yourself (Brady) |
| `data/human_demos/*.npz` | the 17 curated episodes (git-tracked) |
| `scripts/build_demos.py` | logs → `runs/demos/m1.npz` by deterministic replay |
| `scripts/bc.py` | behaviour cloning → `runs/bc/m1/<ts>/` (PPO-compatible checkpoint) |
| `data/bc/m1_clone.zip` | the clone used for the warm start (git-tracked) |
| `src/handoff/train.py`, `configs/base.yaml` | warm-start settings `training.init_policy`, `init_log_std`, `critic_warmup_control_steps` |
| `docs/human-data-pipeline.md` | this file |
