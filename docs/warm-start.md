# Warm start (M1): human drives → behaviour cloning → PPO

A warm start begins PPO from a network that already imitates human driving (a behaviour-cloning "clone") instead of
from random weights. Question for the report: does it reach good driving with less RL experience than training from
scratch (same method, reward, budget, seeds)? Lessons from the runs: `docs/rl-lessons.md` §7–8; batch results:
`docs/TRAINING.md` §5.

Since batch 6 (2026-11-23) only **M1** is warm-started. M2 / M3 now decide lanes (DESIGN §4), which the old
relabelled demos (d\*, v\* and waypoint labels) do not fit, and their low-noise warm starts had pinned them to
following traffic. The relabelling code is in git history (commit d55d766: `scripts/build_demos.py --method m2/m3`,
`scripts/demo_relabel.py`).

```
 you drive            curate               dataset                    clone (BC)               PPO warm start
 scripts/drive.py ──▶ data/human_demos/ ──▶ scripts/build_demos.py ──▶ scripts/bc.py ─────────▶ train.py
 runs/drive/<ts>/     17 .npz, tracked      replay + relabel           same network as PPO      training.init_policy
                                            runs/demos/m1.npz          data/bc/m1_clone.zip
```

## Commands

```bash
python scripts/drive.py --seeds 11 87 142 --out runs/drive/<batch>     # drive (display); TRAINING seeds 0-999 only
cp runs/drive/<batch>/episodes/<id>.npz data/human_demos/              # keep the good ones (see Curation)

CLONE_ONLY=1 scripts/laptop_warm_trial.sh             # dataset -> clone -> data/bc/m1_clone.zip (REBUILD=1 to redo)
MINUTES=15 scripts/laptop_warm_trial.sh               # + a short laptop warm-start trial
python scripts/view.py live --checkpoint data/bc/m1_clone.zip --seeds 1003   # watch the clone drive

JOBS=m1_warm bash slurm/launch_batch.sh               # Great Lakes (DRY=1 to print only); commit the clone first
```

## 1. Recording and curation (`scripts/drive.py`)

The same env the RL trains in; only the controls come from a keyboard / gamepad (MetaDrive's `ManualControlPolicy`).
W throttle, S brake, A / D steer; **X ends the episode and saves it**; Esc quits without saving; avoid T (MetaDrive's
autopilot would be logged as you) and R (reset). Seeds 1000+ are the evaluation seeds: never drive them.

Kept: `success`, no contact, no sustained tailgating (the clone copies everything). 17 kept of 30 driven: seeds
11 25 61 87 239 318 391 428 512 563 579 677 765 822 856 871 903, ~8,200 control steps (~14 min), 73–78 km/h,
2–10 lane changes each, all on a keyboard (steering is exactly 0 on 88 % of steps; braking on ~1 %), spawned at
40 km/h; `build_demos.py` replays drives at their own first logged speed, whatever the training spawn speed. 13 more drives spawned at 80 km/h (2026-11-23) are on the local branch
`archive/2026-11-23-session`; drives at different spawn speeds must be built into separate datasets.

## 2. Dataset (`scripts/build_demos.py`)

* **Observations by replay.** Logs hold states and controls, not the policy's observation. MetaDrive is
  deterministic, so each drive is replayed on its seed with the logged steering / throttle and the observation is
  recorded before every step (checked against the log to ≤ 5 cm; same termination at the same step). Sample t =
  (observation before step t, action at step t). The dataset holds whatever observation the env builds today, so
  rebuild it whenever the observation changes.
* **Labels in M1's action**: the logged steering → target lateral acceleration at the speed it was applied
  (kinematic bicycle, ÷ `steer_lat_accel` 8 m/s²), the logged throttle_brake → target acceleration through the
  actuation table; both clipped to [−1, 1].

## 3. Clone (`scripts/bc.py`)

* **Same network as PPO** (SB3 `MultiInputPolicy` with M1's `ppo` kwargs), saved as a normal SB3 `.zip`: it loads
  into PPO and drives in `view.py` / `evaluate.py` unchanged.
* **Only the actor is trained** (policy MLP + action head; weighted MSE on the action mean, each dimension divided
  by its label variance). Critic and log_std untouched.
* **Copycat guard.** The ego vector holds the previous step's steering / throttle ≈ the label; a net given those
  learns "repeat the last action". They are zeroed and their first-layer weights held at 0 (`--keep-prev-action` off).
* **Imbalance.** Steering and braking samples are up-weighted by inverse frequency (capped, `--max-weight`).
* **Checkpoint chosen by driving**, not validation loss (which did not predict driving): each epoch the clone drives
  seeds 100–109 (training range, not in the demos); best = furthest mean distance. Held-out validation episodes: 3.
* The chosen clone is copied to `data/bc/m1_clone.zip` (tracked, so Great Lakes gets it with `git pull`).

What the BEV-era clone did alone: 0 % success (drifted off in edge lanes within seconds, never braked: the keyboard
demos have no "drifting → steer back" or "slow car → brake" examples, so it was covariate shift; regularisation did
not help). Warm-started PPO then reached 25–35 % success at ~75 km/h but kept crashing into cars ahead without
braking.

## 4. Warm start (`src/handoff/train.py`)

| setting | warm job | what it does |
|---|---|---|
| `training.init_policy` | `data/bc/m1_clone.zip` | copies the clone's **actor** into the fresh PPO model; critic and log_std stay fresh. Shapes are checked (a clone for another observation errors out). |
| `training.init_log_std` | from `configs/m1.yaml`: [−1.9, −0.7] | std 0.15 on lateral (1.2 m/s²), 0.5 on accel. One std of 0.135 (batches 1–5) never sampled a brake, since the clone sits at accel +0.8. |
| `training.critic_warmup_control_steps` | 100,000 | actor **frozen**, only the critic trains, so the first real PPO updates use sensible advantages instead of a random critic's and do not wreck the clone |

All off by default except the noise. Everything else is identical to a from-scratch run; warm runs write to
`runs/warm/`. The step-0 eval is the clone itself. In the log: `warm start: actor from ...`,
`[warm start] actor frozen / unfrozen at ...`; `train/actor_frozen` in `train_progress.csv`; `warm_start.txt` in the
run folder. A drop right after unfreezing = warm-up too short or noise too large.

## Files

| path | what |
|---|---|
| `scripts/drive.py` | drive it yourself |
| `data/human_demos/*.npz` | the 17 curated drives (tracked) |
| `scripts/build_demos.py` | drives → `runs/demos/m1.npz` (replay + M1 labels) |
| `scripts/bc.py` | behaviour cloning → `runs/bc/m1/<ts>/` |
| `data/bc/m1_clone.zip` | the clone the warm job loads (tracked; rebuild after observation changes) |
| `scripts/laptop_warm_trial.sh` | dataset → clone → short laptop trial (M2 / M3: from scratch) |
| `slurm/launch_batch.sh` | Great Lakes batch: m1, m2, m3 from scratch and m1_warm, per seed |
