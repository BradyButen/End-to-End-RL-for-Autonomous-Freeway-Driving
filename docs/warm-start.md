# How the warm start works

A warm start begins PPO from a network that already imitates human driving (a behaviour-cloning "clone"), instead
of from random weights. The question for the report: does it reach good driving with less RL experience than
training from scratch (same method, reward, budget and seeds)? This page is the mechanism end to end; data details
and results are in `docs/human-data-pipeline.md`, lessons in `docs/rl-lessons.md` §7.

```
 human drives          dataset (per method)        clone (BC)                 PPO warm start
 data/human_demos/ ──▶ build_demos.py ────────────▶ bc.py ──────────────────▶ train.py (training.init_policy)
 17 keyboard drives    replay + relabel            same network as PPO        actor copied, critic learns first,
                       runs/demos/<m>.npz          data/bc/<m>_clone.zip      then normal PPO
```

## 1. Dataset: turn drives into (observation, action) pairs — `scripts/build_demos.py`

* **Observations by replay.** The logs hold states and controls, not the 84×84 top-down image the policy sees.
  MetaDrive is deterministic, so each drive is replayed on its seed with the logged steering / throttle, and the
  real observation is recorded before every step (checked against the log to ≤ 5 cm). All methods see the same
  observation, so one replay serves M1, M2 and M3.
* **Actions in each method's own terms (hindsight relabelling).** The human pressed throttle and steered; M2 and M3
  never output those, so the label is what the human's next seconds show they were going for:

  | method | decision | label |
  |---|---|---|
  | M1 | 0.1 s | the human's [steering, throttle] → M1's [steering, target accel] via the actuation table |
  | M2 | 0.5 s | d\*, v\* = the human's lateral offset and speed **1.3 s** later (matched to the 2 s quintic, `docs/m2-reference-timing.md`) |
  | M3 | 1 s | 5 waypoints on the human's path, spaced at speed × 2 s; v\* = the human's highest speed over the next 3 s; plans over the lateral-accel limit smoothed |

  Labels are encoded with the method's own `encode()` into its [−1, 1] action.

## 2. Clone: supervised learning on that dataset — `scripts/bc.py`

* **Same network as PPO.** `bc.py` builds exactly the model `train.py` builds (SB3 `MultiInputPolicy` with the
  method's `ppo` settings: CNN on the image + ego vector → MLP → action mean) and saves it as a normal SB3 `.zip`.
  That is what makes the clone loadable into PPO, and drivable with `view.py` / `evaluate.py`.
* **Only the actor is trained:** CNN + policy MLP + action head, loss = weighted MSE between the network's action
  mean and the label (each action dimension divided by its label variance). The critic and the action noise (log_std)
  are left untouched.
* **Guards:** the ego vector contains the previous step's steering / throttle, which almost equals the label; a net
  given those learns "repeat the last action" and ignores the road, so they are zeroed and their weights held at 0.
  Rare samples (steering, braking, lane moves, slow-downs) are up-weighted by inverse frequency.
* **Checkpoint chosen by driving,** not by validation loss (which turned out unrelated to driving): every epoch (M3:
  every 2nd) the clone drives seeds 100–109 (training range, not in the demos; eval seeds untouched) and the best
  epoch is kept — furthest distance for M1, best return for M2 / M3.
* The chosen clone is copied to `data/bc/<m>_clone.zip` (git-tracked, so Great Lakes gets it with `git pull`).

## 3. Warm start: loading the clone into PPO — `src/handoff/train.py`

Three settings (all off by default; `slurm/launch_batch.sh` sets them for `*_warm` jobs):

| setting | value | what it does |
|---|---|---|
| `training.init_policy` | `data/bc/<m>_clone.zip` | after PPO builds its model, `warm_start()` copies the clone's **actor** weights (CNN, policy MLP, action head) into it. The critic and log_std keep their fresh values. Shapes are checked: a clone for another method errors out. |
| `training.init_log_std` | −2.0 | sets the action noise to std 0.135 (from scratch: 0.5). The clone's actions are precise (human steering std ~0.06); std 0.5 would bury them in noise and PPO would start from something close to random. |
| `training.critic_warmup_control_steps` | 100,000 (laptop trial: 60,000) | the actor is **frozen** and only the critic trains for this long (~6 PPO updates on Great Lakes). |

Why the critic warm-up matters: PPO's update direction is the *advantage* (how much better an action did than the
critic expected). A freshly initialised critic predicts nonsense, so the first advantages are noise — and with a
good actor, noisy updates mostly make it worse ("the first PPO updates wreck the clone"). Freezing the actor lets
the critic first learn what the clone's driving is worth; the first real policy updates then use sensible
advantages. Mechanics (`TrainingCallback`): `_freeze_actor(True)` at the start sets `requires_grad=False` on the actor
parameters — including the shared CNN, so the critic learns its own MLP head on the clone's image features; after the
first rollout past the warm-up, `_freeze_actor(False)` and training is ordinary PPO. The log prints
`[warm start] actor frozen …` / `actor unfrozen at … control steps`, `train_progress.csv` has `train/actor_frozen`,
and the run folder has `warm_start.txt`.

Everything else is identical to a from-scratch run: reward, budget, PPO settings, eval seeds, reward normalisation
(starts fresh). The **step-0 eval is the clone itself**, so a warm run's learning curve starts at the clone's
performance. Warm runs write to `runs/warm/<m>/…` to keep the two groups apart.

## 4. What it looks like in practice

| method | clone alone (test drives) | expectation for the warm start |
|---|---|---|
| M1 | 0 % success (never brakes, drifts in edge lanes) | laptop pilot: never fell into crawling; first positive M1 returns within ~300–500k steps |
| M2 | ~10 % (keeps lane, speeds up, rear-ends) | uncertain: untrained M2 already finishes ~100 % at 40 km/h; the clone starts faster but crashier |
| M3 | 100 % (987 m, return −205) | strongest case: PPO starts from a policy that finishes, and has to learn to go faster without crashing |

What to watch in the logs: `warm start: actor from data/bc/<m>_clone.zip` at start-up; the step-0 eval
(`eval_0000000000`) ≈ the clone; whether success / return hold up after `actor unfrozen` (a drop right after
unfreezing = the critic warm-up was too short or the noise too large); and the comparison against the from-scratch
curve at equal control steps.

## 5. Commands

```bash
# build a clone (laptop): dataset -> BC -> data/bc/<m>_clone.zip
METHOD=m3 CLONE_ONLY=1 scripts/laptop_warm_trial.sh            # REBUILD=1 to redo the dataset + clone
python scripts/view.py live --checkpoint data/bc/m3_clone.zip --seeds 1003   # watch the clone

# 1 h laptop trial of the warm start
METHOD=m3 scripts/laptop_warm_trial.sh

# Great Lakes: warm-start jobs (and from-scratch baselines)
JOBS="m1_warm m2_warm m3_warm" bash slurm/launch_batch.sh
JOBS="m3" bash slurm/launch_batch.sh

# by hand, any method
python scripts/train.py --method m3 --set training.init_policy=data/bc/m3_clone.zip \
  --set training.init_log_std=-2.0 --set training.critic_warmup_control_steps=100000 \
  --set experiment.runs_root=runs/warm
```

## Possible next steps

* **DAgger** (corrective data): the clone drives, a human takes over when it goes wrong, those states get the human's
  labels, retrain. Targets exactly what the M1 / M2 clones lack (recovery, braking for traffic).
* **Ablation:** warm start with the default noise (drop `init_log_std`) to separate "good initial actor" from "less
  exploration".
* **Critic pre-training on demo returns** instead of (or before) the on-policy warm-up.
