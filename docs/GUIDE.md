# Guide

Everything needed to install, run, train and understand the project. Motivation: [README](../README.md).

1. [Setup](#1-setup)
2. [Watch the trained models](#2-watch-the-trained-models)
3. [Train](#3-train)
4. [Warm start for M1 (human drives → clone)](#4-warm-start-for-m1)
5. [Training on a Slurm cluster](#5-training-on-a-slurm-cluster)
6. [How it works](#6-how-it-works)
7. [Repository layout](#7-repository-layout)

## 1. Setup
Python 3.10–3.12. MetaDrive is installed from a pinned GitHub commit because the PyPI wheel does not support 3.12.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or a CUDA build
pip install -e ".[dev]"
python -m metadrive.pull_asset                                         # MetaDrive models / textures (once)
```
If ROS is installed, run `unset PYTHONPATH` in every shell. The 3D viewer needs a display (X11 or WSLg). On WSL,
`export GALLIUM_DRIVER=d3d12` renders it on the GPU. On headless machines, set `SDL_VIDEODRIVER=dummy`.

## 2. Watch the trained models
`models/` holds the final policy for each method: the best evaluation checkpoint (`checkpoints/best.zip`, with
`best.txt` saying which eval round) and the config it was trained with.

| folder | method |
|---|---|
| `models/m1_warm` | M1, warm-started from the human-drive clone (seed 0) |
| `models/m2` | M2, RL + PID (seed 0) |
| `models/m3` | M3, RL + MPC (seed 0) |

```bash
python scripts/view.py live --checkpoint models/m3 --seeds 1003 1027     # any run folder or .zip
python scripts/view.py live --method b0 --seeds 1003                     # the rule-based baseline
python scripts/view.py live --checkpoint models/m2 --no-render           # headless: reports only
python scripts/view.py live --checkpoint models/m3 --clean --paused      # no MetaDrive HUD, start paused
python scripts/view.py replay --episode <out>/episodes/<id>.npz --checkpoint models/m3   # re-simulate a logged episode
```
Seeds 1000–1039 are the evaluation seeds (default: the first three). Every episode's full log is written to
`runs/view/<method>_<time>/episodes/<id>.npz` (`replay` re-simulates it). Any config value can be changed with
`--set`, e.g. `--set scenario.traffic.density=0.25`.

Viewer keys:

| key | action |
|---|---|
| Space / P | pause / resume |
| N | play 0.2 s, then pause |
| M | all overlays on / off |
| I / D | text panel on / off / move it to another corner |
| 1–6 | one overlay layer each: trail, reference / lane plan, MPC prediction, corridor, keep-out boxes, points |
| O | corridor display style (all stages / first stage per face / every 1 s / fade / box at 1 s) |
| C | screenshot |
| B / Q, mouse wheel | top-down / chase camera, zoom |
| X | end the episode |

Avoid R, [ and ]: MetaDrive resets or changes the map without the env knowing.

## 3. Train
```bash
python scripts/train.py --method m3 --seed 0 --set training.n_envs=8
python scripts/train.py --method m1 --seed 0 --set training.init_policy=data/bc/m1_clone.zip \
  --set training.critic_warmup_control_steps=100000 --set experiment.runs_root=runs/warm   # M1 warm start
```
Each run writes `runs/<method>/r5/seed_<n>/<timestamp>/`:
* `config.yaml`
* `checkpoints/`: `best.zip` + `best.txt`, `final.zip`, and periodic checkpoints
* `episodes.csv`: one row per training or eval episode, with termination, return per reward term, speed, gaps,
  comfort and solver stats
* `train_progress.csv`
* `episodes/*.npz`: full logs of every eval episode and every 50th training episode

Training stops at `training.total_control_steps` or `training.max_wall_minutes`, whichever comes first. It
evaluates on the 40 evaluation seeds every `eval.freq_control_steps`. The best checkpoint is the one with the
highest eval success rate, with ties broken by return. A status line (steps/s, memory, latest eval) prints
every `training.progress_seconds`.

## 4. Warm start for M1
M1 is hard to train from scratch, so it can start from a behaviour-cloning "clone" of human drives instead of
random weights. The clone used for `models/m1_warm` is tracked at `data/bc/m1_clone.zip`, so steps 1–3 are only
needed to rebuild it, for example after the observation changes.

```
 1. drive                  2. dataset                        3. clone                       4. PPO
 warm_start/drive.py  ──▶  warm_start/build_demos.py  ──▶   warm_start/bc.py  ──▶          train.py
 keep good drives in       replays each drive, records       PPO's own network fit          training.init_policy=
 data/human_demos/         observations + M1 action labels   to the human's actions         data/bc/m1_clone.zip
```
```bash
python scripts/warm_start/drive.py --seeds 11 87 142            # drive it yourself (W/S/A/D, X ends + saves)
cp runs/drive/<ts>/episodes/<id>.npz data/human_demos/          # keep clean, successful drives
python scripts/warm_start/build_demos.py                        # data/human_demos/*.npz -> runs/demos/m1.npz
python scripts/warm_start/bc.py --n-workers 4                   # -> runs/bc/m1/<ts>/
cp runs/bc/m1/<ts>/checkpoints/best.zip data/bc/m1_clone.zip    # use it
python scripts/view.py live --checkpoint data/bc/m1_clone.zip   # watch the clone alone
```
Notes:
* **Drive on training seeds only** (0–999). Seeds 1000+ are the evaluation seeds.
* **Avoid T while driving.** It switches on MetaDrive's autopilot, which would be logged as you.
* **What the 17 tracked drives contain.** About 14 minutes of keyboard driving at 73–78 km/h, with 2–10 lane
  changes each. All start at 40 km/h.
* **Observations by replay.** The logs hold states and controls, not the policy's observation. `build_demos.py`
  replays each drive on its seed (MetaDrive is deterministic) and checks the replay against the log to 5 cm.
* **Labels.** The logged steering becomes a target lateral acceleration (kinematic bicycle) and the throttle
  becomes a target acceleration (actuation table). Together these are M1's action.
* **What the clone learns.** Only the actor (policy MLP and action head) is fitted; the critic is not. The
  previous-action inputs are masked, so the clone cannot learn "repeat the last action". Steering and braking
  samples are up-weighted.
* **Which epoch is kept.** The epoch whose clone drives furthest on seeds 100–109, not the one with the lowest
  validation loss.
* **Warm-start settings.** `training.init_policy` copies only the actor into PPO. The critic then trains alone,
  with the actor frozen, for `critic_warmup_control_steps`, so the first PPO updates do not wreck the clone.
  Exploration noise comes from `configs/m1.yaml`.

## 5. Training on a Slurm cluster
The models were trained on the University of Michigan's Great Lakes cluster. The scripts work on any Slurm cluster
that uses environment modules; adjust the partitions and module names in `slurm/` to match yours.
```bash
bash slurm/setup.sh                        # once, on a login node: venv + CUDA PyTorch + MetaDrive assets
CPU_ONLY=1 bash slurm/setup.sh             # ... or CPU-only PyTorch
export SBATCH_ACCOUNT=<your account>
sbatch --export=ALL,METHOD=m3,SEED=0 slurm/train.sbatch              # one run
bash slurm/launch_batch.sh                                           # m3, m2, m1_warm, m1 for seed 0
SEEDS="0 1" JOBS="m3 m1_warm" DRY=1 bash slurm/launch_batch.sh       # print the sbatch commands only
```
`train.sbatch` uses 20 cores: 14 training envs, 4 eval workers and the main process. The policies are small MLPs,
so `launch_batch.sh` runs on CPU nodes. Jobs are given 8 h. `max_wall_minutes` (450 for M1 / M2, 420 for M3) stops
training in time for the final evaluation, and the step budgets are set above what 7.5 h can reach, so the wall
clock is the real stop. Methods are compared at equal simulated experience (control steps), not at equal wall time.

## 6. How it works

### Scenario
* **Simulator.** Stock MetaDrive `MetaDriveEnv`, configured but not patched (`sim.py`). MetaDrive's own observation
  is not used.
* **Road.** 17 straight blocks, giving an 852–1144 m route. Three lanes, 3.5 m wide. Crossing the centre line or
  the road edge ends the episode.
* **Traffic.** MetaDrive's IDM cars at density 0.15 (about 36 cars). They are placed over the route and all drive
  from t = 0, at 45 / 40 / 35 km/h in the left / middle / right lane.
* **Ego car.** Fixed dynamics (`static_default`). Spawns in the first block, in a random lane, at 40 km/h. Speed
  cap 80 km/h.
* **Seeds.** Training draws the map and traffic seed from [0, 1000); evaluation uses the fixed seeds 1000–1039.
* **Timing.** Physics runs at 0.02 s and control at 0.1 s. Episodes are capped at 150 s.
* **Episode end.** Success at the end of the route, otherwise collision, off-road or timeout.
* **Determinism.** Objects are destroyed on reset (`force_destroy`), so an episode re-simulates bit-exactly from
  its seed and checkpoint.

### The shared env (`env.py`)
One Gymnasium env serves every method. One RL step is one decision:
```
decision = method.decide(action)          # M1: controls; M2 / M3: lane command + speed command
for each control step of the decision:    # M1 1, M2 5, M3 10 (0.1 s each)
    [steering, throttle_brake] = method.control()
    MetaDrive step -> ego and car states -> reward terms -> log
```
**Observation**, the same ground truth for every method (what a perception and tracking stack would provide):

| part | size | contents |
|---|---|---|
| `ego` | 8 | speed, position across the road, offset from the lane centre, heading error, last steering, last throttle, target lane − current lane, time left |
| `lanes` | 33 | for the lane to the left, the ego's lane and the lane to the right: whether the lane exists, then the nearest car ahead and behind (gap, lateral offset, relative speed, 1/TTC) within 100 m |
| `cmd` | 1 | M2 / M3 only: their current target speed |

The `ego` vector is frozen, because the M1 clone was trained on exactly that layout. Additions for M2 / M3 go in
`cmd`.

**Reward** (`r5`, `configs/r5.yaml`; raw terms in `reward.py`). The aim is to get to the end fast, like an
aggressive but competent driver. Positive terms reward progress and speed (break-even at 60 km/h, so following
35–45 km/h traffic costs reward every second). A crash or going off the road costs −4000, so crashing is always
worse than any finish. Smaller penalties cover distance from the lane centre, heading error, rapid steering changes
and headways under 0.5 s. Finishing adds +150.

### The methods
**M1, end-to-end** (`m1.py`). The action is continuous: [lateral acceleration (±8 m/s²), target acceleration
(−4 to +2.9 m/s²)], every 0.1 s. A kinematic bicycle model turns the lateral acceleration into a wheel angle. A
measured throttle→acceleration lookup table (`actuation.table` in `base.yaml`) turns the target acceleration into
throttle or brake. Working in physical units keeps the exploration noise meaningful: with raw throttle, zero-mean
noise netted out to hard braking.

**Lane and speed commands** (M2 and M3, `reference.py`). The action is `MultiDiscrete [3, 3]`:
* **Lane:** keep / left / right moves the target lane by one, saturating at the edge lanes. A lane change runs to
  completion before new lane commands count.
* **Speed:** hold / +10 / −10 km/h on a target speed v\*. It starts at 40 km/h, persists until changed, and is part
  of the observation.

Training starts biased toward "keep lane, hold speed" (`training.init_default_prob`).

**M2, RL + PID** (`m2.py`). Every 0.5 s it builds a 2 s quintic from the current lateral state to the target lane
centre, plus an acceleration-limited speed ramp to v\*. Cascaded PIDs (heading, lateral, speed, with acceleration
feed-forward) track it. There is no safety layer: braking for a car ahead is the RL's job.

**M3, RL + MPC** (`m3.py`, `mpc.py`). Every 1 s the RL sets the target lane and the maximum speed v\*. Everything
below that is the controller's:
1. **Lane plan.** Built when the target lane changes: a quintic d(s) to the new lane centre over 2.5 s of travel,
   lengthened to keep lateral acceleration within 5 m/s².
2. **Corridor.** Rebuilt every control step: one road-aligned box per MPC stage. The sides sit at the outer lanes.
   Each car's keep-out zone adds a front, back or side face, decided stage by stage from where the plan and the
   car's predicted position are at that stage. This lets the plan pass a slower car and merge in front of it.
3. **MPC.** CasADi + IPOPT on a kinematic bicycle with 0.1 s steering lag, over a 5 s horizon (50 × 0.1 s). It
   tracks the plan with v\* as a speed ceiling and brakes down to −8 m/s². The corridor is a soft constraint with an
   exact penalty: it holds whenever possible, and otherwise the solver returns the least-violating plan rather
   than failing.
4. **Failures.** A failed solve is retried from a braking guess. After that the previous plan is replayed for up
   to 1 s, then the car holds speed and lane until a solve succeeds. Failures never end the episode.

**B0, baseline.** MetaDrive's IDM policy drives at 70 km/h with its own lane changes. It is evaluation only.

### PPO settings (`configs/base.yaml`, per-method overrides in `configs/m*.yaml`)
* **Algorithm.** Stable-Baselines3 PPO with separate 256-256 policy and value MLPs, identical for all methods.
* **Equal experience.** Budgets and eval / checkpoint cadences count control steps (0.1 s of driving), so each
  method gets the same simulated experience. Rollouts are 128 s of driving per env for every method
  (`n_steps` M1 1280, M2 256, M3 128).
* **Discount and GAE.** γ = 0.999 and GAE λ = 0.995 are set per 0.1 s and scaled to each decision period, so all
  methods get the same ~100 s lookahead and ~7 s credit window. The long lookahead makes the +150 finish matter
  from the start; with a ~10 s horizon, crawling looked cheaper than risking a crash.
* **Small updates.** `n_epochs` 4 and `target_kl` 0.03 on every method (plus a learning rate of 1e-4 on M2) keep
  each update small. Larger updates made M2 collapse.
* **Exploration.** M1 uses a per-dimension initial std: 1.2 m/s² lateral, ~1.5 m/s² acceleration, so braking gets
  sampled. M2 / M3 use an entropy bonus of 0.01 and the "keep lane, hold speed" bias.
* **Reward normalisation.** Rewards are normalised by the running return std (clipped at ±100, so the crash penalty
  survives). Logs and evals report raw returns.
* **Watchdog.** If no progress is made for 15 minutes, a hung worker is assumed and training stops with exit
  code 3. A crashed worker ends the job and writes `FAILED.txt` and `checkpoints/crash.zip`.

## 7. Repository layout
```
configs/            base.yaml (shared), freeway.yaml (scenario), m1/m2/m3/b0.yaml (methods), r5.yaml (reward)
src/handoff/
  config.py         load and merge the YAML configs, --set overrides
  sim.py            MetaDrive setup, road (Frenet) frame, ego / car state, traffic speeds, actuation table
  env.py            DrivingEnv: decision loop, observation, reward, logging; the Method base class
  reward.py         reward terms
  m1.py             M1 end-to-end, B0 baseline
  m2.py             M2: quintic reference + cascaded PID
  m3.py             M3: lane plan, corridor
  mpc.py            the CasADi MPC and its fallbacks
  reference.py      lane / speed commands, lane plan, quintic polynomial
  train.py          PPO training loop, evaluation / checkpoint cadence, watchdog
  evaluate.py       parallel deterministic evaluation (used during training)
  logs.py           run folders, per-episode recorder (.npz), episode metrics
  viz.py            the 3D viewer and its overlays, episode plots
scripts/
  train.py          train one (method, seed)
  view.py           watch / replay a policy
  warm_start/       drive.py, build_demos.py, bc.py: human drives -> M1 clone
slurm/              setup.sh, train.sbatch, launch_batch.sh
data/               human_demos/ (17 drives), bc/m1_clone.zip (the M1 warm start)
models/             final trained policies (M1 warm, M2, M3)
results/robustness/ robustness evaluation of the final policies: raw per-episode data + summary tables
runs/               outputs (not tracked)
```
