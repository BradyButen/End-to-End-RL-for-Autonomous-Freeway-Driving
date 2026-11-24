# Where Should Learning Hand Off to Control?

Nav 565 Team 16 · freeway driving in [MetaDrive](https://github.com/metadriverse/metadrive).

A ladder of RL authority on one freeway with 35–45 km/h traffic: **M1** RL drives end-to-end (lateral and
longitudinal acceleration), **M2** RL picks a lane and a speed that a PID tracks, **M3** RL picks a lane and a max speed (both
as keep / change commands) and an MPC drives it (trajectory, braking, collision-free corridor), **B0** is MetaDrive's rule-based IDM. All RL is PPO
(Stable-Baselines3), same ground-truth observation (the nearest cars ahead / behind in each lane), same reward.

| doc | contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | how the simulator, methods and logging work |
| [docs/TRAINING.md](docs/TRAINING.md) | reward, PPO settings and why, results so far + open problems, Great Lakes workflow |
| [docs/warm-start.md](docs/warm-start.md) | human drives -> dataset -> behaviour-cloning clone -> PPO warm start (M1) |
| [docs/m2-reference-timing.md](docs/m2-reference-timing.md) | why M2's label lookahead had to match its quintic horizon (L ≈ 0.7 T; old d\* action) |
| [docs/m3-mpc-lessons.md](docs/m3-mpc-lessons.md) | M3 corridor + MPC: soft constraints, solver warm starts, space-time corridor rules |
| [docs/rl-lessons.md](docs/rl-lessons.md) | RL training lessons: update size, eval noise, action bias, reward ordering, discount, exploration units, warm starts |
| [docs/SPEC_v2.md](docs/SPEC_v2.md) | original course specification (historical) |

## Setup (dev machine)
Python 3.12; MetaDrive from a pinned GitHub commit (the PyPI wheel does not support 3.12).
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or a CUDA wheel
pip install -e ".[dev]"
python -m metadrive.pull_asset                                         # MetaDrive models/textures (first time)
```
Every shell: `unset PYTHONPATH` (ROS). Under WSL, `export GALLIUM_DRIVER=d3d12` renders the 3D viewer on the GPU.

## Commands
```bash
python scripts/train.py --method m3 --seed 0 --set training.n_envs=8
python scripts/evaluate.py --checkpoint <run>/checkpoints/best.zip --n-workers 4
python scripts/evaluate.py --method b0
python scripts/view.py live --checkpoint <run> --seeds 1003 1027              # needs a display
python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip
python scripts/plot.py curves runs/m*/r5/seed_*/* --metric success
python scripts/drive.py --seeds 1000 1003                                   # drive it yourself; reward plot after
python scripts/plot.py reward runs/drive/<ts>/episodes/<id>.npz --reward r5 # re-score a drive / episode
python scripts/compare.py runs/eval/b0 runs/m*/r5/seed_*/* --out results/  # method comparison table
python scripts/robustness.py --b0 --runs runs/m*/r5/seed_*/*               # robustness sweep (configs/robustness.yaml)
sbatch --export=ALL,POLICY=<run dir or b0> slurm/robustness.sbatch     # robustness v2 on Great Lakes (configs/robust_v2.yaml)

python scripts/validate.py run configs/validation/overtake.json             # M3 on hand-placed waypoints + cars
python scripts/validate.py edit --out configs/validation/my_test.json       # place waypoints / cars, then Run
python scripts/validate.py check                                            # pass/fail suite (before a launch)
CLONE_ONLY=1 scripts/laptop_warm_trial.sh                                   # rebuild data/bc/m1_clone.zip (M1 warm start)
METHOD=m3 MINUTES=15 scripts/laptop_warm_trial.sh                           # 15 min laptop trial (m1: warm-started)
bash slurm/launch_batch.sh                                                  # Great Lakes: m3 m2 m1_warm m1 (seed 0)
```
Any config value can be overridden with `--set key.sub=value`.

## Layout
```
configs/          base.yaml  freeway.yaml (scenario)  m1/m2/m3/b0.yaml (methods)  r5.yaml (reward)
                  validation/*.json (M3 scenarios: waypoints + cars)
src/handoff/
  config.py       load + merge the YAML configs, shared CLI arguments
  sim.py          MetaDrive setup, Frenet road frame, ego / car state, traffic speeds, actuation table
  env.py          DrivingEnv: the one Gym env (decision loop, ground-truth observation, reward, logging) + Method
  reward.py       reward terms (weights in configs/r5.yaml)
  m1.py           M1 end-to-end (lateral / longitudinal accel), B0 IDM baseline
  m2.py           M2: [lane command, speed command] -> quintic reference -> cascaded PID
  m3.py           M3: [lane command, speed command] -> lane plan -> road corridor -> MPC; waypoint spline (validation)
  mpc.py          the CasADi MPC and its fallbacks
  reference.py    lane command (LaneTarget), M3's LanePlan, reference path type, quintic polynomial
  train.py        PPO training (callback, eval / checkpoint cadence, watchdog)
  evaluate.py     evaluation on fixed seeds (parallel workers)
  logs.py         run folders, per-episode recorder (.npz), episode metrics
  viz.py          MetaDrive viewer + overlays, episode overview plots, learning curves
scripts/          train, evaluate, view, drive, plot, validate, compare, robustness, build_demos + bc (M1 clone),
                  laptop_warm_trial.sh, calibrate_actuation, matlab/ (M2 PID tuner)
slurm/            Great Lakes setup + job scripts
runs/             outputs (git-ignored)
```
