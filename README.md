# Where Should Learning Hand Off to Control?

Nav 565 Team 16 · freeway driving in [MetaDrive](https://github.com/metadriverse/metadrive).

A ladder of RL authority on one freeway with 40 km/h traffic: **M1** RL drives end-to-end, **M2** RL picks a
lateral target + speed that a PID tracks, **M3** RL picks sparse waypoints + a speed that an MPC executes inside a
collision-free corridor, **B0** is MetaDrive's rule-based IDM. All RL is PPO (Stable-Baselines3), same observation,
same reward.

| doc | contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | how the simulator, methods and logging work |
| [docs/TRAINING.md](docs/TRAINING.md) | reward, PPO settings and why, pilot results, Great Lakes workflow |
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
python scripts/plot.py curves runs/m*/r4_straight/seed_*/* --metric success
python scripts/drive.py --seeds 1000 1003                                   # drive it yourself; reward plot after
python scripts/plot.py reward runs/drive/<ts>/episodes/<id>.npz --reward r4 # re-score a drive / episode
python scripts/compare.py runs/eval/b0 runs/m*/r4_straight/seed_*/* --out results/  # method comparison table
python scripts/robustness.py --b0 --runs runs/m*/r4_straight/seed_*/*               # robustness sweep (configs/robustness.yaml)

python scripts/validate.py run configs/validation/overtake.json             # M3 on hand-placed waypoints + cars
python scripts/validate.py edit --out configs/validation/my_test.json       # place waypoints / cars, then Run
python scripts/validate.py check                                            # pass/fail suite (before a launch)
```
Any config value can be overridden with `--set key.sub=value`.

## Layout
```
configs/          base.yaml  freeway.yaml (scenario)  m1/m2/m3/b0.yaml (methods)  r4_straight.yaml (reward)
                  validation/*.json (M3 scenarios: waypoints + cars)
src/handoff/
  config.py       load + merge the YAML configs, shared CLI arguments
  sim.py          MetaDrive setup, Frenet road frame, ego / car state, traffic speeds, actuation table
  env.py          DrivingEnv: the one Gym env (decision loop, observation, reward, logging) + Method base class
  reward.py       reward terms (weights in configs/r4_straight.yaml)
  m1.py           M1 end-to-end, B0 IDM baseline
  m2.py           M2: (d*, v*) -> quintic reference -> cascaded PID
  m3.py           M3: waypoints -> spline -> road corridor -> e-brake gate -> MPC
  mpc.py          the CasADi MPC and its fallbacks
  reference.py    reference path type + quintic polynomial (shared by M2 and M3)
  train.py        PPO training (callback, eval / checkpoint cadence, watchdog)
  evaluate.py     evaluation on fixed seeds (parallel workers)
  logs.py         run folders, per-episode recorder (.npz), episode metrics
  viz.py          MetaDrive viewer + overlays, episode overview plots, learning curves
scripts/          train, evaluate, view, drive, plot, validate, calibrate_actuation, matlab/ (M2 PID tuner)
slurm/            Great Lakes setup + job scripts
runs/             outputs (git-ignored)
```
