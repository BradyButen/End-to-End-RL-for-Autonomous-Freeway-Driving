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
python scripts/evaluate.py --method m3 --checkpoint <run>/checkpoints/best.zip
python scripts/evaluate.py --method b0
python scripts/view.py live --method m3 --checkpoint <run>/checkpoints/best.zip     # needs a display
python scripts/plot.py curves runs/m*/r3_fast/seed_*/* --metric success
python scripts/compare.py runs/eval/b0 runs/m*/r3_fast/seed_*/* --out results/        # method comparison table
python scripts/robustness.py --b0 --runs runs/m*/r3_fast/seed_*/*                      # robustness sweep (configs/robustness.yaml)

python scripts/m3_harness.py edit --out configs/m3_harness/my_test.json   # M3 controller harness (hand-placed waypoints)
scripts/run_all_m3.sh                                                      # run every harness scenario in turn
python scripts/track_sine.py                                               # M2 PID tracking test
python tests/benchmark_throughput.py --n-envs 1 4                          # steps/s per method
```
Any config value can be overridden with `--set key.sub=value`.

## Layout
```
configs/        base.yaml, scenario/freeway.yaml, method/{m1,m2,m3,b0}.yaml, reward/r3_fast.yaml, m3_harness/*.json
src/handoff/
  config.py     config loading/merging + shared CLI arguments
  simulator/    MetaDrive config, Frenet road frame, state extraction, viewer/overlays, plots
  rl/env/       core.py (obs, reward, logging, one 0.1 s step), decision.py (decision loop),
                factory.py (method -> env, per-method γ/λ), rewards.py (reward terms)
  rl/methods/   b0_main.py, m1_main.py, m2_main.py -> m2_trajectory.py -> m2_pid.py,
                m3_main.py -> m3_spline.py -> m3_corridor.py -> m3_mpc.py, common.py
  rl/eval/      evaluation harness + episode metrics;  rl/records/  logs and run folders
  rl/train.py, rl/callbacks.py   PPO entry point, eval/checkpoint/watchdog
scripts/        train, evaluate, view, plot, m3_harness, track_sine, calibrate_actuation, matlab/
slurm/          Great Lakes setup + job scripts
tests/          benchmark_throughput.py, profile_timing.py (validation suite to be rebuilt)
11_19_data/     pilot round 1 results
runs/           outputs (git-ignored)
```
