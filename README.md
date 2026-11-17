# Where Should Learning Hand Off to Control?

Nav 565 Team 16 · simulated freeway driving in [MetaDrive](https://github.com/metadriverse/metadrive).

We compare a **ladder of RL authority** on one freeway scenario: **M1** RL drives end-to-end, **M2** RL picks a
short trajectory (lateral target + speed) that a PID tracks, **M3** RL picks sparse waypoints that an MPC executes
inside a hard collision-free corridor, and **B0** is MetaDrive's rule-based IDM. All RL uses PPO (Stable-Baselines3).

| doc | contents |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | **start here** — what works, validation results, known issues, next steps |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the code works, module by module |
| [docs/SPEC_v2.md](docs/SPEC_v2.md) | the specification (v2, changes from v1.0 highlighted) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | every decision / deviation from the v1.0 spec, with IDs referenced in code |
| [docs/GREAT_LAKES.md](docs/GREAT_LAKES.md) | running training on the cluster |

## Setup (dev machine)
Python 3.12 (3.10–3.12 work). MetaDrive is installed from a pinned GitHub commit because the PyPI wheel does not
support 3.12.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or a CUDA wheel
pip install -e ".[dev]"
python -m metadrive.pull_asset                                         # MetaDrive models/textures (first time)
```
Every shell: `unset PYTHONPATH` if ROS or another Python stack is on it. Under WSL, `export GALLIUM_DRIVER=d3d12`
makes the 3D viewer use the GPU (otherwise it renders in software).

## Common commands
```bash
pytest                                              # unit tests (no simulator, ~30 s)
python scripts/validate.py --all                    # spec §17 checks against MetaDrive -> runs/validation/
python scripts/profile_timing.py                    # where the time goes per decision + PPO update cost
python scripts/benchmark_throughput.py --n-envs 1 4 # steps/s per method

python scripts/view.py live --method b0 --scenario freeway_dense          # watch (needs a display)
python scripts/view.py live --method m3 --checkpoint <run>/checkpoints/best.zip
python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip

python scripts/train.py --method m3 --reward r0 --scenario freeway_pass --seed 0 --set training.n_envs=8
python scripts/evaluate.py --method m3 --checkpoint <run>/checkpoints/best.zip
python scripts/evaluate.py --method b0
python scripts/plot.py curves runs/m*/r0/seed_*/* --metric success
```
Any config value can be overridden with `--set key.sub=value` (e.g. `--set method.mpc.horizon=30`).

## Layout
```
configs/        base.yaml, scenario/, method/ (m1 m2 m3 b0), reward/ (r0 r1_sparse r2_dense_comfort)
src/handoff/    sim/ (MetaDrive env, scenario, Frenet road frame, state) · controllers/ (PID, trajectory, spline,
                safe set, corridor, MPC, actuation) · wrappers/ (core env + method wrappers) · rewards/ · rl/ ·
                eval/ · logging/ · viz/
scripts/        train, evaluate, validate, benchmark, profile, view, plot, calibrate; debug/ probes
slurm/          Great Lakes setup + job scripts
tests/          pytest unit tests (kinematic plant stands in for MetaDrive)
runs/           outputs (git-ignored)
```
