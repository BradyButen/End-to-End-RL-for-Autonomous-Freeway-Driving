# Running on Great Lakes

Training runs on UMich Great Lakes (SLURM). Everything is headless: MetaDrive physics + the 2D BEV observation need
no display or GPU; the GPU (if any) only speeds up PPO's gradient updates.

> **Untested on the cluster.** The scripts in `slurm/` were written on the dev machine. Items marked TODO
> (account, Python module name, CUDA wheel) must be confirmed on Great Lakes first.

## 1. One-time setup (login node — needs internet)
```bash
git clone <repo-url> handoff && cd handoff          # repo not on GitHub yet
module avail python                                  # pick 3.10–3.12; set PY_MODULE accordingly
PY_MODULE=python/3.12 bash slurm/setup_greatlakes.sh            # CUDA PyTorch (gpu / spgpu partitions)
PY_MODULE=python/3.12 CPU_ONLY=1 bash slurm/setup_greatlakes.sh # or CPU-only PyTorch (standard partition)
```
The script creates `.venv`, installs PyTorch (CUDA `cu121` wheel by default; override `CUDA_INDEX`), installs the
package, **downloads MetaDrive's assets** (compute nodes may have no internet), and runs the unit tests and a
headless smoke check.

Edit `#SBATCH --account=TODO_ACCOUNT` in `slurm/*.sbatch` to your account.

## 2. Measure before committing hours
```bash
srun --account=<acct> --partition=gpu --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 --pty bash
source .venv/bin/activate && export SDL_VIDEODRIVER=dummy && unset PYTHONPATH
python tests/profile_timing.py --threads 2                     # per-decision time breakdown + PPO update time
python tests/benchmark_throughput.py --n-envs 4 8 16 --decisions 150
```
Use the numbers to set `training.total_control_steps`, `training.n_envs`, and `--time`.

## 3. Training jobs
```bash
# one run
sbatch --export=ALL,METHOD=m3,REWARD=r0,SCENARIO=freeway_pass,SEED=0 slurm/train.sbatch
# 3 methods × 3 seeds for one reward variant (array 0-8; %3 = at most 3 at once)
sbatch --array=0-8%3 --export=ALL,REWARD=r0,SCENARIO=freeway_pass slurm/train_array.sbatch
# extra config overrides
sbatch --export=ALL,METHOD=m1,EXTRA="--set training.total_control_steps=500000 --set ppo.n_epochs=5" slurm/train.sbatch
```
`train.sbatch` requests 1 GPU + 20 cores and runs `cores − 6` training envs, 4 evaluation workers, 2 PyTorch threads.
For CPU-only: `--partition=standard`, remove `--gres`, and add
`EXTRA="--set training.device=cpu --set training.torch_threads=4"`.

Each run writes `runs/<method>/<reward>/seed_<n>/<timestamp>/` (config, git commit, device, CSV logs, npz episode
logs, checkpoints). `device.txt` records whether CUDA was used.

## 4. Evaluation and analysis
```bash
sbatch --export=ALL,METHOD=m3,CKPT=runs/m3/r0/seed_0/<ts>/checkpoints/best.zip slurm/evaluate.sbatch
sbatch --export=ALL,METHOD=b0,SCENARIO=freeway_pass,OUT=runs/eval/b0 slurm/evaluate.sbatch
python scripts/plot.py curves runs/m*/r0/seed_*/* --metric success --x control_steps --out curves.png
python scripts/plot.py table runs/m*/r0/seed_*/*/final_eval/episodes.csv runs/eval/b0/episodes.csv
```
Copy a run directory back to a machine with a display to watch it: `python scripts/view.py replay --episode
<run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip` (bit-exact re-simulation).

## 5. Time estimates (from laptop measurements; verify with step 2)
Assuming linear scaling with envs, per run at the default 2M control steps with 16 envs + 1 GPU: M1 ≈ 1 h
(gradient updates dominate), M2 ≈ 0.4 h, M3 ≈ 0.75 h (MPC dominates); a full study of 3 methods × 3 seeds × 3
reward variants ≈ 15–20 GPU-node hours. **These are optimistic**: on the dev machine 4 envs gave only 1.4–2× the
1-env throughput (see STATUS.md §4), so measure scaling on the cluster first. The 2M budget itself is a placeholder
until learning curves exist.

## 6. Gotchas
* Always `unset PYTHONPATH` if a system Python stack (e.g. ROS) is on it, and `export SDL_VIDEODRIVER=dummy`.
* `OMP_NUM_THREADS=1` keeps the env subprocesses from oversubscribing cores; PPO's threads are set by
  `training.torch_threads`.
* If the IPOPT CPU-time limit (0.1 s per MPC solve) is hit on a loaded node, M3 falls back and becomes
  load-dependent (non-deterministic). Check `solve_time_p95` / `fallback_count` in `episodes.csv`.
