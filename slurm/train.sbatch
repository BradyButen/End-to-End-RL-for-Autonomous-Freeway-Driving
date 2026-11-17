#!/bin/bash
# One training run: (method, reward variant, scenario, seed). Submit from the repo root:
#
#   sbatch --export=ALL,METHOD=m3,REWARD=r0,SCENARIO=freeway,SEED=0 slurm/train.sbatch
#   sbatch --export=ALL,METHOD=m1,EXTRA="--set training.total_control_steps=500000" slurm/train.sbatch
#
# GPU node by default (PPO updates on the GPU; simulation + MPC on the CPU cores). For a CPU-only run, change
# --partition to standard, drop --gres, and add EXTRA="--set training.device=cpu --set training.torch_threads=4".
#
#SBATCH --job-name=handoff-train
#SBATCH --account=TODO_ACCOUNT          # TODO: your Great Lakes account (e.g. the course/lab account)
#SBATCH --partition=gpu                 # gpu (V100) or spgpu (A40); standard for CPU-only
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20              # training envs + eval workers + main process
#SBATCH --mem-per-cpu=4g
#SBATCH --time=04:00:00                 # estimate ~0.5–1 h per run at 2M control steps; 4 h margin
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
METHOD="${METHOD:?set METHOD=m1|m2|m3}"
REWARD="${REWARD:-r0}"
SCENARIO="${SCENARIO:-freeway}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
module load "${PY_MODULE:-python/3.12}"   # TODO: same module as in setup_greatlakes.sh
source "${VENV:-$REPO_DIR/.venv}/bin/activate"

export SDL_VIDEODRIVER=dummy              # headless pygame (BEV observation)
export OMP_NUM_THREADS=1                  # env subprocesses: 1 thread each (PPO threads set below)
export MKL_NUM_THREADS=1
unset PYTHONPATH

CPUS="${SLURM_CPUS_PER_TASK:-8}"
EVAL_WORKERS=4
N_ENVS=$(( CPUS - EVAL_WORKERS - 2 ))     # leave the main process + a spare core
(( N_ENVS < 1 )) && N_ENVS=1

nvidia-smi || true
echo "method=$METHOD reward=$REWARD scenario=$SCENARIO seed=$SEED n_envs=$N_ENVS eval_workers=$EVAL_WORKERS"

# shellcheck disable=SC2086
python scripts/train.py --method "$METHOD" --reward "$REWARD" --scenario "$SCENARIO" --seed "$SEED" \
  --set training.n_envs="$N_ENVS" --set eval.n_workers="$EVAL_WORKERS" --set training.torch_threads=2 \
  $EXTRA
