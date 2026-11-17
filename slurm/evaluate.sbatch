#!/bin/bash
# Final evaluation of a checkpoint (or B0) on the 20 fixed evaluation seeds. CPU only.
#
#   sbatch --export=ALL,METHOD=m3,CKPT=runs/m3/r0/seed_0/<ts>/checkpoints/best.zip slurm/evaluate.sbatch
#   sbatch --export=ALL,METHOD=b0,SCENARIO=freeway slurm/evaluate.sbatch
#
#SBATCH --job-name=handoff-eval
#SBATCH --account=TODO_ACCOUNT
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
METHOD="${METHOD:?set METHOD}"
SCENARIO="${SCENARIO:-freeway}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PY_MODULE:-python/3.12}"
source "${VENV:-.venv}/bin/activate"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1
unset PYTHONPATH
ARGS=(--method "$METHOD" --scenario "$SCENARIO" --n-workers "${SLURM_CPUS_PER_TASK:-8}")
[[ -n "${CKPT:-}" ]] && ARGS+=(--checkpoint "$CKPT")
[[ -n "${OUT:-}" ]] && ARGS+=(--out "$OUT")
python scripts/evaluate.py "${ARGS[@]}"
