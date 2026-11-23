#!/bin/bash
# Final evaluation of a checkpoint (or B0) on the fixed evaluation seeds. CPU only.
#
#   sbatch --export=ALL,CKPT=runs/m3/r5/seed_0/<ts>/checkpoints/best.zip slurm/evaluate.sbatch
#   sbatch --export=ALL,METHOD=b0 slurm/evaluate.sbatch
#
#SBATCH --job-name=handoff-eval
#SBATCH --account=rob535_navarch565_f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL             # email when the job ends or fails (to the submitting uniqname@umich.edu)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PY_MODULE:-python/3.12.1}"
source "${VENV:-.venv}/bin/activate"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1
unset PYTHONPATH
ARGS=(--n-workers "${SLURM_CPUS_PER_TASK:-8}")
if [[ -n "${CKPT:-}" ]]; then ARGS+=(--checkpoint "$CKPT"); else ARGS+=(--method "${METHOD:?set CKPT or METHOD=b0}"); fi
[[ -n "${OUT:-}" ]] && ARGS+=(--out "$OUT")
python scripts/evaluate.py "${ARGS[@]}"
