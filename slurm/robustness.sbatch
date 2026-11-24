#!/bin/bash
# Robustness suite for ONE policy (configs/robust_v2.yaml by default). One job per policy, all writing to the same
# OUT; finished evaluations are skipped, so a resubmitted job resumes. Submit from the repo root:
#
#   for P in runs/m3/r5/seed_0/<ts> runs/m2/r5/seed_0/<ts> runs/warm/m1/r5/seed_0/<ts> runs/warm/m1/r5/seed_1/<ts> b0; do
#     sbatch --export=ALL,POLICY=$P slurm/robustness.sbatch
#   done
#   python scripts/robustness.py --report runs/robustness/v2      # after the last job (each job also writes it)
#
#SBATCH --job-name=handoff-robust
#SBATCH --account=rob535_navarch565_f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4g
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PY_MODULE:-python/3.12.1}"
source "${VENV:-.venv}/bin/activate"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
unset PYTHONPATH

POLICY="${POLICY:?set POLICY=<run dir> or b0}"
CONFIG="${CONFIG:-configs/robust_v2.yaml}"
OUT="${OUT:-runs/robustness/v2}"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))
if [[ $POLICY == b0 ]]; then WHO=(--b0); else WHO=(--runs "$POLICY"); fi
echo "policy=$POLICY config=$CONFIG out=$OUT workers=$WORKERS"
python scripts/robustness.py --config "$CONFIG" --out "$OUT" --n-workers "$WORKERS" "${WHO[@]}"
