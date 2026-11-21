#!/usr/bin/env bash
# Submit one training batch on Great Lakes: M1 from scratch, M1 warm-started from the behaviour-cloning clone
# (docs/human-data-pipeline.md), M2 and M3, for each seed. Budgets / wall-clock stops are in configs/<method>.yaml.
#
#   bash slurm/launch_batch.sh                    # seed 0: 4 jobs
#   SEEDS="0 1 2" bash slurm/launch_batch.sh      # 12 jobs
#   JOBS="m1_warm m3" bash slurm/launch_batch.sh  # a subset (m1, m1_warm, m2, m3)
#   DRY=1 bash slurm/launch_batch.sh              # print the sbatch commands only
#
# Output: runs/<method>/r4_straight/seed_<n>/<ts>/ ; warm start: runs/warm/m1/r4_straight/seed_<n>/<ts>/.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-0}
JOBS=${JOBS:-m1 m1_warm m2 m3}
MEM=${MEM:-7g}                                     # x 20 cores = 140 GB
CLONE=${CLONE:-data/bc/m1_clone.zip}
WARM="--set training.init_policy=$CLONE --set training.init_log_std=-2.0"
WARM+=" --set training.critic_warmup_control_steps=100000 --set experiment.runs_root=runs/warm"

[[ -f "$CLONE" ]] || { echo "missing $CLONE (git pull: it is tracked in data/bc/)"; exit 1; }
run() { if [[ -n "${DRY:-}" ]]; then echo "$*"; else "$@"; fi; }

for S in $SEEDS; do
  for J in $JOBS; do
    case $J in
      m1)      run sbatch --partition=gpu --mem-per-cpu="$MEM" --export="ALL,METHOD=m1,SEED=$S" slurm/train.sbatch ;;
      m1_warm) run sbatch --partition=gpu --mem-per-cpu="$MEM" --job-name=handoff-warm \
                 --export="ALL,METHOD=m1,SEED=$S,EXTRA=$WARM" slurm/train.sbatch ;;
      m2)      run sbatch --partition=standard --gres=none --mem-per-cpu="$MEM" \
                 --export="ALL,METHOD=m2,SEED=$S" slurm/train.sbatch ;;
      m3)      run sbatch --partition=standard --gres=none --mem-per-cpu="$MEM" \
                 --export="ALL,METHOD=m3,SEED=$S" slurm/train.sbatch ;;
      *) echo "unknown job $J (m1, m1_warm, m2, m3)"; exit 1 ;;
    esac
  done
done
[[ -n "${DRY:-}" ]] || squeue -u "$USER"
