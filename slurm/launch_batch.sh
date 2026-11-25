#!/usr/bin/env bash
# Submit one training batch on a Slurm cluster: each method from scratch (m1, m2, m3) and / or M1 warm-started from its
# behaviour-cloning clone of the human drives (m1_warm: data/bc/m1_clone.zip, docs/GUIDE.md), for each seed.
# Budgets / wall-clock stops / initial exploration are in configs/<method>.yaml.
#
#   bash slurm/launch_batch.sh                         # seed 0, all 4 jobs
#   SEEDS="0 1 2" JOBS="m3 m2" bash slurm/launch_batch.sh
#   DRY=1 bash slurm/launch_batch.sh                   # print the sbatch commands only
#
# Set SBATCH_ACCOUNT=<account> first. Jobs start in submission order, so list the ones you want first.
# Output: runs/<method>/r5/seed_<n>/<ts>/ ; warm: runs/warm/<method>/r5/...
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-0}
JOBS=${JOBS:-m3 m2 m1_warm m1}                     # M3 first: the longest jobs and the main question
MEM=${MEM:-7g}                                     # x 20 cores = 140 GB
WARMUP=${WARMUP:-100000}                           # critic-only control steps (~6 PPO updates for every method)
run() { if [[ -n "${DRY:-}" ]]; then echo "$*"; else "$@"; fi; }

submit() {   # method seed extra-args job-name
  local m=$1 s=$2 extra=$3 name=$4 part=(--partition=standard --gres=none)   # MLP policies: no GPU needed
  run sbatch --time=08:00:00 "${part[@]}" --mem-per-cpu="$MEM" --job-name="$name" \
    --export="ALL,METHOD=$m,SEED=$s${extra:+,EXTRA=$extra}" slurm/train.sbatch
}

for S in $SEEDS; do
  for J in $JOBS; do
    m=${J%_warm}
    [[ $J =~ ^(m1|m2|m3|m1_warm)$ ]] || { echo "unknown job $J (m1 m2 m3 m1_warm)"; exit 1; }
    if [[ $J == *_warm ]]; then
      clone=data/bc/${m}_clone.zip
      [[ -f $clone ]] || { echo "missing $clone (build it: docs/GUIDE.md; it is tracked in git)"; exit 1; }
      submit "$m" "$S" "--set training.init_policy=$clone \
--set training.critic_warmup_control_steps=$WARMUP --set experiment.runs_root=runs/warm" "handoff-warm-$m"
    else
      submit "$m" "$S" "" "handoff-$m"
    fi
  done
done
[[ -n "${DRY:-}" ]] || squeue -u "$USER"
