#!/usr/bin/env bash
# Submit one training batch on Great Lakes: each method from scratch (m1, m2, m3) and / or warm-started from its
# behaviour-cloning clone of the human drives (m1_warm, m2_warm, m3_warm: data/bc/<method>_clone.zip,
# docs/human-data-pipeline.md), for each seed. Budgets / wall-clock stops are in configs/<method>.yaml.
#
#   bash slurm/launch_batch.sh                         # seed 0, all 6 jobs
#   JOBS="m1_warm m2_warm m3_warm" bash slurm/launch_batch.sh   # warm starts only
#   SEEDS="0 1 2" JOBS="m3_warm" bash slurm/launch_batch.sh
#   DRY=1 bash slurm/launch_batch.sh                   # print the sbatch commands only
#
# The account runs at most 2 jobs at a time (AssocMaxJobsLimit): jobs start in submission order, so list the ones
# you want first. Output: runs/<method>/r4_straight/seed_<n>/<ts>/ ; warm: runs/warm/<method>/r4_straight/...
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-0}
JOBS=${JOBS:-m1_warm m2_warm m3_warm m1 m2 m3}
MEM=${MEM:-7g}                                     # x 20 cores = 140 GB
WARMUP=${WARMUP:-100000}                           # critic-only control steps (~6 PPO updates for every method)
run() { if [[ -n "${DRY:-}" ]]; then echo "$*"; else "$@"; fi; }

submit() {   # method seed extra-args job-name
  local m=$1 s=$2 extra=$3 name=$4 part=(--partition=standard --gres=none)
  [[ $m == m1 ]] && part=(--partition=gpu)          # M1's CNN updates are big (18k samples x 4 epochs)
  run sbatch --time=08:00:00 "${part[@]}" --mem-per-cpu="$MEM" --job-name="$name" \
    --export="ALL,METHOD=$m,SEED=$s${extra:+,EXTRA=$extra}" slurm/train.sbatch
}

for S in $SEEDS; do
  for J in $JOBS; do
    m=${J%_warm}
    [[ $m =~ ^m[123]$ ]] || { echo "unknown job $J (m1 m2 m3 m1_warm m2_warm m3_warm)"; exit 1; }
    if [[ $J == *_warm ]]; then
      clone=data/bc/${m}_clone.zip
      [[ -f $clone ]] || { echo "missing $clone (build it: docs/human-data-pipeline.md; it is tracked in git)"; exit 1; }
      submit "$m" "$S" "--set training.init_policy=$clone --set training.init_log_std=-2.0 \
--set training.critic_warmup_control_steps=$WARMUP --set experiment.runs_root=runs/warm" "handoff-warm-$m"
    else
      submit "$m" "$S" "" "handoff-$m"
    fi
  done
done
[[ -n "${DRY:-}" ]] || squeue -u "$USER"
