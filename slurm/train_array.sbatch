#!/bin/bash
# Job array over methods × seeds for one reward variant (default 3 methods × 3 seeds = 9 tasks):
#
#   sbatch --export=ALL,REWARD=r4_straight,SCENARIO=freeway slurm/train_array.sbatch
#   sbatch --array=0-8%3 --export=ALL,REWARD=r4_straight slurm/train_array.sbatch   # at most 3 at a time
#
# Task i -> METHOD = METHODS[i / n_seeds], SEED = SEEDS[i % n_seeds]. Resources as in train.sbatch.
#
#SBATCH --job-name=handoff-array
#SBATCH --account=rob535_navarch565_f26_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4g
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=END,FAIL             # email when the job ends or fails (to the submitting uniqname@umich.edu)

set -euo pipefail
METHODS=(${METHODS:-m1 m2 m3})
SEEDS=(${SEEDS:-0 1 2})
i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
export METHOD="${METHODS[$(( i / ${#SEEDS[@]} ))]}"
export SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"
export REWARD="${REWARD:-r4_straight}"
export SCENARIO="${SCENARIO:-freeway}"
exec bash "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/train.sbatch"
