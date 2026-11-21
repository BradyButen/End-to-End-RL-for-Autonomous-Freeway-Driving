#!/usr/bin/env bash
# Laptop trial: M1 PPO warm-started from the behaviour-cloning clone (docs/human-data-pipeline.md), for a fixed
# wall-clock time, with laptop-sized envs / evals. Output: runs/warm_trial/m1/r4_straight/seed_<n>/<timestamp>/.
#
#   scripts/laptop_warm_trial.sh                 # 60 min, data/bc/m1_clone.zip
#   REBUILD=1 scripts/laptop_warm_trial.sh       # first rebuild demos (accel-action labels) + retrain the clone (~5 min)
#   MINUTES=30 SEED=1 scripts/laptop_warm_trial.sh --set training.init_log_std=-1.5   # extra --set go to train.py
set -euo pipefail
cd "$(dirname "$0")/.."

unset PYTHONPATH                                   # ROS PYTHONPATH breaks the venv
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PY=${PYTHON:-.venv/bin/python}
MINUTES=${MINUTES:-60}
SEED=${SEED:-0}
CLONE=${CLONE:-data/bc/m1_clone.zip}
WARMUP=${WARMUP:-60000}                            # critic-only steps (~4 PPO updates at 12 envs)
LOG=runs/warm_trial_$(date +%Y%m%d-%H%M%S).log
mkdir -p runs

if [[ "${REBUILD:-0}" == 1 ]]; then
  $PY scripts/build_demos.py                       # data/human_demos/*.npz -> runs/demos/m1.npz
  $PY scripts/bc.py --n-workers 4 --out runs/bc/m1/laptop_trial
  CLONE=runs/bc/m1/laptop_trial/checkpoints/best.zip
fi

$PY scripts/train.py --method m1 --seed "$SEED" \
  --set experiment.runs_root=runs/warm_trial \
  --set training.init_policy="$CLONE" \
  --set training.init_log_std=-2.0 \
  --set training.critic_warmup_control_steps="$WARMUP" \
  --set training.max_wall_minutes="$MINUTES" \
  --set training.total_control_steps=10000000 \
  --set training.n_envs=12 \
  --set training.torch_threads=6 \
  --set eval.n_workers=3 \
  --set eval.n_episodes=20 \
  --set eval.freq_control_steps=100000 \
  "$@" 2>&1 | tee "$LOG"
echo "log: $LOG"
