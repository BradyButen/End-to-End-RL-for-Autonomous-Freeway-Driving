#!/usr/bin/env bash
# Human-demo warm start on the laptop, for M1, M2 or M3 (docs/human-data-pipeline.md):
#   1. build the method's dataset from data/human_demos/ (scripts/build_demos.py --method <m>)   [if missing / REBUILD=1]
#   2. behaviour-clone it (scripts/bc.py) and copy the clone to data/bc/<m>_clone.zip            [if missing / REBUILD=1]
#   3. PPO warm-started from the clone for MINUTES of wall clock, laptop-sized envs / evals      [skip: CLONE_ONLY=1]
# Output: runs/demos/<m>.npz, runs/bc/<m>/laptop/, data/bc/<m>_clone.zip (commit it for Great Lakes),
#         runs/warm_trial/<m>/r4_straight/seed_<n>/<timestamp>/ + runs/warm_trial_<m>_<ts>.log
#
#   METHOD=m3 CLONE_ONLY=1 scripts/laptop_warm_trial.sh     # just build data/bc/m3_clone.zip
#   METHOD=m3 scripts/laptop_warm_trial.sh                  # clone (if missing) + 60 min warm-start trial
#   METHOD=m2 MINUTES=30 REBUILD=1 scripts/laptop_warm_trial.sh
#   scripts/laptop_warm_trial.sh --set training.init_log_std=-1.5      # extra --set go to train.py
set -euo pipefail
cd "$(dirname "$0")/.."

unset PYTHONPATH                                   # ROS PYTHONPATH breaks the venv
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PY=${PYTHON:-.venv/bin/python}
METHOD=${METHOD:-m1}
MINUTES=${MINUTES:-60}
SEED=${SEED:-0}
WARMUP=${WARMUP:-60000}                            # critic-only control steps (~4 PPO updates at 12 envs)
BC_EPOCHS=${BC_EPOCHS:-$([[ $METHOD == m3 ]] && echo 20 || echo 40)}   # M3's driving tests (MPC) are slow
CLONE=data/bc/${METHOD}_clone.zip
DATA=runs/demos/${METHOD}.npz
mkdir -p runs data/bc

if [[ "${REBUILD:-0}" == 1 || ! -f $CLONE ]]; then
  if [[ "${REBUILD:-0}" == 1 || ! -f $DATA ]]; then
    $PY scripts/build_demos.py --method "$METHOD"   # replay the human drives -> $DATA
  fi
  rm -rf "runs/bc/$METHOD/laptop"
  $PY scripts/bc.py --data "$DATA" --epochs "$BC_EPOCHS" --n-workers 4 --out "runs/bc/$METHOD/laptop"
  cp "runs/bc/$METHOD/laptop/checkpoints/best.zip" "$CLONE"
  cp "runs/bc/$METHOD/laptop/bc_summary.json" "data/bc/${METHOD}_clone_summary.json"
  echo "clone: $CLONE (commit it to use it on Great Lakes: JOBS=${METHOD}_warm bash slurm/launch_batch.sh)"
fi
[[ "${CLONE_ONLY:-0}" == 1 ]] && exit 0

LOG=runs/warm_trial_${METHOD}_$(date +%Y%m%d-%H%M%S).log
$PY scripts/train.py --method "$METHOD" --seed "$SEED" \
  --set experiment.runs_root=runs/warm_trial \
  --set training.init_policy="$CLONE" \
  --set training.init_log_std=-2.0 \
  --set training.critic_warmup_control_steps="$WARMUP" \
  --set training.max_wall_minutes="$MINUTES" \
  --set training.total_control_steps=100000000 \
  --set training.n_envs=12 \
  --set training.torch_threads=6 \
  --set eval.n_workers=3 \
  --set eval.n_episodes=20 \
  --set eval.freq_control_steps=100000 \
  "$@" 2>&1 | tee "$LOG"
echo "log: $LOG"
