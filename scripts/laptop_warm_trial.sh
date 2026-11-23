#!/usr/bin/env bash
# Laptop trial (docs/warm-start.md). M1 (default) is warm-started from the human-demo clone:
#   1. build the dataset from data/human_demos/ (scripts/build_demos.py)                         [if missing / REBUILD=1]
#   2. behaviour-clone it (scripts/bc.py) and copy the clone to data/bc/m1_clone.zip             [if missing / REBUILD=1]
#   3. PPO warm-started from the clone for MINUTES of wall clock, laptop-sized envs / evals     [skip: CLONE_ONLY=1]
# METHOD=m2 / m3 (or WARM=0 for M1): PPO from scratch with the same laptop sizes, no clone.
# Output: runs/demos/m1.npz, runs/bc/m1/laptop/, data/bc/m1_clone.zip (commit it for Great Lakes),
#         runs/warm_trial/<m>/r5/seed_<n>/<timestamp>/ + runs/warm_trial_<m>_<ts>.log
#
#   CLONE_ONLY=1 scripts/laptop_warm_trial.sh               # just build data/bc/m1_clone.zip
#   MINUTES=15 scripts/laptop_warm_trial.sh                 # clone (if missing) + 15 min warm-start trial
#   METHOD=m3 MINUTES=15 scripts/laptop_warm_trial.sh       # 15 min of M3 from scratch
#   scripts/laptop_warm_trial.sh --set training.n_envs=8     # extra --set go to train.py
set -euo pipefail
cd "$(dirname "$0")/.."

unset PYTHONPATH                                   # ROS PYTHONPATH breaks the venv
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PY=${PYTHON:-.venv/bin/python}
METHOD=${METHOD:-m1}
WARM=${WARM:-$([[ $METHOD == m1 ]] && echo 1 || echo 0)}
MINUTES=${MINUTES:-60}
SEED=${SEED:-0}
WARMUP=${WARMUP:-60000}                            # critic-only control steps (~4 PPO updates at 12 envs)
CLONE=data/bc/m1_clone.zip
DATA=runs/demos/m1.npz
mkdir -p runs data/bc

WARM_ARGS=()
if [[ $WARM == 1 ]]; then
  if [[ "${REBUILD:-0}" == 1 || ! -f $CLONE ]]; then
    $PY scripts/build_demos.py --out "$DATA"        # replay the human drives -> $DATA (always: it must match
                                                    # today's observation)
    rm -rf runs/bc/m1/laptop
    $PY scripts/bc.py --data "$DATA" --epochs "${BC_EPOCHS:-40}" --n-workers 4 --out runs/bc/m1/laptop
    cp runs/bc/m1/laptop/checkpoints/best.zip "$CLONE"
    cp runs/bc/m1/laptop/bc_summary.json data/bc/m1_clone_summary.json
    echo "clone: $CLONE (commit it to use it on Great Lakes: JOBS=m1_warm bash slurm/launch_batch.sh)"
  fi
  [[ "${CLONE_ONLY:-0}" == 1 ]] && exit 0
  WARM_ARGS=(--set training.init_policy="$CLONE" --set training.critic_warmup_control_steps="$WARMUP")
fi

LOG=runs/warm_trial_${METHOD}_$(date +%Y%m%d-%H%M%S).log
$PY scripts/train.py --method "$METHOD" --seed "$SEED" \
  --set experiment.runs_root=runs/warm_trial \
  "${WARM_ARGS[@]}" \
  --set training.max_wall_minutes="$MINUTES" \
  --set training.total_control_steps=100000000 \
  --set training.n_envs=12 \
  --set training.torch_threads=6 \
  --set eval.n_workers=3 \
  --set eval.n_episodes=20 \
  --set eval.freq_control_steps=100000 \
  "$@" 2>&1 | tee "$LOG"
echo "log: $LOG"
