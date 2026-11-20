#!/usr/bin/env bash
# Laptop training trial: M1, then M2, then M3, each trained for a fixed wall-clock time with the SAME pipeline
# and settings as the Great Lakes runs (configs/*, r3_fast, eval, logging, checkpoints) — only on CPU and stopped
# by the clock instead of the control-step budget. Then writes learning curves + a summary table.
#
#   scripts/laptop_trial.sh                          # 10 min per method, 14 envs + 4 eval workers (as on the cluster)
#   MINUTES=5 scripts/laptop_trial.sh                # shorter
#   METHODS="m3" scripts/laptop_trial.sh             # one method
#   scripts/laptop_trial.sh --set ppo.learning_rate=1e-4   # extra overrides go to every run
#
# Each method also runs the step-0 eval (before training) and a final eval (after the time limit; adds a few
# minutes, longest for M3). Output: runs/laptop_trial/<timestamp>/{m1,m2,m3}/ (same layout as a cluster run:
# config.yaml, episodes.csv, train_progress.csv, episodes/*.npz, checkpoints/) + report/ (PNGs, summary).
set -u
cd "$(dirname "$0")/.."

unset PYTHONPATH                                   # ROS Jazzy's PYTHONPATH breaks the venv
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PY=${PYTHON:-.venv/bin/python}
MINUTES=${MINUTES:-10}
N_ENVS=${N_ENVS:-14}
EVAL_WORKERS=${EVAL_WORKERS:-4}
SEED=${SEED:-0}
OUT=${OUT:-runs/laptop_trial/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
echo "=== $(date +%H:%M:%S) trial start: ${METHODS:-m1 m2 m3}, ${MINUTES} min each (+ step-0 and final evals) ==="

for m in ${METHODS:-m1 m2 m3}; do
    echo "=== $(date +%H:%M:%S) $m: ${MINUTES} min, ${N_ENVS} envs, ${EVAL_WORKERS} eval workers -> $OUT/$m ==="
    "$PY" scripts/train.py --method "$m" --seed "$SEED" --run-dir "$OUT/$m" \
        --set training.device=cpu --set training.torch_threads=4 \
        --set training.n_envs="$N_ENVS" --set eval.n_workers="$EVAL_WORKERS" \
        --set training.max_wall_minutes="$MINUTES" --set training.total_control_steps=1000000000 \
        "$@" 2>&1 | tee "$OUT/$m.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && echo "!!! $m exited with $rc (see $OUT/$m.log; FAILED.txt in the run dir if it crashed)"
done

echo "=== $(date +%H:%M:%S) all runs done; writing report ==="
"$PY" scripts/trial_report.py "$OUT"
