#!/usr/bin/env bash
# Run every scenario in configs/m3_harness/ through the M3 harness, one after another.
# Each run starts when the previous one exits (window closed / X / horizon end).
#
#   scripts/run_all_m3.sh                          # all *.json in configs/m3_harness
#   scripts/run_all_m3.sh --no-render              # extra args go to every `m3_harness.py run`
#   SCENARIO_DIR=some/dir scripts/run_all_m3.sh    # different folder
set -u
cd "$(dirname "$0")/.."

unset PYTHONPATH                                   # ROS Jazzy's PYTHONPATH breaks the venv
PY=${PYTHON:-.venv/bin/python}
DIR=${SCENARIO_DIR:-configs/m3_harness}

shopt -s nullglob
scenarios=("$DIR"/*.json)
[ ${#scenarios[@]} -eq 0 ] && { echo "no *.json in $DIR"; exit 1; }

failed=()
for i in "${!scenarios[@]}"; do
    sc=${scenarios[$i]}
    echo "=== [$((i + 1))/${#scenarios[@]}] $sc ==="
    "$PY" scripts/m3_harness.py run "$sc" "$@" || { echo "!!! $sc exited with $?"; failed+=("$sc"); }
done

echo "=== done: $(( ${#scenarios[@]} - ${#failed[@]} ))/${#scenarios[@]} ok ==="
for f in "${failed[@]}"; do echo "  failed: $f"; done
[ ${#failed[@]} -eq 0 ]
