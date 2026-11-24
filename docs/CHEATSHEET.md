# Cheat sheet: pull, watch, re-create, evaluate

Copy-paste commands for working with finished runs. Run them from the repo root on the laptop, in a shell with
`source .venv/bin/activate && unset PYTHONPATH`. `<uniqname>` = your Great Lakes login.

Current runs (batch 6, reward r5), as pulled to the laptop:

| model | run folder (laptop) | on Great Lakes |
|---|---|---|
| M1 seed 0 (warm) | `runs/gl_batch6/warm/m1/r5/seed_0/20261123-154324` | `runs/warm/m1/r5/seed_0/20261123-154324` |
| M1 seed 1 (warm) | `runs/gl_batch6/warm/m1/r5/seed_1/20261123-154324` | `runs/warm/m1/r5/seed_1/20261123-154324` |
| M2 | `runs/gl_batch6/m2/r5/seed_0/20261123-231538` | `runs/m2/r5/seed_0/20261123-231538` |
| M3 | `runs/gl_batch6/m3/r5/seed_0/20261123-231525` | `runs/m3/r5/seed_0/20261123-231525` |

Each run folder: `config.yaml`, `checkpoints/best.zip` (+ `best.txt`: which eval round), `final.zip`,
`episodes.csv` (every training and eval episode), `train_progress.csv`.

## 1. Pull from Great Lakes

```bash
GL=<uniqname>@greatlakes.arc-ts.umich.edu
# training runs: configs, CSVs, best/final checkpoints only (skips the ~300 MB of periodic checkpoints)
for src in warm/m1/r5 m2/r5 m3/r5; do
  mkdir -p runs/gl_batch6/$src
  rsync -av --prune-empty-dirs --include='*/' --include='config.yaml' --include='*.csv' --include='git_commit.txt' \
    --include='best.txt' --include='best.zip' --include='final.zip' --exclude='*' $GL:handoff/runs/$src/ runs/gl_batch6/$src/
done
# robustness v2 results (everything: the CSVs are the data)
rsync -av $GL:handoff/runs/robustness/v2/ runs/robustness/v2/
# job logs
rsync -av "$GL:handoff/slurm-handoff-*.out" runs/gl_batch6/
```

On Great Lakes, job status: `squeue -u $USER -o "%.10i %.20j %.8T %.10M %.10L %R"`; latest log lines:
`tail -5 $(ls -t slurm-handoff-*.out | head -1)`.

## 2. Watch a policy drive (3D viewer)

```bash
python scripts/view.py live --checkpoint <run>/checkpoints/best.zip --seeds 1003 1027
python scripts/view.py live --checkpoint <run>/checkpoints/best.zip --seeds 1034 --clean --paused   # poster shots
python scripts/view.py live --method b0 --seeds 1003                                                 # rule-based B0
python scripts/view.py live --checkpoint <run> --seeds 1003 --no-render                              # headless report
```

Any `--set key=value` changes the world, e.g. `--set scenario.traffic.speed_km_h=30`. Every live episode also writes
`runs/view/<method>_<time>/episodes/<id>.npz` + an overview plot (see §6).

Viewer keys: **Space / P** pause, **N** play 0.2 s then pause, **M** all overlays off, **I** stats panel on / off,
**D** move the panel, **1-7** one overlay layer each (1 trail, 2 reference / lane plan, 3 MPC prediction, 4 corridor,
5 keep-out boxes, 6 gate line, 7 balls), **C** screenshot (PNG in the current folder), **B / Q** top-down / chase
camera, mouse wheel zoom, **X** end the episode. Avoid **R**, **[** and **]** (MetaDrive resets behind the env's back).

## 3. Re-create any episode from a CSV row

Simulation is deterministic: checkpoint + episode `seed` + world settings reproduce an episode exactly. For a row of
a robustness `all_episodes.csv` (§5), this prints the command that replays it in the viewer:

```bash
python - <<'EOF'
import pandas as pd
e = pd.read_csv("runs/robustness/v2/all_episodes.csv", low_memory=False)
r = e[(e.method == "m2") & (e.termination == "collision")].sort_values("crash_closing_speed").iloc[-1]  # <- your pick
c = r.checkpoint                                   # laptop copy of a Great Lakes path (unchanged if already local)
c = c if "gl_batch6" in c else "runs/gl_batch6/" + c.split("runs/", 1)[1]
ckpt = "--method b0" if c == "b0" or r.checkpoint == "b0" else f"--checkpoint {c}"
offsets = "[" + str(r.lane_offsets_km_h).replace("_", ", ") + "]"
print(f"python scripts/view.py live {ckpt} --seeds {r.seed} --set scenario.traffic.density={r.density} "
      f"--set scenario.traffic.speed_km_h={r.traffic_speed_km_h} "
      f"--set 'scenario.traffic.lane_speed_offset_km_h={offsets}' --set sim.ego.spawn_speed_km_h={r.spawn_speed_km_h} "
      f"--set sim.timeout_steps=3000")
EOF
```

`checkpoint` holds the path on the machine that ran the suite (Great Lakes: `.../runs/m3/...`); the script maps it
to the laptop copy under `runs/gl_batch6/`. The robustness evaluations use `best.zip`. For a training-eval row
(`<run>/episodes.csv`, `is_eval == 1`) only `--seeds <seed>` is needed (nominal world).

## 4. Evaluate and compare

```bash
python scripts/evaluate.py --checkpoint <run>/checkpoints/best.zip --n-workers 8 --out runs/eval/<name>   # 40 eval seeds
python scripts/evaluate.py --method b0 --out runs/eval/b0
python scripts/compare.py runs/eval/b0 runs/gl_batch6/{m2,m3,warm/m1}/r5/seed_*/*                  # table, last eval round
python scripts/compare.py <run> --tag $(cut -d' ' -f1 <run>/checkpoints/best.txt)                  # at the best round
python scripts/plot.py curves runs/gl_batch6/{m2,m3,warm/m1}/r5/seed_*/* --metric success --out success.png
python scripts/plot.py curves runs/gl_batch6/{m2,m3,warm/m1}/r5/seed_*/* --metric mean_speed --out speed.png
```

## 5. Robustness suites

```bash
# v2 (configs/robust_v2.yaml): 54 conditions x 5 policies x 100 episodes. Great Lakes, one job per policy:
for P in runs/m3/r5/seed_0/20261123-231525 runs/m2/r5/seed_0/20261123-231538 \
         runs/warm/m1/r5/seed_0/20261123-154324 runs/warm/m1/r5/seed_1/20261123-154324 b0; do
  sbatch --export=ALL,POLICY=$P slurm/robustness.sbatch
done
# a job that hit the 8 h limit: submit it again (finished evaluations are skipped)

# laptop
R="runs/gl_batch6/m2/r5/seed_0/20261123-231538 runs/gl_batch6/m3/r5/seed_0/20261123-231525 \
   runs/gl_batch6/warm/m1/r5/seed_0/20261123-154324 runs/gl_batch6/warm/m1/r5/seed_1/20261123-154324"
python scripts/robustness.py --config configs/robust_v2.yaml --b0 --runs $R --n-workers 16 --out runs/robustness/v2
python scripts/robustness.py --config configs/robust_v2.yaml --b0 --runs $R --dry-run         # list the evaluations
python scripts/robustness.py --config configs/robust_v2.yaml --b0 --sweeps traffic_speed --n-episodes 1 \
  --out /tmp/rv2_smoke                                                                         # 1-episode smoke test
python scripts/robustness.py --report runs/robustness/v2          # rebuild report.md, CSVs and plots from results
```

Outputs in `runs/robustness/v2/`:

| file | contents |
|---|---|
| `all_episodes.csv` | **every episode**, one row each: test settings + all metrics (below) |
| `results.csv` | one row per sweep value x policy (averages over its 100 episodes) |
| `grid_results.csv` | one row per density x speed cell x policy |
| `report.md` | tables per sweep / grid, mean ± std over M1's two seeds |
| `<sweep>.png`, `density_x_speed_<metric>.png` | line plots per sweep, heatmaps per method |
| `<condition>/<policy>/` | the raw evaluation: `episodes.csv`, `summary.json`, `eval.log` |

### `all_episodes.csv` columns

Speeds are **m/s** (× 3.6 = km/h) unless the name says km/h; times in s; distances in m; `frac_*` are 0-1 shares of
the episode's control steps.

| group | columns |
|---|---|
| test | `condition` (non-nominal settings, or `nominal`), `policy`, `method`, `run_seed`, `density`, `traffic_speed_km_h`, `lane_offsets_km_h` (left_middle_right), `spawn_speed_km_h`, `checkpoint`, `seed` (episode seed: map + traffic) |
| outcome | `termination`, `success`, `collision`, `emergency_brake`, `off_road`, `timeout`, `sim_time`, `time_to_complete`, `distance`, `route_length`, `route_completion` |
| crash | `crash_type` (rear_end = ego hit the car ahead, hit_from_behind, side = lane-change swipe, off_road, other), `crash_ego_speed`, `crash_other_speed`, `crash_closing_speed` (how fast the gap was closing) |
| speed | `mean_speed`, `min_speed`, `max_speed`, `p10_speed`, `frac_time_below_10kmh` (crawling), `frac_time_above_70kmh` |
| traffic | `n_cars` (traffic cars placed), `cars_passed`, `passed_by`, `lane_changes`, `frac_time_lane0/1/2` (0 = leftmost, fastest offset) |
| safety | `min_gap`, `min_ttc`, `frac_time_ttc_below_3s` |
| comfort | `rms_long_jerk`, `rms_lat_accel`, `rms_steer_rate`, `rms_steer_rate_rad` |
| M2 / M3 RL commands | `n_speed_up`, `n_speed_down` (+10 / -10 commands), `mean_target_speed`, `max_target_speed`, `final_target_speed` |
| controllers | `tracking_error_rms/max`, `solve_time_mean/p95/max` (M3 MPC), `fallback_count`, `gated_brake_count`, `infeasible_count` |
| reward | `return`, `return_<term>` (weighted sum), `raw_<term>` (unweighted) per reward term |

## 6. One episode in detail (needs its .npz: view.py live writes one per seed)

```bash
python scripts/plot.py episode runs/view/<method>_<time>/episodes/<id>.npz   # overview.png + per-step steps.csv
python scripts/plot.py reward  runs/view/<method>_<time>/episodes/<id>.npz   # how the return built up
```

## 7. Training (Great Lakes)

```bash
JOBS="m3 m2" bash slurm/launch_batch.sh            # from scratch; SEEDS="0 1", DRY=1 prints only
JOBS=m1_warm SEEDS="0 1" bash slurm/launch_batch.sh # M1 warm start from data/bc/m1_clone.zip
grep '\[eval.*success' $(ls -t slurm-handoff-*.out | head -1) | tail -4     # latest evals of the newest job
cat runs/*/r5/seed_*/*/checkpoints/best.txt runs/warm/*/r5/seed_*/*/checkpoints/best.txt
```
