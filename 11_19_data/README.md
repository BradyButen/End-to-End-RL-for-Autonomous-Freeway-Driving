# 11_19_data — pilot training round 1 (2026-11-18/19)

Snapshot of the first Great Lakes training round: one seed per method, reward **r0**, scenario **freeway**,
2M control steps each. Results and what changed for round 2: [../docs/TRAINING.md](../docs/TRAINING.md) §5.
Code version: commit `0c4d225` plus uncommitted edits on the cluster (`git_commit.txt` says `-dirty`; run
`git status`/`git diff` in `~/handoff` on Great Lakes to see what they were). Run settings beyond the config
files: `n_envs=14`, `eval.n_workers=4`, `eval.freq_control_steps=100000`, `torch_threads=2`.

## Layout
Same layout as `runs/`, so every repo command works by pointing it here instead.
```
11_19_data/
  m1/r0/seed_0/20261118-193700/    M1 end-to-end   — job 66397238, gpu (V100), 2.45 h
  m2/r0/seed_0/20261118-220746/    M2 RL + PID     — job 66397239, standard (CPU), 2.24 h
  m3/r0/seed_0/20261118-193700/    M3 RL + MPC     — job 66397237, standard (CPU), 4.40 h
  slurm_logs/<method>_job<id>.out  full stdout of each job (PPO tables, [eval] lines)
  plots/                           learning curves (all_<metric>.png = M1/M2/M3 overlaid)
```
Each run folder:
| file | contents |
|---|---|
| `config.yaml` | the fully merged config the run used |
| `git_commit.txt`, `device.txt` | code version; cpu/cuda + torch version |
| `train_progress.csv` | every PPO table (train/*, eval/*, counters/*, time/*) |
| `episodes.csv` | one row per episode — training episodes and every eval episode (`eval_tag` column) |
| `checkpoints/best.zip`, `best.txt` | best eval checkpoint and which eval it came from |
| `checkpoints/final.zip` | policy at the end of training |
| `episodes/*.npz` | per-step logs of logged episodes (**M1 only** — see below) |

Not included: the 20 intermediate checkpoints per run (`ckpt_*.zip`, ~13 MB each) and M2/M3 episode
`.npz` logs. They are still on Great Lakes under `~/handoff/runs/` (until scratch/home cleanup). To add the
M2/M3 episode logs here:
```bash
rsync -avz brbuten@greatlakes.arc-ts.umich.edu:handoff/runs/m2/r0/seed_0/20261118-220746/episodes/ 11_19_data/m2/r0/seed_0/20261118-220746/episodes/
rsync -avz brbuten@greatlakes.arc-ts.umich.edu:handoff/runs/m3/r0/seed_0/20261118-193700/episodes/ 11_19_data/m3/r0/seed_0/20261118-193700/episodes/
```

## Analyzing from a clean clone
```bash
git clone https://github.com/BradyButen/learning-driving-control.git && cd learning-driving-control
python3 -m venv .venv && source .venv/bin/activate && unset PYTHONPATH
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"
python -m metadrive.pull_asset                                    # only needed for the viewer

# learning curves (any column of the eval rows: success, collision, off_road, return, mean_speed, ...)
python scripts/plot.py curves 11_19_data/m*/r0/seed_0/* --metric success --x control_steps --out success.png

# eval results as a table (pandas)
python -c "import pandas as pd, glob
for f in sorted(glob.glob('11_19_data/m*/r0/seed_0/*/episodes.csv')):
    d = pd.read_csv(f, low_memory=False); e = d[d.eval_tag.notna()]
    print(f.split('/')[1], e.groupby('eval_tag')[['success','collision','off_road','mean_speed','return']].mean().tail(3), sep='\n')"

# watch a policy (3D, needs a display; overlays: reference, MPC path, corridor)
python scripts/view.py live --method m2 --checkpoint 11_19_data/m2/r0/seed_0/20261118-220746/checkpoints/best.zip --seeds 1017
python scripts/view.py live --method m3 --checkpoint 11_19_data/m3/r0/seed_0/20261118-193700/checkpoints/final.zip --seeds 1017
```
Eval seeds are 1000–1019; evaluation is deterministic, so re-running a checkpoint on a seed reproduces the
logged eval episode.

## Headline numbers (final eval, 20 episodes)
| | success | collision | off-road | mean speed |
|---|---|---|---|---|
| M1 | 0% | 85% | 5% | 13.5 m/s |
| M2 | 65% | 35% | 0% | 10.8 m/s |
| M3 | 100% | 0% | 0% | 10.0 m/s |
