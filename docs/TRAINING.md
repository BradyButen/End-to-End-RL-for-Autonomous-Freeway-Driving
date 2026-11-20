# Training

RL setup, why each setting is what it is, what the pilot taught us, and how to run on Great Lakes.
System design: [DESIGN.md](DESIGN.md).

## 1. Algorithm
PPO (Stable-Baselines3, `MultiInputPolicy`: NatureCNN on the BEV + MLP on the ego vector, shared by actor and
critic), identical for M1/M2/M3. Budgets and cadences are counted in **control steps** (0.1 s of simulated driving),
summed over parallel envs, so every method gets the same simulated experience (D-07).

## 2. Reward: `r3_fast` (the only reward)
Per control step, summed within a decision; terminal terms added on the last step.

| term | weight | why |
|---|---|---|
| progress (Δs, m) | +0.1 /m | dense signal: tells an untrained policy which way is good (~+100 for a full route) |
| time | −0.1 /step (−1 /s) | arriving sooner is directly better; net per step 0.01·v − 0.1 > 0 only above 10 m/s |
| success | +50 | finishing beats every non-finish |
| collision | −500 | also charged for an M3 `emergency_brake` episode end |
| off_road | −500 | |
| lane_change | −2 per lane | stops weaving; a pass (out and back, −4) pays when it saves > 4 s |

Expected episode scores: fast finish at 22 m/s ≈ +105 > finish behind 35–45 km/h traffic ≈ +55 to +65 > timeout ≈ −20 to −50
> crash ≈ −400 to −500. A crash never pays: the whole 150 s time cost is −150 ≪ −500. Re-scoring the pilot's logged
eval episodes confirmed the ordering (crashes −495, timeouts −78, slow finishes +40 to +48).
r0 (MetaDrive's default, used in the pilot) and the untuned example variants were removed: under r0 a finished run
earned ~+1040 and a crash only −5, so crashing late was almost as good as finishing.

## 3. PPO settings and why
Base (`configs/base.yaml`) unless a method overrides it (`configs/method/m*.yaml`).

| setting | value | why |
|---|---|---|
| γ | 0.99 per 0.1 s → M1 0.99, M2 0.951, M3 0.904 per decision | same ~10 s lookahead in seconds for every method |
| GAE λ | 0.995 per 0.1 s → M1 0.995, M2 0.975, M3 0.951 per decision | **new.** A fixed λ = 0.95 per *decision* gave M1 a ~2 s credit window and M3 ~20 s. Scaled like γ, all methods get the same ~7 s window (1/(1−γλ) in seconds). |
| rollout `n_steps` | M1 1280, M2 256, M3 128 per env | **new for M1.** = 128 s of driving per env per rollout for every method, so updates per simulated hour are equal (with 14 envs: one update per ~30 min of driving). Pilot: M1 70, M2 14, M3 7 updates in 2M steps. |
| `batch_size` / `n_epochs` | M1 512 / 4; M2, M3 256 / 10 | M1's pilot updates were too large (approx_kl rose 0.01 → 0.23, clip fraction 0.35–0.49, value loss 0.1 → 22): fewer epochs, bigger batches, plus `target_kl 0.03` (stops an update early when the policy moves too far). M2/M3 rollouts are smaller (1.8k–3.6k decisions), so the default 10 epochs are fine. |
| `log_std_init` | −0.7 (initial std 0.5) | **new.** SB3's default std 1 on a [−1, 1] action box clips ~32 % of samples to the limits. Pilot: M3's std stayed 0.98 for the whole run (near-random rollouts: 50 % training collisions vs 0 % in deterministic eval); M2's first third was 100 % off-road within 8 s; M1 steered randomly at speed. Std 0.5 clips ~5 %. |
| `ent_coef` | 0 | exploration is set by `log_std_init`; an entropy bonus would push std back up |
| reward normalisation | on (`VecNormalize`, rewards only), clip ±100 | **new.** Returns span −500…+105; normalising by the running return std keeps the value loss from swamping the policy loss (pilot explained variance ≈ 0.5). Clip raised from SB3's 10 so the crash penalty is not clipped early in training. Logs and eval stay raw. |
| `learning_rate`, `clip_range` | 3e-4, 0.2 | SB3 defaults; no evidence yet to change |
| eval | 40 seeds (1000–1039), deterministic, every 200k control steps | 2.5 % resolution on rates (20 seeds = 5 %); best checkpoint = highest success, ties by return |
| episode cap | 150 s | longest route stuck behind the slowest lane (35 km/h) takes 118 s; 120 s left no margin |

## 4. Run robustness
* A crashed env or eval worker (e.g. a native segfault) ends the job: `FAILED.txt` (traceback),
  `checkpoints/crash.zip`, all child processes killed, non-zero exit.
* Watchdog: no PPO step or eval for 30 min (`training.watchdog_minutes`) → kill workers, exit code 3 (hung worker).
* `train.sbatch` sets `PYTHONUNBUFFERED=1` (live log) and `PYTHONFAULTHANDLER=1` (traceback on a segfault).
* `checkpoints/vecnormalize_*.pkl` is saved with every checkpoint (needed only to resume; resume is not implemented).

## 5. Pilot round 1 (2026-11-18/19, `11_19_data/`, reward r0, 2M control steps, seed 0)
| | M1 | M2 | M3 |
|---|---|---|---|
| final eval success / collision | 0 % / 85 % | 65 % / 35 % | 100 % / 0 % |
| mean speed | 13.5 m/s | 10.8 m/s | 10.0 m/s |
| PPO updates | ~70 | ~14 | ~7 |

What it showed (and what changed because of it):
* **Reward** made crashing late nearly as good as finishing → `r3_fast`.
* **Too few updates** for M2/M3 → equal 128 s rollouts. M2 only started working after ~1.2M steps and was still
  improving.
* **Exploration too wide** → `log_std_init −0.7`.
* **M3 never passed**: lane changes were exactly 1 on off-centre spawns (it moved to the middle lane and stayed),
  0 otherwise; its policy barely changed (std 0.98) and it followed 30 km/h traffic. M3 also could not choose its
  speed → v\* channel (target and ceiling).
* **M3 off-road in ~30 % of training episodes** → road corridor with fixed lateral faces.
* **Randomised ego dynamics**: the pilot trained on MetaDrive's `default` vehicle (engine/brake resampled every
  episode), the controllers were tuned on `static_default` → fixed.
* No episode crashed in its first 3 s (spawns are clean).

## 6. Known risks for round 2
* **Emergency brake = crash is M3-only** — M3's crash rate counts predicted crashes; report `emergency_brake`
  next to `collision`.
* **M3 decides once per second**: a car cutting in mid-decision is the corridor/MPC's problem; an infeasible
  problem ends the episode.
* **No time-left feature** in the observation (value estimates slightly off near the 150 s cap).
* **20 → 40 eval seeds** doubles eval cost; M3 eval is the slowest.

## 7. Great Lakes workflow
One-time setup (login node, needs internet):
```bash
git clone https://github.com/BradyButen/learning-driving-control.git handoff && cd handoff
PY_MODULE=python/3.12.1 bash slurm/setup_greatlakes.sh            # CUDA PyTorch (gpu partitions)
PY_MODULE=python/3.12.1 CPU_ONLY=1 bash slurm/setup_greatlakes.sh # CPU-only PyTorch (standard partition)
```
Measure throughput before setting budgets (M3 is heavier than in the pilot: horizon 50, corridor every step):
```bash
srun --account=rob535_navarch565_f26_class --partition=standard --cpus-per-task=20 --time=00:30:00 --pty bash
source .venv/bin/activate && export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 && unset PYTHONPATH
python tests/benchmark_throughput.py --n-envs 14 --decisions 300
```
Budget for a 4–5 h run ≈ control steps/s × 4.5 h × 3600 × 0.8 (eval + PPO updates).

Launch (seed 0; `r3_fast` is the default reward):
```bash
EV='--set training.total_control_steps'
sbatch --partition=standard --gres=none --time=05:30:00 --export="ALL,METHOD=m3,SEED=0,EXTRA=$EV=<M3 budget>" slurm/train.sbatch
sbatch --partition=gpu       --time=05:30:00 --export="ALL,METHOD=m1,SEED=0,EXTRA=$EV=<M1 budget>" slurm/train.sbatch
sbatch --partition=standard --gres=none --time=05:30:00 --export="ALL,METHOD=m2,SEED=0,EXTRA=$EV=<M2 budget>" slurm/train.sbatch
sbatch --array=0-8%3 slurm/train_array.sbatch     # later: 3 methods x 3 seeds
```
`train.sbatch`: 20 cores → 14 training envs + 4 eval workers + main; `OMP_NUM_THREADS=1`. Each run writes
`runs/<method>/r3_fast/seed_<n>/<timestamp>/`. Check `sacct -j <id> --format=MaxRSS` and lower `--mem-per-cpu`
if far below 80 GB.

Analysis:
```bash
python scripts/plot.py curves runs/m*/r3_fast/seed_*/* --metric success --x control_steps --out curves.png
python scripts/evaluate.py --method m3 --checkpoint <run>/checkpoints/best.zip
python scripts/view.py live --method m3 --checkpoint <run>/checkpoints/best.zip --seeds 1017   # needs a display
python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip
```
Always `unset PYTHONPATH` (ROS) and `export SDL_VIDEODRIVER=dummy` on headless nodes.
