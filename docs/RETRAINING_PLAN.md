# Retraining plan — after pilot round 1 (2026-11-19)

Status: **proposed, not yet adopted.** Reward weights and PPO settings below are starting points to decide on.

## Pilot round 1 (r0, seed 0, 2M control steps, Great Lakes)

| | M1 end-to-end | M2 RL+PID | M3 RL+MPC | B0 rule-based |
|---|---|---|---|---|
| success | 0% | 65% (still rising) | 100% (flat from step 0) | 85% |
| collision | 85% | 35% | 0% | 0% |
| mean speed | 13.5 m/s | 10.8 m/s | 10.0 m/s | 17.7 m/s |
| RMS long. jerk | 18.1 | 30.0 | 0.86 | — |
| PPO updates | ~70 | ~14 | ~7 | — |
| wall time | 2.45 h (GPU) | 2.24 h (CPU) | 4.40 h (CPU) | — |

Takeaways:
* **M1** did not learn; PPO updates too large (approx_kl ≈ 0.17, clip_fraction 0.4–0.5); collisions rose over training.
* **M2** learned (15% → 65%) by slowing down; was still improving at the end of the budget.
* **M3** succeeds only because of the controller stack — its policy barely changed (7 updates). It follows
  30 km/h traffic and never passes, so the run says nothing yet about what RL adds at this level.

## 1. Reward (main problem)
Under r0 a completed ~1 km run earns ~+1000 from progress, success only +10, and a crash only −5 (= 5 m of
progress). Crashing late is almost as good as finishing, and nothing rewards driving at the target speed.

Proposal: keep r0 as the MetaDrive-default baseline; add **`r3_safe`**:

| term | r0 | r3_safe (proposed) |
|---|---|---|
| progress | 1.0 /m | 1.0 /m |
| collision | −5 | −300 |
| off_road | −5 | −300 |
| success | +10 | +200 |
| speed_tracking | 0 | 0.3 |
| jerk / steer_rate | 0 | small (as in r2), optional |

Sanity check before training: score a few logged episodes under r3 and confirm crash < slow-safe < fast-safe.

## 2. PPO settings — equalise updates, calm M1
| | change (`--set`) | effect |
|---|---|---|
| M1 | `ppo.n_epochs=4 ppo.batch_size=512 ppo.target_kl=0.03` | smaller, stabler updates |
| M2 | `ppo.n_steps=256 ppo.batch_size=256` | ~14 → ~110 updates |
| M3 | `ppo.n_steps=128 ppo.batch_size=256` | ~7 → ~110 updates |

## 3. Budget
* M1, M2: 4M control steps (~5 h). M3: 3M (~6.6 h) — the 8 h class-account walltime cap and no resume
  limit it; 4M for M3 needs checkpoint resume.
* Evaluate every 200k control steps instead of 100k.

## 4. Fix before the 3-seed sweep
1. A crashed training job must exit (it currently hangs and keeps its slot + billing).
2. Checkpoint resume (needed for budgets > ~3.5M on M3).
3. `setup_greatlakes.sh`: default to the cu126 PyTorch wheel (cu128+ dropped V100 support);
   `PYTHONUNBUFFERED=1` in the sbatch scripts.
4. `lane_changes` metric is stuck at 0.80 for M2/M3 — verify before using it for "does M3 pass?".
5. Commit the viewer overlay fixes (lines now visible; current corridor box only).
6. Check peak memory (`sacct -j <id> --format=MaxRSS`); lower `--mem-per-cpu` if far below 80 GB.

## 5. Next round
One seed-0 pilot per method with r3_safe + the PPO changes (~5–7 h wall time with 2 job slots). Compare with
round 1 on success, collision, speed, lane changes. When all three learn, launch 3 methods × 3 seeds.

```bash
COMMON='PYTHONUNBUFFERED=1,REWARD=r3_safe,SEED=0'
EV='--set eval.freq_control_steps=200000'
sbatch --partition=standard --gres=none --time=08:00:00 --export="ALL,METHOD=m3,$COMMON,EXTRA=$EV --set training.total_control_steps=3000000 --set ppo.n_steps=128 --set ppo.batch_size=256" slurm/train.sbatch
sbatch --partition=gpu --time=08:00:00 --export="ALL,METHOD=m1,$COMMON,EXTRA=$EV --set training.total_control_steps=4000000 --set ppo.n_epochs=4 --set ppo.batch_size=512 --set ppo.target_kl=0.03" slurm/train.sbatch
sbatch --partition=standard --gres=none --time=08:00:00 --export="ALL,METHOD=m2,$COMMON,EXTRA=$EV --set training.total_control_steps=4000000 --set ppo.n_steps=256 --set ppo.batch_size=256" slurm/train.sbatch
```
(`r3_safe` does not exist yet — create `configs/reward/r3_safe.yaml` first.)
