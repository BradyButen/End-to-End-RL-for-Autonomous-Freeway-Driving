# RL training: what the runs taught (2026-11-21 to 11-23)

Learning notes from round 3 (Great Lakes) and the laptop trials. Each item: what we saw, the cause, the fix, and
the general lesson. Related: `docs/m2-reference-timing.md` (M2 label timing), `docs/m3-mpc-lessons.md` (M3
controller), `docs/warm-start.md` (demos, cloning, warm start).

## 1. Read the eval logs before trusting "best"

Round 3: M1 and M2 both got *worse* with training. M2's `best.zip` was the **untrained** policy (step-0 eval:
100 % success, return −291); by 2.2M steps every eval episode went off-road. M1's best round was 800k (48 %); from
1M on, every eval episode was a 60 km/h collision. Lesson: plot every eval round, not just the final number; a
"best" checkpoint at step 0 means training made things worse.

## 2. PPO update size: watch approx_kl and clip fraction

M2's approx_kl rose 0.006 → 0.55 and clip fraction to 0.66 (healthy: < ~0.02 and 0.1–0.2) — each update moved the
policy far from the one that collected the data, the next data was worse, and it spiralled. M2 had collapsed the
same way in round 2. **Fix: the update guard** — `target_kl 0.03` (stop an update's epochs early once KL exceeds
it), `n_epochs 4`, M2 `learning_rate 1e-4`. M1 already had it and stayed at KL ~0.01. Lesson: when a policy gets
worse with more training, check update size before blaming the reward.

## 3. Noise on vs. off: the policy you train is not the policy you evaluate

Training samples actions (mean + noise); eval uses the mean (`eval.deterministic: true`). M1 late in training: noisy
actions braked 10 % of steps → 38 km/h, ~50 % finishes; the mean action **never** braked → 62 km/h, every episode a
rear-end. The noise was doing the braking. Lesson: a large gap between training success and deterministic eval
success means the behaviour lives in the noise, not the policy.

## 4. Action-space asymmetry turns zero-mean noise into a bias

M1 sent its action straight to MetaDrive's `throttle_brake`, which is ~5× stronger on the brake side (+0.5 →
+1.5 m/s², −0.5 → ~−10 m/s²). Zero-mean noise (std 0.5) therefore averaged **−3.4 m/s²**: the untrained car crawled
at 2–9 km/h for the first ~0.8M steps. **Fix:** M1's action is a target acceleration (0 = hold speed, ±range via
the measured actuation table, `method.accel_range`); the same noise now averages −0.2 m/s². Lesson: make the action's
zero the "do nothing" behaviour and keep the two sides of the range comparable, or exploration is biased before
learning starts.

## 5. The reward's ordering decides the local optimum

`r4_straight` ranks: stand still (−3150) < crash (−2000) < crawl to timeout (−1670) < slow finish < fast finish.
A policy that cannot steer yet cannot finish, so among what it *can* reach, crawling beats crashing — "slow down and
survive" is the correct first lesson, and both M1 (laptop) and M3 (round 3, ~25 km/h) learned it. M3 additionally
paid ~−1250 per episode in the speed term for it. Lesson: before training, list what an untrained policy can reach
and how the reward ranks those outcomes — that is where it will go first. (Those are the 2026-11-21 weights; the
2026-11-22 weights rank stand still −9450 < crawl to timeout −5350 < crash −4000 < slow finish < fast finish, §10.)

## 6. Memory that grows with episodes: MetaDrive's map cache

Memory grew ~10 MB per episode reset (laptop warm-start trial: 38 GB in 28 min, OOM; M2 killed at 80 GB in round 2;
M3 at 110 GB after 1.8M steps). Our recorder cleared itself every episode; the growth was MetaDrive's `store_map`
(default true), which keeps every visited seed's map in each process — up to 1040 seeds × ~10 MB × 14 workers.
**Fix:** `sim.store_map: false` (rebuild the seed's map on reset; same map, so replays and evals are unchanged).
Lesson: memory that scales with *episodes* rather than *steps* points at per-reset state; fast-crashing runs hit it
first.

## 7. Warm starts: what a behaviour-cloned start buys

* **M1** (end-to-end, 17 keyboard drives, ~14 min): the clone alone finishes 0 % (never brakes; drifts off in edge
  lanes — classic covariate shift). Warm-started PPO with low noise (std 0.135) and a critic-only warm-up never fell
  into crawling, tripled its distance in 500k laptop steps and produced the first **positive** M1 returns (+79).
* **M3** (waypoints + MPC): the clone finishes **100 %** of its test drives — the MPC handles braking and avoidance,
  so the clone only has to copy the route and speed.
* **M2** (d\*, v\* + PID): the clone finishes ~10 % and gets worse with training (faster, crashes sooner); its test
  drives were identical across three different label sets — it learned "stay in lane, go faster" and ignores
  traffic. Untrained M2 (100 % at 40 km/h) starts better.

Lesson: cloning works when the controller under the policy supplies the safety-critical reflexes (M3); when the
policy itself must learn when to brake or change lanes from a few thousand camera frames (M1, M2), the clone mostly
provides speed and road-following, and corrective data (DAgger) is the next step.

Great Lakes batch (`docs/TRAINING.md` §5) confirmed it: warm M2 went 22 % → 100 % (from scratch it collapsed), warm M3
held 100 %, warm M1 reached 25 % but never brakes. And a new lesson: a warm start with low noise converges to the
*nearest safe* behaviour — M2 and M3 both settled on following 40 km/h traffic and never discovered passing.

## 8. Labels must be in the controller's frame

Two cases where the human labels were "correct" but meant something different to the controller:
* **M2:** d\* = where the human is 2 s ahead, with a 2 s quintic replanned every 0.5 s, ran 0.7 s *early* and
  sideswiped cars the human had waited for. Zero offset needs a lookahead ≈ 0.65 × the horizon → 1.3 s
  (`docs/m2-reference-timing.md`).
* **M3:** v\* is both the speed target and a hard ceiling. Mean speed over the ~8 s waypoint stretch slowed M3
  175 m early; next-second speed capped it so it could never catch up; the human's max over the next 3 s matched.

And one where a better-fitting representation was rejected: M2 with a 1.5 s horizon fit the human drives best
(0.09 m) but failed the step-response check (one-lane step ≈ 9 m/s², at grip). A representation has to fit the data
**and** survive the actions the learner will try.

## 9. Penalise outcomes, not requests

M3's `infeasible_plan` ended the episode at −2000 when a plan asked for more lateral acceleration than the limit.
Exploration noise (0.8 m per waypoint) made lane changes cross that line ~3 % of episodes, so every sideways try was
a lottery ticket for a crash penalty, and the warm-started M3 learned to follow traffic instead of passing. Removed
2026-11-22: the plan is flown (the MPC enforces the limit) and counted. Lesson: a terminal penalty on what the agent
*asks for* is a penalty on exploring near the limit; let the controller enforce limits and let the reward judge what
actually happens (`docs/m3-mpc-lessons.md` §8).

## 10. Reward weights: ratios decide, not magnitudes

For "drive as fast as possible without crashing" (2026-11-22), re-scoring the logged episodes under candidate weights
(each episode's raw term sums are in `episodes.csv`, so any weight set can be re-scored without training) showed:
* **Scaling every term together changes nothing.** Reward normalisation divides out a common scale, and the decision
  that matters — try a pass or stay behind? — depends on gain / (gain + loss): gain = fast finish − stuck finish, loss
  = stuck finish − crash. That ratio is the crash risk at which passing starts to pay. Old weights: 16 %. Speed ×3 and
  crash ×3 together: still ~17 %.
* **The crash penalty is the speed-vs-safety dial.** Speed terms ×3 with crash kept at −2000 → 46 % (accepts
  near coin-flip passes); crash −4000 → ~26 % (chosen); −6000 → 18 %.
* **A big success bonus barely reaches the policy.** The discount looks ~10 s ahead (`gamma_ref` 0.99 per 0.1 s);
  a bonus at the end of a 45–90 s episode is ~1 % of its value at the start. Per-step terms (speed, time) are what
  it feels.
* **Raising the per-second cost of slowness makes crashing relatively cheap** for a policy that cannot drive yet:
  crawling (−5350) and standing still (−9450) now score below a crash (−4000). Fine for M2 / M3 (they never stand
  still); watch early M1. Fix if needed: charge a crash the speed penalty it skips, rather than a bigger constant.

## 11. The discount decides what "worse" means

§10 ranks outcomes by episode totals, where crawling (−5350) is worse than crashing (−4000). PPO does not optimise
totals: with `gamma_ref` 0.99 per 0.1 s it looks ~10 s ahead, and over 10 s crawling at 4 km/h costs ~−570 while a
crash costs −4000. A warm-started M1 that crashed on ~80 % of its noisy episodes learned to crawl at 4 km/h within
250k steps (laptop trial, 2026-11-23): driving only beats crawling if the crash risk over the next ~10 s is below
~14 %. Lesson: rank outcomes by their *discounted* value over the horizon, not by episode totals. A longer horizon
(0.998, ~50 s) needs a time-left feature in the observation, since the value then depends on how much episode is left.

## 12. Shape the per-step reward so neither escape pays

Two escapes kept appearing, and each fix for one opened the other (laptop M1, 2026-11-23):
* **Crawling** wins when slow driving costs little: r4 with a ~10 s discount; a positive (v / v_max)² alone, which is
  flat near 0 (4 km/h earned 0.003 per step, so going a little faster earned nothing and the policy stayed slow).
* **Quitting** wins when every step costs: a time cost plus a big quadratic speed penalty with a −100 crash. Once
  the policy could not finish, each second alive was a loss, and it learned to drive off the road within ~5 s.
The shape that closes both: per-step reward **positive for normal driving** (progress + a big quadratic speed
reward), **negative only at crawl speeds** (a smaller quadratic shortfall), net +4.5 /s at 80 km/h, 0 at ~28 km/h,
−2 /s stopped, with a crash (−200) worse than any finish but better than crawling to the timeout (~−265). Lesson:
check the sign of the per-step reward along the behaviours the policy can fall into, not only the episode totals.

## 13. Exploration noise has physical units

M1 from scratch (laptop, 2026-11-23): deterministic evals improved quickly (0 → 35 % success at 100k steps, ~40 km/h)
while the *training* episodes left the road within 20–40 m and then crawled. PPO's noise is white: std 0.5 on the
steering action = ±20° of wheel, resampled every 0.1 s; the noisy policy only survives slowly. Spawned at 80 km/h it
was worse: lane keeping needs ±1–2° there, so nearly the whole raw steering range (±40°) means "leave the road".
Tried: a per-dimension initial std (steering 0.05, accel 0.5) and steering as a target lateral acceleration
(±6 m/s², wheel angle from the current speed). Lesson: size exploration noise per action dimension from what it does
to the plant at the control rate, and make an action's range mean the same physical thing at every speed (§4).

Batch 6 (2026-11-23) brought back, in a single code path: M1's lateral-acceleration steering and per-dimension
noise (§13), a ground-truth observation (lane-structured, `DESIGN.md` §3), and lane commands for M2 / M3 (discrete,
M3's MPC owns speed), and then (after laptop M1 trials under r4 crawled and crashed in turn) `r5` (progress + quadratic speed reward, a smaller quadratic shortfall, crash −200),
gamma 0.998 + `t_left` and a smooth lane term. Not brought back: the IDM gap keeper, the curriculum, 80 km/h spawns. The 11-23 experiments are on the local branch `archive/2026-11-23-session`.

## 14. Operations (Great Lakes)

* The class account runs **2 jobs at a time** (`AssocMaxJobsLimit`), in submission order — list the jobs you care
  about first.
* Pending jobs load the repo's code **when they start**, not when submitted: don't `git pull` changes you don't want
  them to run.
* A job's time limit comes from the `train.sbatch` it was submitted with — the launcher now pins `--time=08:00:00`.
* `training.max_wall_minutes` must leave time for the final eval (M3: up to ~40 min).
* Copy runs down (`rsync`) before cleaning up; the `ckpt_*` / `vecnormalize_*` files are what make runs ~1 GB.
