# M3 controller: what broke, why, and what it taught (2026-11-21)

Learning notes from reworking M3's corridor + MPC (`src/handoff/m3.py`, `src/handoff/mpc.py`, `configs/m3.yaml`).
The test that drove most of it: fly a human drive's route through M3 (`scripts/demo_relabel.py --methods m3`) and
compare with the human. Final result on seed 428: human 52.1 s / 68.9 km/h, M3 51.9 s / 69.2 km/h, same 4 lane
changes. Before: 81.1 s / 44.3 km/h.

## 1. Hard constraints make "no solution" a normal event

The corridor (one road-aligned box per MPC stage) was a **hard** constraint. Hard state constraints in MPC are
infeasible whenever the current state cannot reach the set in time, and that happens in ordinary driving:

* the car is already 0.5 m outside a lateral face (faces sat at the outer lane centres ± 0.3 m: legal driving);
* a car cuts in closer than stopping distance.

IPOPT then returns "infeasible", there is no plan at all, and the old fallback was an emergency brake that ended the
episode at −2000. So the RL was punished for situations it did not create.

**Fix: soft constraints with an exact penalty.** Each half-space gets a slack σ ≥ 0: `A p ≤ b + σ`, cost
`l1·Σσ + l2·Σσ²` (`mpc.soft_corridor`, 1000 / 1000). Theory: an L1 penalty larger than the constraint's Lagrange
multipliers is *exact* — whenever the hard problem is feasible, the soft one has the same solution (σ = 0). When it
is not, you get the least-violating plan instead of nothing. The L2 term keeps it smooth near σ = 0. Offline: the car
starting 0.48 m outside a face went from "infeasible" to a plan that steers back in; the plan's largest slack is
logged as `corridor_violation` (a useful "predicted intrusion" signal).

## 2. The initial guess decides whether you get an answer in time

With the soft corridor a stopped car 25 m ahead at 72 km/h no longer failed as infeasible — it timed out. The first
solve after a car becomes unavoidable needed **~244 IPOPT iterations (346 ms)** from the previous plan (cruising),
against a 0.1 s CPU budget. Once one solve succeeds, warm-started solves take ~18 iterations.

**Fix: on a failed solve, retry once from a full-braking guess** along the planned path (`MPC._brake_guess`):
34 iterations, 49 ms. Normal cases are unaffected (12–31 ms with either guess). Lesson: NLP solvers are local; a
guess on the wrong side of a "brake vs. not" decision costs an order of magnitude in iterations. When the
qualitative answer changes, give the solver a guess with the new qualitative shape.

## 3. Don't replay a stale plan for long

On failure the MPC replays its last plan. With a 49-step budget, the `blocked` scenario (cars across all lanes)
replayed a bad swerving plan for 31 steps — re-accelerating — and left the road. Now 10 steps (1 s), then hold speed
and lane until a solve succeeds (`mpc.replay_steps`). A replayed plan is open loop; its error grows with every step.

## 4. A safety layer that pre-empts the controller can cause the behaviour it guards against

The "gate" emergency-braked (full brake, −15.8 m/s², episode over) whenever stopping for a car in the path needed
more than 6 m/s², skipping the MPC — which can brake to −8 m/s². It ended runs the MPC could have saved, and every
firing cost −2000. The cheapest way for the RL to avoid it was to drive slowly, and M3 had learned to crawl at
~25 km/h. Gate off (`ebrake.gate: false`); braking for cars is the MPC's job.

## 5. Corridor geometry must be checked in time, not just in space

**Bug 1 (the big slowdown).** A car counted as "in path" if the planned spline crossed where the car *will be*,
without asking whether the ego would be there at the same time; a car ahead *now* then got a front face at *every*
stage. So M3 could never overtake a slower car and merge in front of it — exactly what the human did. In the demo it
braked from 78 to 40 km/h with no car in its own lane, following a 40 km/h car in the next lane; the logged corridor
showed that car's front face 14 m ahead mid-horizon and 4–22 m *behind* the stage points at the end.

**Fix: decide each car's face stage by stage** from where the stage point and the car's predicted zone are at that
stage, with a state carried along the horizon: behind it and overlapping laterally → front face; level beside it →
lateral face on that side; past it → back face.

**Bug 2 (introduced by the fix, caught by `validate.py check scenarios`).** "Behind → passing" was allowed as soon as
the stage point was beyond the car and laterally clear — including when the plan got there by driving *through* it.
In `blocked` (all lanes closed at 199 m, waypoints returning to the middle lane beyond the wall) the front face
vanished for the last stages, the MPC planned to be past the wall, the solves timed out, and the replayed plan left
the road. **Fix:** "behind → passing" only at a stage that is level with the car *and* beside it; a plan that has run
into a car keeps the front face for the rest of the horizon.

Lessons: (1) a constraint on a moving obstacle is a space-*time* object; (2) when you rewrite a geometric rule, the
regression scenarios (blocked road, overtake, weave) are what catch the new corner case — run them every time.

## 6. Limits shape behaviour: the lateral-acceleration bound

`mpc.max_lat_accel` bounds the MPC (|v² tan δ / L|) and the waypoint feasibility check. At 4 m/s² (comfort) M3
could not reproduce human lane changes (keyboard swerves peaked at 9.3 m/s²) and half the human plans failed the
check. Raised to 9.3 (MetaDrive's grip is ~8.8 m/s² at wheel_friction 0.9; the kinematic model cannot see skidding,
so expect some tracking loss near the top). In the human-route demo M3 peaked at 5.2 m/s².

## 7. Still open: the spline goes flat at every waypoint

`spline_reference` passes every waypoint with zero slope and curvature, so a lateral move spread over several
waypoints becomes a staircase of S-curves, and the feasibility check (5.77·|Δd|·v²/Δs² per segment) inherits it.
Consequence: putting waypoints *closer* makes plans *less* feasible (allowed |Δd| ∝ Δs²). A spline with continuous
slope through the waypoints, checked on its actual curvature (v²·d″), would let dense waypoints follow a path
closely. Deferred.

## Validation that covers these

```bash
python scripts/validate.py check m3          # pass_lead, stopped_car, infeasible_plan, failed_solve
python scripts/validate.py check scenarios   # blocked, overtake, weaving, ... (configs/validation/*.json)
python scripts/demo_relabel.py --methods m3  # human route through M3 (3D; white balls = waypoints)
```
