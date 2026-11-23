# M2's reference: why the label lookahead has to match the quintic horizon

> **Historical (since 2026-11-23).** M2's action is now a lane command + speed level (DESIGN §4) and M2 is no longer
> warm-started, so no d\* labels are built. The timing lesson (a receding-horizon reference lags its target by
> ~0.3 T) still describes how M2's quintic flies a lane change.

Notes from 2026-11-21. Change: human-demo labels for M2 read the human **1.3 s** ahead instead of 2 s
(`scripts/build_demos.py --m2-ahead-s`, default 1.3), matched to M2's unchanged quintic horizon `trajectory.horizon`
= 2 s. Result: M2 flying the human's own labels goes from 0.60 m rms / 0.7 s early (sideswiping traffic on almost
every drive) to 0.14 m rms with no timing offset. (A 1.5 s horizon with 1 s labels fit even better, 0.09 m, but
failed the step-response check: see section 4.)

## 1. How M2 turns (d\*, v\*) into motion

Every decision (P = 0.5 s) the RL outputs a lateral target d\* and a speed v\*. `m2.build_m2_reference`:

1. **Lateral:** a quintic d(t) over the horizon T, from the car's current state (d, ḋ, d̈) — taken from the previous
   reference so it is continuous across replans — to d(T) = d\*, ḋ(T) = d̈(T) = 0.
2. **Longitudinal:** an acceleration-limited speed profile to v\* (+2 / −4 m/s²).
3. The cascaded PID tracks that reference (measured: ~0.03 m rms cross-track error, so tracking is not the issue).
4. **After P = 0.5 s it replans.** The rest of the quintic is never flown.

So the path M2 actually drives is a **chain of quintic starts**, never a whole quintic. Each quintic is shaped to
*arrive gently* at d\* (zero lateral speed at T), but that gentle arrival is always cut off; what M2 flies is the
steep early part, aimed at a target that keeps moving. This is receding-horizon control, and it means d\* does not
mean "where to be at T" — it means "where to be roughly 0.7 T from now".

## 2. The lead / lag, derived

Take the middle of a lane change, the target path drifting sideways at a constant rate r.

* The label is the human's position L seconds ahead: the target is always **L·r ahead of the human**.
* In steady state the car moves at r too. For the chained quintic to advance r·P per replan while it is built to
  stop at its end, its target must sit some δ ahead of the car. δ depends only on T and P (solve the quintic for it).
* So the car is **L·r − δ ahead of the human**: it matches the human exactly when L = δ / r.

Solving numerically (P = 0.5 s):

| horizon T | zero-offset lookahead L = δ / r | as a fraction of T |
|---|---|---|
| 1.0 s | 0.69 s | 0.69 |
| 1.5 s | **0.97 s** | 0.65 |
| 2.0 s | 1.26 s | 0.63 |

Rule of thumb: **L ≈ 0.65–0.7 T.** The old setting (L = 2 s, T = 2 s) put the target ~0.74 s of drift too far
ahead: M2 ran ~0.7 s early.

```python
# reproduce the table (no simulator)
import numpy as np
from handoff.reference import quintic_coefficients, eval_quintic
P = 0.5
for T in (1.0, 1.5, 2.0):
    f = lambda dl: eval_quintic(quintic_coefficients(0.0, 1.0, 0.0, dl, T), np.array([P]), T)[0][0] - P
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    print(T, round(mid, 2))      # car drifts at 1 m/s: target sits `mid` m (= s) ahead
```

## 3. Checked against the 17 human drives

Offline: chain the quintics exactly as M2 does (replan every 0.5 s, continuity of ḋ, d̈, perfect tracking) on each
recorded human path, and compare with the human's d(t).

| lookahead L | horizon T | rms error | max error | timing vs human |
|---|---|---|---|---|
| 2.0 s | 2.0 s (old) | 0.60 m | 3.0 m | 0.7 s early |
| 1.0 s | 2.0 s | 0.31 m | 1.6 m | 0.3 s late |
| 1.0 s | 1.5 s | 0.09 m | 0.49 m | 0.0 s |
| **1.3 s** | **2.0 s** (used) | **0.14 m** | **0.76 m** | **0.0 s** |
| 0.75 s | 1.0 s | 0.04 m | 0.25 m | 0.0 s |
| 0.5 s | 0.6 s | 0.07 m | 0.54 m | +0.1 s |

Every row where L ≈ 0.7 T has zero timing offset; the error grows as L moves away from it in either direction.

**In the simulator** (`scripts/demo_relabel.py --methods m2`, M2 flying the human's labels on the same map and
traffic): with 2 s / 2 s, 1 of the first 8 drives succeeded. Every collision was a **side impact during a lane
change** — the other car level with M2 (Δs −5…+1 m) one car width over (Δd ≈ 2 m) — with M2 ~1 m further into the
lane change than the human at the same point. Keyboard drivers often wait for a slower car to clear and then cut
across late; M2, a second early, moved over while that car was still alongside. The PID tracked its reference to
≤ 0.18 m throughout: the controller was doing exactly what the labels asked. With 1 s labels it was clearly better (your sweep).

## 4. Why not make T as short as possible

T is also the fastest lateral move the RL can command. A quintic moving Δd in T peaks at **5.77 · Δd / T²** lateral
acceleration (in m/s², for d in m along time). For a full lane change, Δd = 3.5 m:

| T | peak lateral acceleration |
|---|---|
| **2.0 s** | **~5 m/s²** (used) |
| 1.5 s | ~9 m/s² — at MetaDrive's grip (wheel_friction 0.9 → ~8.8) |
| 1.0 s | ~20 m/s² — beyond grip |

Tried first: T = 1.5 s with 1 s labels (best fit, 0.09 m). `validate.py check m2` steps d\* by a whole lane at
60 km/h — exactly what an exploring RL does — and at T = 1.5 s the PID overshot 1.6 m and left the road (tolerance
1.0 m). The human labels never step (they move d\* gradually), so the demo looked fine; the RL would not be. Kept
T = 2 s and moved the label instead: 1.3 s ≈ the zero-offset lookahead for T = 2 (1.26 s), 0.14 m rms. Lesson: a
representation has to fit the data AND survive the actions the learner will try.

## 5. What it means for the RL and the warm start

* M2's lateral action has a physical meaning: **"where I want to be laterally ~1.3 s from now."**
* The human labels are in that same frame, so the clone learns targets the controller can reproduce and PPO
  fine-tunes the same quantity.
* M2's speed channel is a different mechanism (acceleration-limited profile, not a quintic). The 1 s lookahead was
  chosen for the lateral fit; if M2's speed trace drifts from the human's while the lateral path matches, revisit
  v\* separately.
* M2's controller is unchanged (T = 2 s), so from-scratch M2 runs before and after this change are comparable; only
  the M2 clone (labels) changed.

## 6. The same lesson elsewhere

Any receding-horizon pipeline where the policy outputs an *end point* and only the start of the plan is executed has
this property: the executed motion leads or lags the target depending on the horizon and the replan period. M3 has
an analogous choice in how its speed target v\* is labelled (it is both target and hard ceiling there):
averaging the human's speed over the ~8 s waypoint stretch made M3 slow ~175 m early; the next-second speed made it
unable to catch up; the human's highest speed over the next 3 s matched (`docs/warm-start.md`, step 5).

## Files

| what | where |
|---|---|
| M2 reference + PID | `src/handoff/m2.py` (`build_m2_reference`, `CascadedPID`) |
| quintic | `src/handoff/reference.py` (`quintic_coefficients`, `eval_quintic`) |
| horizon | `configs/m2.yaml` `method.trajectory.horizon` |
| label lookahead | `scripts/build_demos.py --m2-ahead-s` (also `scripts/demo_relabel.py`) |
| human-route demo | `scripts/demo_relabel.py --methods m2` |
