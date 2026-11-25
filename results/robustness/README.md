# Robustness results

The final policies (B0, M1 warm-started with seeds 0 and 1, M2, M3), frozen, evaluated under traffic conditions they
were not trained on. Each condition varies traffic density, traffic speed and/or the per-lane speed offsets, with
everything else at the trained (nominal) values. There are 100 episodes per policy and condition, the episode cap is
300 s, and the ego spawns at the average traffic speed.

| file | contents |
|---|---|
| `report.md` | the summary tables per sweep and per density × speed grid |
| `results.csv` | one row per sweep value × policy (means over its 100 episodes) |
| `grid_results.csv` | one row per density × traffic-speed cell × policy |
| `all_episodes.csv` | the raw data: one row per episode, every setting and metric |

## `all_episodes.csv` columns
Speeds are in m/s (× 3.6 = km/h) unless the name says km/h, times are in s and distances in m. `frac_*` columns are
the fraction of the episode's control steps.

| group | columns |
|---|---|
| test | `condition` (its non-nominal settings, or `nominal`), `policy`, `method`, `run_seed`, `density`, `traffic_speed_km_h`, `lane_offsets_km_h` (left_middle_right), `spawn_speed_km_h`, `checkpoint`, `seed` (episode seed: map + traffic) |
| outcome | `termination`, `success`, `collision`, `emergency_brake`, `off_road`, `timeout`, `sim_time`, `time_to_complete`, `distance`, `route_length`, `route_completion` |
| crash | `crash_type` (rear_end = the ego hit the car ahead, hit_from_behind, side = lane-change contact, off_road, other), `crash_ego_speed`, `crash_other_speed`, `crash_closing_speed` |
| speed | `mean_speed`, `min_speed`, `max_speed`, `p10_speed`, `frac_time_below_10kmh`, `frac_time_above_70kmh` |
| traffic | `n_cars` (traffic cars placed), `cars_passed`, `passed_by`, `lane_changes`, `frac_time_lane0/1/2` (0 = leftmost, fastest lane) |
| safety | `min_gap`, `min_ttc`, `frac_time_ttc_below_3s` |
| comfort | `rms_long_jerk`, `rms_lat_accel`, `rms_steer_rate`, `rms_steer_rate_rad` |
| M2 / M3 commands | `n_speed_up`, `n_speed_down` (+10 / −10 km/h commands), `mean_target_speed`, `max_target_speed`, `final_target_speed` |
| controllers | `tracking_error_rms/max`, `solve_time_mean/p95/max` (M3's MPC), `fallback_count`, `gated_brake_count`, `infeasible_count` |
| reward | `return`, then `return_<term>` (weighted) and `raw_<term>` (unweighted) for each reward term |

Every episode can be re-simulated exactly in the viewer from its checkpoint (the matching folder under `models/`),
`seed` and condition settings, for example:
```bash
python scripts/view.py live --checkpoint models/m2 --seeds <seed> --set scenario.traffic.density=<density> \
  --set scenario.traffic.speed_km_h=<traffic_speed_km_h> --set sim.ego.spawn_speed_km_h=<spawn_speed_km_h> \
  --set 'scenario.traffic.lane_speed_offset_km_h=[5, 0, -5]'
```
The `crash_*`, speed-profile, lane-share and passing columns came from analysis code that is no longer in the
repository; they are kept here as recorded. `emergency_brake`, `gated_brake_count` and `infeasible_count` belong to
an M3 emergency-brake gate that was off for every run (since removed); they are always 0.
