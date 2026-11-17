#!/usr/bin/env python
"""Measure the static (throttle_brake -> longitudinal acceleration) table and the steering response used by
the controllers (spec §19 "actuation mapping"; open-loop, no system identification / fitting).

For each command level the ego is brought to a reference speed on the empty map, the command is held for
0.5 s and the mean acceleration is recorded. Prints a YAML snippet for ``actuation.table``.

    python scripts/calibrate_actuation.py --speed 15
"""

import argparse

import numpy as np

from handoff.config import load_config
from handoff.simulator.metadrive_env import FreewayMetaDriveEnv, make_metadrive_config
from handoff.simulator.scenario import ScenarioSpec, VehicleSpec

LEVELS = [-1.0, -0.6, -0.3, -0.15, -0.05, 0.0, 0.25, 0.5, 1.0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speed", type=float, default=15.0, help="reference speed (m/s)")
    p.add_argument("--hold", type=int, default=5, help="control steps to hold each command")
    args = p.parse_args()
    cfg = load_config("m1")
    env = FreewayMetaDriveEnv(make_metadrive_config(cfg))
    table = []
    for u in LEVELS:
        spec = ScenarioSpec(seed=0, ego=VehicleSpec(lane=1, s=80, speed=args.speed, role="ego"),
                            lead=VehicleSpec(lane=0, s=990, speed=1.0, model="xl", role="lead"))
        env.reset_with_spec(spec)
        a = env.agent
        env.step([0.0, 0.1])  # settle
        v0 = a.speed
        for _ in range(args.hold):
            env.step([0.0, u])
        acc = (a.speed - v0) / (args.hold * cfg["sim"]["control_dt"])
        table.append((u, round(float(acc), 3)))
        print(f"throttle_brake {u:+.2f}: accel {acc:+.3f} m/s^2")
    # steering step response
    spec = ScenarioSpec(seed=0, ego=VehicleSpec(lane=1, s=80, speed=args.speed, role="ego"),
                        lead=VehicleSpec(lane=0, s=990, speed=1.0, model="xl", role="lead"))
    env.reset_with_spec(spec)
    a = env.agent
    for _ in range(3):
        env.step([0.0, 0.12])
    psi = [a.heading_theta]
    for _ in range(8):
        env.step([0.1, 0.12])
        psi.append(a.heading_theta)
    r = np.diff(psi) / cfg["sim"]["control_dt"]
    print("yaw-rate response to a 0.1 steering step (rad/s per 0.1 s):", np.round(r, 3))
    env.close()
    mono = [table[0]]
    for row in table[1:]:
        if row[1] > mono[-1][1]:
            mono.append(row)
    print("\nactuation:\n  table:")
    for u, acc in mono:
        print(f"    - [{u}, {acc}]")


if __name__ == "__main__":
    main()
