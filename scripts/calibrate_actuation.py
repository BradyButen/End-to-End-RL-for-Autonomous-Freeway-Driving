#!/usr/bin/env python
"""Measure the static throttle_brake -> longitudinal acceleration table (``actuation.table`` in base.yaml) and the
steering response, open loop on the empty map: for each command level the ego starts at --speed, the command is
held for --hold control steps and the mean acceleration is recorded. Prints a YAML snippet.

    python scripts/calibrate_actuation.py --speed 15
"""

import argparse

import numpy as np

from handoff.config import load_config
from handoff.sim import make_metadrive_env

LEVELS = [-1.0, -0.6, -0.3, -0.15, -0.05, 0.0, 0.25, 0.5, 1.0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=15.0, help="start speed (m/s)")
    p.add_argument("--hold", type=int, default=5, help="control steps to hold each command")
    args = p.parse_args()
    cfg = load_config("m1", overrides=["scenario.traffic.density=0", "sim.ego.spawn_lane=1",
                                       f"sim.ego.spawn_speed_km_h={args.speed * 3.6}"])
    dt = float(cfg["sim"]["control_dt"])
    env = make_metadrive_env(cfg)
    table = []
    for u in LEVELS:
        env.reset(seed=0)
        env.step([0.0, 0.1])   # settle
        v0 = env.agent.speed
        for _ in range(args.hold):
            env.step([0.0, u])
        acc = (env.agent.speed - v0) / (args.hold * dt)
        table.append((u, round(float(acc), 3)))
        print(f"throttle_brake {u:+.2f}: accel {acc:+.3f} m/s^2")
    env.reset(seed=0)
    for _ in range(3):
        env.step([0.0, 0.12])
    psi = [env.agent.heading_theta]
    for _ in range(8):
        env.step([0.1, 0.12])
        psi.append(env.agent.heading_theta)
    print("yaw-rate response to a 0.1 steering step (rad/s, per 0.1 s):", np.round(np.diff(psi) / dt, 3))
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
