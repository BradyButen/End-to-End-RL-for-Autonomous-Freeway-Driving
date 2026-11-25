#!/usr/bin/env python
"""Drive the RL's freeway yourself and see what the reward thinks of it (needs a display: WSLg / X11).

Same env the RL trains in (map, 35-45 km/h traffic, ego car, episode cap, termination rules, reward, logging); only
the ego's controls come from you. Real time (10 Hz).

    python scripts/warm_start/drive.py                          # eval seed 1000
    python scripts/warm_start/drive.py --seeds 1003 1027        # one episode per seed, back to back
    python scripts/warm_start/drive.py --reward my_reward       # drive under configs/my_reward.yaml instead of r5
    python scripts/warm_start/drive.py --controller xbox        # gamepad (or wheel); falls back to the keyboard

Keys: W throttle, S brake, A / D steer (click the window first). X = end the episode, M = text / trail on / off.
Avoid T (MetaDrive's expert takeover), R (reset) and Esc (quits without saving).

The text panel shows the return so far and each term's share, updated every 0.1 s. After each episode the
return per reward term is printed and the full log is saved to runs/drive/<timestamp>/episodes/<id>.npz. To use a
drive for the warm start, copy its .npz into data/human_demos/ and rebuild the clone (docs/GUIDE.md).
"""

import argparse
import os
import time
from pathlib import Path

from handoff.config import DEFAULT_REWARD


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[1000], help="MetaDrive seeds (map + traffic)")
    p.add_argument("--reward", default=DEFAULT_REWARD, help="configs/<name>.yaml")
    p.add_argument("--controller", default="keyboard", choices=["keyboard", "xbox", "wheel"])
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="default runs/drive/<timestamp>")
    args = p.parse_args()
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ.pop("SDL_VIDEODRIVER")

    from metadrive.policy.manual_control_policy import ManualControlPolicy

    from handoff.config import load_config
    from handoff.env import make_env
    from handoff.reward import TERMS
    from handoff.viz import Viewer, drive_episode

    # M1's env (one control step per decision), with MetaDrive's keyboard / gamepad policy driving the ego
    cfg = load_config("m1", reward=args.reward, overrides=args.overrides)
    cfg["experiment"]["method"] = "manual"
    cfg["sim"]["metadrive_overrides"] = {**(cfg["sim"].get("metadrive_overrides") or {}), "manual_control": True,
                                         "controller": args.controller, "agent_policy": ManualControlPolicy}
    out = Path(args.out or f"runs/drive/{time.strftime('%Y%m%d-%H%M%S')}")
    env = make_env(cfg, render=True, is_eval=True, run_dir=out)
    viewer = Viewer(fps=10.0, show_return=True)

    def idle(_obs):          # the action is ignored: ManualControlPolicy reads the controller
        return [0.0, 0.0]

    try:
        for seed in args.seeds:
            print(f"seed {seed}: drive! (W/S/A/D, X to end)")
            summ = drive_episode(env, idle, seed, "drive", viewer)
            print(f"seed {seed}: {summ['termination']} after {summ['sim_time']:.1f} s, "
                  f"mean {summ['mean_speed'] * 3.6:.1f} km/h, {summ['lane_changes']} lane changes")
            for n in TERMS:
                if summ[f"return_{n}"]:
                    print(f"  {n:>12s}: {summ[f'return_{n}']:+9.2f}")
            print(f"  {'return':>12s}: {summ['return']:+9.2f}\n  log: {summ['log_file']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
