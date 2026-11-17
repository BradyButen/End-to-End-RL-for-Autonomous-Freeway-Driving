#!/usr/bin/env python
"""MetaDrive viewer with overlays (spec §13). Needs a display (WSLg / X11); training always runs headless.

Live:    python scripts/view.py live --method m3 --checkpoint .../best.zip --seeds 1000 1001
         python scripts/view.py live --method b0
Replay:  python scripts/view.py replay --episode runs/.../episodes/<id>.npz --checkpoint .../best.zip
Plots:   see scripts/plot.py
Add --topdown to use MetaDrive's 2D top-down renderer instead of the 3D window (text overlay only).
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    lv = sub.add_parser("live")
    rp = sub.add_parser("replay")
    for sp in (lv, rp):
        sp.add_argument("--checkpoint", default=None)
        sp.add_argument("--topdown", action="store_true")
        sp.add_argument("--set", dest="overrides", action="append", default=[])
    lv.add_argument("--method", required=True, choices=["m1", "m2", "m3", "b0"])
    lv.add_argument("--reward", default="r0")
    lv.add_argument("--scenario", default="freeway")
    lv.add_argument("--seeds", nargs="+", type=int, default=None)
    rp.add_argument("--episode", required=True, help="logged episode .npz")
    rp.add_argument("--config", default=None, help="run config.yaml (default: inferred from the episode path)")
    args = p.parse_args()

    if args.topdown and os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ.pop("SDL_VIDEODRIVER")
    if args.topdown:
        os.environ["SDL_VIDEODRIVER"] = os.environ.get("SDL_VIDEODRIVER_VIEW", "x11")

    from pathlib import Path

    import yaml

    from handoff.config import deep_merge, load_config, parse_override
    from handoff.simulator.viewer import live, replay

    if args.mode == "live":
        cfg = load_config(args.method, args.reward, args.scenario, args.overrides)
        if args.checkpoint:
            run_cfg = Path(args.checkpoint).parent.parent / "config.yaml"
            if run_cfg.exists():
                cfg = yaml.safe_load(run_cfg.read_text())
        seeds = args.seeds or cfg["scenario"]["seeds"]["eval"][:3]
        summaries = live(cfg, args.checkpoint, seeds, topdown=args.topdown)
        for s in summaries:
            print(f"seed {s['seed']}: {s['termination']}  t={s['sim_time']:.1f}s  v={s['mean_speed'] * 3.6:.1f} km/h")
    else:
        ep = Path(args.episode)
        cfg_path = Path(args.config) if args.config else ep.parent.parent / "config.yaml"
        if not cfg_path.exists():
            sys.exit(f"config not found: {cfg_path} (pass --config)")
        cfg = yaml.safe_load(cfg_path.read_text())
        for ov in args.overrides:
            cfg = deep_merge(cfg, parse_override(ov))
        if cfg["method"]["name"] != "b0" and not args.checkpoint:
            sys.exit("--checkpoint is required to re-simulate an RL episode")
        res = replay(cfg, str(ep), args.checkpoint, topdown=args.topdown)
        print(res)


if __name__ == "__main__":
    main()
