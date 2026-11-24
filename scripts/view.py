#!/usr/bin/env python
"""Watch a policy drive in the MetaDrive 3D window with everything the RL and the controller are doing drawn on
top (needs a display: WSLg / X11). Overlay legend and all keys: src/handoff/viz.py. Keys: Space pause, N step 0.2 s,
M all overlays, I / D hide / move the text panel, 1-7 one overlay layer each, C screenshot, X end episode.

Live (an episode report per seed: npz + overview.png + steps.csv in --out):
    python scripts/view.py live --checkpoint runs/gl_round2/m3 --seeds 1003 1027   # run folder: final.zip, else best
    python scripts/view.py live --checkpoint <run>/checkpoints/best.zip --fps 0     # as fast as it renders
    python scripts/view.py live --method b0
    python scripts/view.py live --checkpoint <run> --no-render                      # headless: reports only
    python scripts/view.py live --checkpoint <run> --clean                          # no dashboard/marks, panel hidden
    python scripts/view.py live --checkpoint <run> --clean --paused --seeds 1003     # poster shots: starts paused
Replay (re-simulation from the seed + checkpoint, checked against the log):
    python scripts/view.py replay --episode <run>/episodes/<id>.npz --checkpoint <run>/checkpoints/best.zip
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    lv, rp = sub.add_parser("live"), sub.add_parser("replay")
    for sp in (lv, rp):
        sp.add_argument("--checkpoint", default=None, help="SB3 .zip or run folder (omit for b0)")
        sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    lv.add_argument("--method", default=None, choices=["b0"], help="b0 (no checkpoint)")
    lv.add_argument("--seeds", nargs="+", type=int, default=None, help="default: first 3 eval seeds")
    lv.add_argument("--fps", type=float, default=10.0, help="10 = real time, 0 = as fast as possible")
    lv.add_argument("--stochastic", action="store_true", help="sample actions (training behaviour) instead of mean")
    lv.add_argument("--no-render", action="store_true")
    lv.add_argument("--clean", action="store_true",
                    help="no MetaDrive marks / dashboard / FPS; text panel starts hidden (I shows it)")
    lv.add_argument("--paused", action="store_true", help="start paused (Space resumes, N plays 0.2 s)")
    lv.add_argument("--out", default=None, help="default runs/view/<method>_<timestamp>")
    rp.add_argument("--episode", required=True, help="logged episode .npz")
    args = p.parse_args()
    if os.environ.get("SDL_VIDEODRIVER") == "dummy" and not getattr(args, "no_render", False):
        os.environ.pop("SDL_VIDEODRIVER")

    from handoff.config import checkpoint_path, load_config, load_run_config, run_dir_of
    from handoff.viz import live, replay

    ckpt = checkpoint_path(args.checkpoint) if args.checkpoint else None
    if args.mode == "live":
        if ckpt is None and args.method != "b0":
            sys.exit("--checkpoint is required (or --method b0)")
        overrides = args.overrides + (["sim.render_clean=true"] if args.clean else [])
        cfg = load_run_config(run_dir_of(ckpt), overrides) if ckpt else load_config("b0", overrides=overrides)
        seeds = args.seeds or cfg["scenario"]["seeds"]["eval"][:3]
        out = Path(args.out or f"runs/view/{cfg['method']['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
        for s in live(cfg, ckpt, seeds, out, render=not args.no_render, fps=args.fps,
                      deterministic=not args.stochastic, paused=args.paused,
                      panel=not args.clean):
            print(f"seed {s['seed']}: {s['termination']}  t={s['sim_time']:.1f}s  v={s['mean_speed'] * 3.6:.1f} km/h  "
                  f"return {s['return']:.1f}")
        print(f"wrote {out}/")
    else:
        ep = Path(args.episode)
        run = run_dir_of(ckpt) if ckpt else ep.parent.parent
        if not (run / "config.yaml").exists():
            sys.exit(f"config not found: {run / 'config.yaml'}")
        cfg = load_run_config(run, args.overrides)
        if cfg["method"]["name"] != "b0" and ckpt is None:
            sys.exit("--checkpoint is required to re-simulate an RL episode")
        print(replay(cfg, str(ep), ckpt))


if __name__ == "__main__":
    main()
