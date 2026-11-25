#!/usr/bin/env python
"""Behaviour cloning (BC) of M1 from human drives: fit PPO's own policy network to the human's actions.

Input: the dataset from scripts/warm_start/build_demos.py (runs/demos/m1.npz). Output: a run folder that looks like a
PPO run, so the clone can be watched with scripts/view.py and loaded as a PPO warm start (training.init_policy):

    runs/bc/m1/<timestamp>/
        config.yaml               M1's config (+ a "bc" section with these settings)
        checkpoints/best.zip      SB3 PPO model of the selected epoch (--select: best driving, or lowest val loss)
        bc_progress.csv           per-epoch train / validation losses, steering metrics, driving results
        bc_summary.json           dataset split, settings, best epoch

    python scripts/warm_start/bc.py                          # defaults: 40 epochs, 3 held-out episodes
    python scripts/warm_start/bc.py --epochs 5               # quick smoke run
    python scripts/warm_start/bc.py --weight-decay 0.05 --max-weight 3  # regularised variant
    python scripts/view.py live --checkpoint runs/bc/m1/<ts>                       # watch it drive

The network is exactly the one scripts/train.py builds (PPO "MultiInputPolicy" with M1's ppo kwargs: the
observation vectors concatenated -> policy MLP -> action mean). Only the actor (policy MLP, action head) is trained;
the critic and log_std keep their initial values (the warm start handles those: docs/GUIDE.md).

Two guards against the usual BC failure modes:
  * copycat: the ego vector holds the PREVIOUS step's steering / throttle_brake, which almost equals the label (the
    human rarely changes the controls between 0.1 s steps). A net given those inputs learns "repeat the last action"
    and ignores the road. They are zeroed in every training sample AND their first-layer weights are held at zero,
    so the clone acts the same whatever they are (PPO can learn to use them later). --keep-prev-action disables it.
  * imbalance: most samples have no steering and ~1 % brake. Samples that steer or brake are up-weighted by the
    inverse of their frequency (capped at --max-weight), and each action dimension's error is divided by its label
    variance.

Checkpoint selection (--select drive, the default): validation loss is a poor guide to whether the clone can drive,
so after every epoch the clone drives --drive-seeds (training-range seeds not in the demos; the 40 evaluation seeds
stay untouched) and the epoch with the furthest mean distance is kept (ties: lower val loss).
"""

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from handoff.config import DEFAULT_REWARD

PREV_ACTION_FEATURES = ("steering", "throttle_brake")   # ego features that are the previous action
STEER_ACTIVE = 0.02     # |lateral action| above this = a steering sample
BRAKE_ACTIVE = -0.01    # accel action below this = a braking sample


def build_model(cfg: dict, seed: int):
    """The PPO model scripts/train.py would build for this config (same spaces, policy class and kwargs), without
    starting MetaDrive."""
    import warnings

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from handoff.env import observation_space as obs_space
    from handoff.train import ppo_kwargs

    warnings.filterwarnings("ignore", message=".*mini-batch size.*")   # rollout-buffer sizing; irrelevant here

    class _Env(gym.Env):
        observation_space = obs_space(cfg)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

    return PPO("MultiInputPolicy", DummyVecEnv([_Env]), seed=seed, device="cpu", verbose=0, **ppo_kwargs(cfg))


def masked_feature_columns(policy, ego_index: list[int]) -> list[int]:
    """Columns of the features vector (input of the policy / value MLPs) that carry the given ego entries. Found by
    probing the extractor rather than assuming the Dict key order."""
    import torch

    space = policy.observation_space
    with torch.no_grad():
        base = {k: torch.zeros((1, *s.shape)) for k, s in space.spaces.items()}
        f0 = policy.extract_features(base, policy.pi_features_extractor)
        cols = []
        for j in ego_index:
            probe = {k: v.clone() for k, v in base.items()}
            probe["ego"][0, j] = 1.0
            diff = (policy.extract_features(probe, policy.pi_features_extractor) - f0).abs()[0]
            hit = torch.nonzero(diff > 0).flatten().tolist()
            if len(hit) != 1:
                raise RuntimeError(f"ego[{j}] reaches {len(hit)} feature columns; expected 1 (pass-through)")
            cols += hit
    return cols


def first_layers(policy):
    """The first Linear of the policy MLP and of the value MLP (both read the features vector)."""
    import torch.nn as nn

    return [next(m for m in net.modules() if isinstance(m, nn.Linear))
            for net in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net)]


def action_mean(policy, obs: dict):
    feats = policy.extract_features(obs, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(feats))


def sample_weights(a: np.ndarray, max_weight: float) -> tuple[np.ndarray, dict]:
    steer = np.abs(a[:, 0]) > STEER_ACTIVE
    brake = a[:, 1] < BRAKE_ACTIVE
    ws = min(max_weight, (~steer).sum() / max(steer.sum(), 1))
    wb = min(max_weight, (~brake).sum() / max(brake.sum(), 1))
    w = np.ones((len(a), 2), np.float32)
    w[steer, 0] = ws
    w[brake, 1] = wb
    return w, {"steer_active_frac": float(steer.mean()), "brake_frac": float(brake.mean()),
               "steer_weight": float(ws), "brake_weight": float(wb)}


def drive_score(summaries: list[dict]) -> dict:
    dist = np.array([float(s["distance"]) for s in summaries])
    term = [s["termination"] for s in summaries]
    return {"drive_dist": float(dist.mean()), "drive_return": float(np.mean([float(s["return"]) for s in summaries])),
            "drive_speed_kmh": float(np.mean([float(s["mean_speed"]) for s in summaries]) * 3.6),
            "drive_success": float(np.mean([t == "success" for t in term])),
            "drive_off_road": float(np.mean([t == "off_road" for t in term])),
            "drive_collision": float(np.mean([t == "collision" for t in term]))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="runs/demos/m1.npz")
    p.add_argument("--out", default=None, help="run folder (default runs/bc/m1/<timestamp>)")
    p.add_argument("--reward", default=DEFAULT_REWARD, help="reward config written into config.yaml (BC ignores it)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam)")
    p.add_argument("--val-episodes", type=int, default=3, help="whole episodes held out for validation")
    p.add_argument("--max-weight", type=float, default=10.0, help="cap of the steer / brake up-weighting")
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better selection score")
    p.add_argument("--select", choices=["drive", "val"], default="drive",
                   help="checkpoint selection: furthest drive (default) or lowest val loss")
    p.add_argument("--drive-seeds", type=int, nargs="+", default=list(range(100, 110)),
                   help="training-range seeds the clone drives for selection (demo seeds are removed)")
    p.add_argument("--n-workers", type=int, default=5, help="parallel driving workers")
    p.add_argument("--keep-prev-action", action="store_true", help="do NOT mask the previous-action ego features")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    from handoff.config import load_config
    from handoff.env import EGO_FEATURES, observation_space
    from handoff.logs import make_run_dir

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = load_config("m1", reward=args.reward)

    # ------------------------------------------------------------------ data
    z = np.load(args.data)
    space = observation_space(cfg).spaces
    stale = [k for k, s in space.items() if f"obs_{k}" not in z.files or z[f"obs_{k}"].shape[1:] != s.shape]
    if stale:
        raise SystemExit(f"{args.data} does not match today's observation ({stale} missing or a different size): "
                         "rebuild it with scripts/warm_start/build_demos.py")
    keys = list(space)
    obs = {k: z[f"obs_{k}"].astype(np.float32) for k in keys}
    act, ep = z["action"].astype(np.float32), z["episode"]
    n_ep = int(ep.max()) + 1
    val_eps = sorted(rng.choice(n_ep, size=min(args.val_episodes, n_ep - 1), replace=False).tolist())
    is_val = np.isin(ep, val_eps)
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]

    mask_idx = [] if args.keep_prev_action else [EGO_FEATURES.index(f) for f in PREV_ACTION_FEATURES]
    obs["ego"][:, mask_idx] = 0.0

    weights = np.ones_like(act)
    weights[tr_idx], wstats = sample_weights(act[tr_idx], args.max_weight)
    weights[va_idx] = sample_weights(act[va_idx], args.max_weight)[0]   # validation weighted with its own counts
    var = np.maximum(act[tr_idx].var(axis=0), 1e-4).astype(np.float32)

    # ------------------------------------------------------------------ model
    model = build_model(cfg, args.seed)
    policy = model.policy
    cols = masked_feature_columns(policy, mask_idx) if mask_idx else []
    layers = first_layers(policy)

    def zero_masked():
        with torch.no_grad():
            for lin in layers:
                lin.weight[:, cols] = 0.0

    zero_masked()
    actor = (list(policy.features_extractor.parameters()) + list(policy.mlp_extractor.policy_net.parameters())
             + list(policy.action_net.parameters()))
    opt = (torch.optim.AdamW(actor, lr=args.lr, weight_decay=args.weight_decay) if args.weight_decay > 0
           else torch.optim.Adam(actor, lr=args.lr))
    var_t = torch.as_tensor(var)

    def batch(idx):
        return ({k: torch.as_tensor(v[idx]) for k, v in obs.items()}, torch.as_tensor(act[idx]),
                torch.as_tensor(weights[idx]))

    def loss_fn(mean, a, w):
        return ((w * (mean - a) ** 2) / var_t).mean()

    def evaluate(idx):
        policy.set_training_mode(False)
        tot, preds = 0.0, []
        with torch.no_grad():
            for k in range(0, len(idx), 512):
                o, a, w = batch(idx[k:k + 512])
                m = action_mean(policy, o)
                tot += float(loss_fn(m, a, w)) * len(a)
                preds.append(m.numpy())
        pr, a = np.concatenate(preds), act[idx]
        steer = np.abs(a[:, 0]) > STEER_ACTIVE
        return {"loss": tot / len(idx), "steer_mae": float(np.mean(np.abs(pr[:, 0] - a[:, 0]))),
                "steer_mae_active": float(np.mean(np.abs(pr[steer, 0] - a[steer, 0]))) if steer.any() else np.nan,
                "steer_sign_acc": float(np.mean(np.sign(pr[steer, 0]) == np.sign(a[steer, 0]))) if steer.any()
                else np.nan, "throttle_mae": float(np.mean(np.abs(pr[:, 1] - a[:, 1])))}

    # ------------------------------------------------------------------ run folder
    cfg["experiment"]["runs_root"] = "runs/bc"
    cfg["bc"] = {"data": args.data, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "weight_decay": args.weight_decay, "select": args.select, "val_episodes": val_eps,
                 "max_weight": args.max_weight, "masked_ego_features": [EGO_FEATURES[i] for i in mask_idx],
                 "seed": args.seed}
    run = Path(args.out) if args.out else Path("runs/bc/m1") / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = make_run_dir(cfg, run)
    ckpt = run / "checkpoints" / "best.zip"
    drive_seeds = [s for s in args.drive_seeds if s not in set(int(x) for x in z["seeds"])]
    evaluator = None
    if args.select == "drive":
        from handoff.evaluate import ParallelEvaluator, policy_spec

        evaluator = ParallelEvaluator(cfg, None, min(args.n_workers, len(drive_seeds)), policy_spec(model))
        print(f"driving test after every epoch on seeds {drive_seeds}")

    # Reference: "zero steering, mean throttle" on the validation set
    mean_label = act[tr_idx].mean(axis=0)
    mean_label[0] = 0.0
    base_pred = np.tile(mean_label, (len(va_idx), 1)).astype(np.float32)
    base_loss = float(((weights[va_idx] * (base_pred - act[va_idx]) ** 2) / var).mean())
    print(f"train {len(tr_idx)} samples ({n_ep - len(val_eps)} episodes), val {len(va_idx)} (episodes {val_eps}); "
          f"masked ego features: {[EGO_FEATURES[i] for i in mask_idx] or 'none'}")
    print(f"weights: steering active {wstats['steer_active_frac']:.0%} x{wstats['steer_weight']:.1f}, "
          f"braking {wstats['brake_frac']:.1%} x{wstats['brake_weight']:.1f}; "
          f"baseline val loss (zero steer, mean throttle) {base_loss:.3f}")
    print(f"run folder: {run}\n")

    log = open(run / "bc_progress.csv", "w", newline="")
    writer = None
    best, best_epoch, stale, t0, best_row = (np.inf,), -1, 0, time.time(), {}
    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for k in range(0, len(perm), args.batch_size):
            o, a, w = batch(perm[k:k + args.batch_size])
            loss = loss_fn(action_mean(policy, o), a, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            zero_masked()
            tot += float(loss.detach()) * len(a)
        va = evaluate(va_idx)
        dr = {}
        if evaluator is not None:
            state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            dr = drive_score(evaluator.evaluate(state, drive_seeds, deterministic=True))
        row = {"epoch": epoch, "train_loss": tot / len(perm), **{f"val_{k}": v for k, v in va.items()},
               "drive_dist": np.nan, "drive_return": np.nan, "drive_speed_kmh": np.nan, "drive_success": np.nan,
               "drive_off_road": np.nan, "drive_collision": np.nan, **dr, "seconds": time.time() - t0}
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        key = (va["loss"],) if args.select == "val" else (-dr["drive_dist"], va["loss"])
        better = key < best
        if better:
            best, best_epoch, stale, best_row = key, epoch, 0, row
            model.save(ckpt)
        else:
            stale += 1
        drive_txt = (f"  drive {dr['drive_dist']:4.0f} m, {dr['drive_speed_kmh']:.0f} km/h, return "
                     f"{dr['drive_return']:.0f}, success {dr['drive_success']:.0%}, off road "
                     f"{dr['drive_off_road']:.0%}" if dr else "")
        print(f"epoch {epoch:3d}  train {row['train_loss']:.3f}  val {va['loss']:.3f}  steer MAE "
              f"{va['steer_mae']:.3f} (active {va['steer_mae_active']:.3f}, sign {va['steer_sign_acc']:.0%})  "
              f"throttle MAE {va['throttle_mae']:.3f}{drive_txt}  {row['seconds']:.0f} s{'  *' if better else ''}",
              flush=True)
        if stale >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break
    log.close()
    if evaluator is not None:
        evaluator.close()

    summary = {"best_epoch": best_epoch, "best": best_row, "baseline_val_loss": base_loss,
               "drive_seeds": drive_seeds, "train_samples": int(len(tr_idx)), "val_samples": int(len(va_idx)),
               "val_episodes": val_eps, "val_seeds": [int(z["seeds"][i]) for i in val_eps], **wstats,
               "args": vars(args), "masked_feature_columns": cols}
    (run / "bc_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nselected epoch {best_epoch} ({args.select}): val loss {best_row.get('val_loss', np.nan):.3f} "
          f"(baseline {base_loss:.3f}), drive {best_row.get('drive_dist', np.nan):.0f} m -> {ckpt}")
    print(f"watch it:  python scripts/view.py live --checkpoint {ckpt}")


if __name__ == "__main__":
    main()
