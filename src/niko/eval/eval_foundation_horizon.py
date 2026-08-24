"""Re-evaluate a trained FoundationModel checkpoint's per-task validation set
at specific rollout horizons (t+1, t+4, t+8, t+16 by default) instead of the
single mean-over-all-K-steps number every other eval in this project reports
-- "DISCO-comparable" numbers, individual-frame accuracy at fixed horizons
rather than a rollout-averaged number.

The checkpoint was trained with rollout_steps=6 (K=6): t+1..t+6 are within
the training horizon, t+8/t+16 push the operator well past it -- a genuine
out-of-distribution rollout-length test, not just a finer-grained view of the
same regime. Reports both well_style_vrmse (std-normalized, matches every
other number in this project) and the Well paper's own NRMSE (Eq. 6,
RMS-normalized, matches this project's --loss-fn nrmse_well) at each horizon,
since they can diverge substantially (see eval_foundation_nrmse.py).

2026-08-24: rewritten to load via load_foundation_checkpoint (the single
drift-proof way to reconstruct a FoundationModel -- see train_foundation.py)
instead of hand-reconstructing every submodule from individual state_dict
keys. The old hand-reconstruction predated that refactor and would have
silently built the model WITHOUT boundary_geometry (FoundationModel's
default when not passed) and with a hardcoded, possibly-stale operator term
list -- exactly the class of checkpoint drift that refactor exists to
prevent, and it would have gone undetected here since nothing would have
errored, just quietly evaluated the wrong architecture.

--full-holdout: by default this script mirrors whatever traj_limit/
val_traj_limit build_tasks() already defines for each task (the same
holdout definition training validated against every epoch). Pass
--full-holdout to override every task's trajectory limit to None (use
every trajectory in the already-held-out validation files) for this one
evaluation -- file-level selection (which val files belong to a task's
holdout at all -- e.g. rayleigh_benard's curated broad8 combos) is
unrelated to this flag and always stays as build_tasks() defines it.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataloader as dl
from training.train_foundation import build_tasks, load_foundation_checkpoint


def step_stats(pred_step: torch.Tensor, target_step: torch.Tensor, eps: float = 1e-7) -> dict:
    """pred_step/target_step: [B, C, H, W] (one horizon) -> scalar dict,
    averaged over B, C (spatial dims reduced first, per-sample-per-channel,
    same convention as well_style_vrmse/eval_foundation_nrmse.py)."""
    spatial_dims = (-2, -1)
    mse = (pred_step - target_step).float().pow(2).mean(dim=spatial_dims)  # [B,C]
    rmse = mse.sqrt()

    var = target_step.float().std(dim=spatial_dims, unbiased=False).pow(2)
    vrmse = torch.sqrt(mse / (var + eps))

    target_rms = (target_step.float().pow(2).mean(dim=spatial_dims) + eps).sqrt()
    nrmse_well = rmse / (target_rms + eps)

    return {"vrmse": vrmse.mean().item(), "nrmse_well": nrmse_well.mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", action="append", dest="tasks", default=None)
    p.add_argument("--horizons", default="1,4,8,16", help="comma-separated t+k values to report")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="/app/data/datasets")
    p.add_argument("--config-root", default="/app/configs")
    p.add_argument("--task-set", default="core3", choices=["core3", "all14", "fast4"])
    p.add_argument("--full-holdout", action="store_true",
                    help="override every task's traj_limit to None for this eval -- use every trajectory in "
                         "the already-held-out validation files instead of whatever (possibly-bounded) "
                         "val_traj_limit training used each epoch. File-level holdout selection is unaffected.")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    horizons = [int(h) for h in args.horizons.split(",")]
    max_horizon = max(horizons)

    tasks_cfg = build_tasks(args.data_root, args.config_root, task_set=args.task_set)

    model, ckpt = load_foundation_checkpoint(args.checkpoint, dev)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    T = ckpt["model_config"]["context_frames"]

    task_names = args.tasks or list(ckpt["model_config"]["tasks"].keys())
    print(f"Loaded {args.checkpoint} (epoch={ckpt['epoch']}, trained rollout_steps={ckpt['rollout_steps']}, "
          f"loss_fn={ckpt.get('loss_fn', 'vrmse')}), evaluating horizons t+{horizons} "
          f"(max t+{max_horizon}, {'within' if max_horizon <= ckpt['rollout_steps'] else 'BEYOND'} training horizon)"
          f"{', full holdout (unbounded traj_limit)' if args.full_holdout else ''}")

    for t in task_names:
        cfg = tasks_cfg[t]
        param_choices = None
        if cfg.get("param_subset"):
            with open(cfg["param_subset"]) as f:
                param_choices = [tuple(c) for c in json.load(f)["combos"]]

        val_traj_limit = None if args.full_holdout else cfg.get("val_traj_limit")
        _, val_loader, _ = dl.create_param_dataloaders(
            cfg["data_dir"], batch_size=args.batch, context_frames=T, predict_frames=max_horizon,
            param_choices=param_choices, train_file_limit=cfg.get("train_file_limit"),
            val_file_limit=cfg.get("val_file_limit"), traj_limit=cfg.get("traj_limit"),
            val_traj_limit=val_traj_limit, pair_stride=cfg.get("pair_stride", 1), field_spec=cfg["field_spec"],
            traj_cache_capacity=cfg["traj_cache_capacity"],
        )
        if len(val_loader) == 0:
            print(f"[{t}] SKIPPED -- no val windows fit context={T}+predict={max_horizon} "
                  f"(trajectory too short for this task)")
            continue

        # Per-batch values, not a running sum -- see median() below. A plain mean here
        # has the exact same outlier-batch problem train_foundation.py's own validation
        # loop had before its 2026-08-23 fix (see EXPERIMENT_LOG.md): a handful of
        # batches landing on a target that's converged to a spatially-constant state
        # can dominate a mean by 5-10x even under nrmse_well, which is far better-
        # behaved than vrmse there but not perfectly immune. Report both so nothing's
        # hidden, but median is the one to actually read.
        per_batch = {h: {"vrmse": [], "nrmse_well": []} for h in horizons}
        n_batches = 0
        with torch.no_grad():
            for xb, yb, _bparams in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb, field_spec=cfg["field_spec"], task=t, steps=max_horizon, return_initial_encode=False)
                for h in horizons:
                    stats = step_stats(pred[:, h - 1], yb[:, h - 1])
                    for k in stats:
                        per_batch[h][k].append(stats[k])
                n_batches += 1

        def median(vals):
            s = sorted(vals)
            n = len(s)
            if n == 0:
                return float("nan")
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

        print(f"\n[{t}]  ({n_batches} val batches, context={T})")
        for h in horizons:
            v_vals, n_vals = per_batch[h]["vrmse"], per_batch[h]["nrmse_well"]
            v_med, v_mean = median(v_vals), sum(v_vals) / max(1, len(v_vals))
            n_med, n_mean = median(n_vals), sum(n_vals) / max(1, len(n_vals))
            tag = "" if h <= ckpt["rollout_steps"] else "  [beyond training horizon]"
            print(f"  t+{h:<3d} vrmse: median={v_med:.6f} mean={v_mean:.6f}   "
                  f"nrmse(Well): median={n_med:.6f} mean={n_mean:.6f}{tag}")


if __name__ == "__main__":
    main()
