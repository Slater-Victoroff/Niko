"""Per-rollout-step validation vrmse for a saved FoundationModel checkpoint --
collapses only batch+channel (well_style_vrmse's [B, K, C] -> mean over B, C,
keep K), unlike train_foundation.py's own validation loop, which collapses K
too and reports one aggregate number per task per epoch.

Built specifically to answer: does a rollout model's own per-step error grow
with horizon (step 1 easiest, step K hardest, from compounding rollout
error)? If so, a K-step-averaged aggregate number isn't directly comparable
to a dedicated single-step (K=1) model's number -- the aggregate is dragged
up by its harder late steps, so "single-step loss ~= K-step average" is NOT
evidence the single-step model is doing about as well as the K-step model's
own first step; it could mean the opposite. See EXPERIMENT_LOG.md for the
question this was built to settle.

Usage: python -u eval/eval_per_step_vrmse.py <checkpoint.pt> --task <name> \
    --data-root <path> --config-root <path> [--device cuda:0] [--max-batches N]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_foundation import load_foundation_checkpoint, build_tasks
from training.losses import well_style_vrmse
import dataloader as dl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--task", required=True)
    p.add_argument("--task-set", default="core3")
    p.add_argument("--data-root", required=True)
    p.add_argument("--config-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-batches", type=int, default=None,
                    help="cap val batches for a quick check; default uses the full val set")
    args = p.parse_args()

    dev = torch.device(args.device)
    model, ckpt = load_foundation_checkpoint(args.checkpoint, dev)
    model.eval()
    T = ckpt["model_config"]["context_frames"]
    K = ckpt["rollout_steps"]
    print(f"Loaded {args.checkpoint}: ctx{T}/roll{K}, epoch {ckpt.get('epoch')}, "
          f"saved val_loss_by_task={ckpt.get('val_loss_by_task')}")

    tasks = build_tasks(args.data_root, args.config_root, task_set=args.task_set)
    cfg = tasks[args.task]
    param_choices = None
    if cfg.get("param_subset"):
        import json
        with open(cfg["param_subset"]) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]
    _, val_loader, _ = dl.create_param_dataloaders(
        cfg["data_dir"], batch_size=cfg["batch"], context_frames=T, predict_frames=K,
        num_workers=0, param_choices=param_choices,
        train_file_limit=cfg.get("train_file_limit"), val_file_limit=cfg.get("val_file_limit"),
        traj_limit=cfg.get("traj_limit"), pair_stride=cfg.get("pair_stride", 1),
        field_spec=cfg["field_spec"], traj_cache_capacity=cfg["traj_cache_capacity"],
        file_select_fn=cfg.get("file_select_fn"),
    )

    per_step_sum = torch.zeros(K, dtype=torch.float64)
    n_batches = 0
    with torch.no_grad():
        for xb, yb, _bparams in val_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb, field_spec=cfg["field_spec"], task=args.task, steps=K, return_initial_encode=False)
            # [B, K, C] -> mean over B, C only, KEEP K -- the whole point vs. train_foundation.py's .mean()
            step_vrmse = well_style_vrmse(pred, yb).mean(dim=(0, 2))
            per_step_sum += step_vrmse.double().cpu()
            n_batches += 1
            if args.max_batches is not None and n_batches >= args.max_batches:
                break

    per_step_mean = (per_step_sum / n_batches).tolist()
    print(f"\n{n_batches} val batches, per-rollout-step vrmse (batch+channel averaged, step kept):")
    for k, v in enumerate(per_step_mean, start=1):
        print(f"  step {k}/{K}: {v:.6f}")
    print(f"\naggregate (mean over all K steps, matches train_foundation.py's own valid_loss[{args.task}]): "
          f"{sum(per_step_mean) / len(per_step_mean):.6f}")


if __name__ == "__main__":
    main()
