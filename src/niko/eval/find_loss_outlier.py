"""Diagnostic: load a foundation_model checkpoint and sweep every batch of
one task's train_loader (no_grad, eval mode) looking for anomalously large
per-batch loss -- used to find/reproduce the single-step loss spike from
foundation_model's epoch 5 (train_loss[rayleigh_benard] averaged 15.7M over
12096 batches despite every logged 50-step sample looking normal, strongly
suggesting one or a few extreme outlier batches rather than a sustained
instability).
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import canonical_field_name
import training.train_foundation as tf
from training.losses import well_style_vrmse
import dataloader as dl

# NOTE 2026-08-20: TASKS below is this script's OWN local copy (data_dir/
# param_subset/field_spec), independent of train_foundation.py's build_tasks() --
# kept only for the dataloading fields (data_dir, param_subset, train_file_limit)
# a checkpoint doesn't carry. Model *architecture* now comes entirely from the
# checkpoint's own saved model_config (see load_foundation_checkpoint below), not
# from this dict, so a mismatch here can no longer silently break reconstruction
# the way it used to. data_dir values are still the old Docker-era /app/data/...
# paths, not migrated to AICR's /work/aihub/... layout -- unrelated to the
# checkpoint-drift fix, not touched here; this script needs --data-root wiring
# (like train_foundation.py has) before it's actually runnable on AICR.


TASKS = {
    "rayleigh_benard": dict(
        field_spec=dl.RB_FIELD_SPEC,
        data_dir="/app/data/datasets/rayleigh_benard/data",
        param_subset="/app/configs/param_subsets/broad8.json",
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True),
    ),
    "shear_flow": dict(
        field_spec=dl.SHEAR_FLOW_FIELD_SPEC,
        data_dir="/app/data/datasets/shear_flow/data",
        param_subset="/app/configs/param_subsets/shear_flow_broad8.json",
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True,
                             scalar_field_names=["pressure", "tracer"]),
    ),
    "active_matter": dict(
        field_spec=dl.ACTIVE_MATTER_FIELD_SPEC,
        data_dir="/app/data/datasets/active_matter/data",
        param_subset=None,
        train_file_limit=8,
        val_file_limit=8,
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["concentration"], tensor_field_names=["D", "E"]),
    ),
}


def union_channel_specs(tasks: dict) -> dict:
    specs = {}
    for cfg in tasks.values():
        for spec in cfg["field_spec"]:
            name = canonical_field_name(spec["key"])
            specs[name] = spec["n_components"]
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=50.0, help="flag any batch with loss above this")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    # Single drift-proof load: architecture rebuilt from the checkpoint's own
    # model_config (not from this file's local TASKS), full state_dict loaded --
    # see build_foundation_model/load_foundation_checkpoint in train_foundation.py.
    model, ckpt = tf.load_foundation_checkpoint(args.checkpoint, dev)
    model.eval()
    K = ckpt["rollout_steps"]
    T = ckpt["model_config"]["context_frames"]
    print(f"Loaded {args.checkpoint} (epoch={ckpt['epoch']}, val_loss_by_task={ckpt['val_loss_by_task']})")

    cfg = TASKS[args.task]
    # field_spec for both the dataloader and the forward call below comes from the
    # checkpoint's own model_config, not this file's local TASKS -- so a stale/
    # mismatched local field_spec can't silently misalign channels the way it could
    # before.
    ckpt_field_spec = ckpt["model_config"]["tasks"][args.task]["field_spec"]
    import json
    param_choices = None
    if cfg.get("param_subset"):
        with open(cfg["param_subset"]) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]

    train_loader, _val_loader, _ = dl.create_param_dataloaders(
        cfg["data_dir"], batch_size=args.batch, context_frames=T, predict_frames=K,
        num_workers=0, param_choices=param_choices,
        train_file_limit=cfg.get("train_file_limit"), field_spec=ckpt_field_spec,
        traj_cache_capacity=8, shuffle_train=False,
    )
    print(f"Sweeping {len(train_loader)} batches of {args.task} train data...")

    worst = []
    n_flagged = 0
    with torch.no_grad():
        for i, (xb, yb, bparams) in enumerate(train_loader):
            xb, yb = xb.to(dev), yb.to(dev)
            initial, pred = model(xb, field_spec=ckpt_field_spec, task=args.task, steps=K,
                                   return_initial_encode=True)
            initial_target = xb[:, -1, ...]
            initial_loss = well_style_vrmse(initial.unsqueeze(1), initial_target.unsqueeze(1)).mean()
            rollout_loss = well_style_vrmse(pred, yb).mean()
            loss = (1.0 / K) * initial_loss + rollout_loss
            loss_val = float(loss.item())

            worst.append((loss_val, i, bparams[0].tolist()))
            if loss_val > args.threshold:
                n_flagged += 1
                print(f"  batch {i}: loss={loss_val:.4f}  finite={torch.isfinite(loss).item()}  bparams[0]={bparams[0].tolist()}")

            if i % 500 == 0:
                print(f"  ...swept {i}/{len(train_loader)}")

    worst.sort(key=lambda t: -t[0])
    print(f"\nDone. {n_flagged} batches above threshold={args.threshold}.")
    print("Top 10 worst batches (loss, batch_idx, bparams[0]):")
    for loss_val, i, bp in worst[:10]:
        print(f"  {loss_val:.4f}  batch={i}  bparams[0]={bp}")


if __name__ == "__main__":
    main()
