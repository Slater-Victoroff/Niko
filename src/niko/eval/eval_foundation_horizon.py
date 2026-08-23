"""Re-evaluate a trained FoundationModel checkpoint's per-task validation set
at specific rollout horizons (t+1, t+4, t+8, t+16 by default) instead of the
single mean-over-all-K-steps number every other eval in this project reports.

The checkpoint was trained with rollout_steps=6 (K=6): t+1..t+6 are within
the training horizon, t+8/t+16 push the operator well past it -- a genuine
out-of-distribution rollout-length test, not just a finer-grained view of the
same regime. Reports both well_style_vrmse (std-normalized, matches every
other number in this project) and the Well paper's own NRMSE (Eq. 6,
RMS-normalized) at each horizon, since they can diverge substantially (see
active_matter in eval_foundation_nrmse.py).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import FieldEmbedder, state_dict_uses_batchnorm
from encoders.context_cond import ContextCondEncoder
from encoders.sequence_conv import SequenceConvEncoder
from operators.transport_operator import TransportOperator
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder
from models.foundation_model import FoundationModel
import dataloader as dl
from training.train_foundation import build_tasks, union_channel_specs


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
    p.add_argument("--task-set", default="core3", choices=["core3", "all14"])
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    horizons = [int(h) for h in args.horizons.split(",")]
    max_horizon = max(horizons)

    tasks_cfg = build_tasks(args.data_root, args.config_root, task_set=args.task_set)
    channel_specs = union_channel_specs(tasks_cfg)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    T = ckpt["context_frames"]
    canonical_dim, latent_dim = ckpt["canonical_dim"], ckpt["latent_dim"]
    hidden_dim, cond_dim = ckpt["hidden_dim"], ckpt["cond_dim"]
    decoder_hidden_dim = ckpt["decoder_hidden_dim"]
    block_kernel_size = ckpt.get("block_kernel_size", 7)

    use_bn = state_dict_uses_batchnorm(ckpt["field_embedder_state_dict"])
    field_embedder = FieldEmbedder(channel_specs, canonical_dim=canonical_dim, use_batchnorm=use_bn).to(dev)
    field_embedder.load_state_dict(ckpt["field_embedder_state_dict"])

    context_cond_encoder = ContextCondEncoder(
        in_channels=canonical_dim, context_frames=T, cond_dim=cond_dim, hidden_dim=hidden_dim,
        block_kernel_size=block_kernel_size,
    ).to(dev)
    context_cond_encoder.load_state_dict(ckpt["context_cond_encoder_state_dict"])

    encoder = SequenceConvEncoder(
        in_channels=canonical_dim, context_frames=T, latent_dim=latent_dim, hidden_dim=hidden_dim,
        block_kernel_size=block_kernel_size,
    ).to(dev)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    complex_term = any("complex_amplitude" in k or "complex_rotation" in k for k in ckpt["operator_state_dict"])
    operator = TransportOperator(
        latent_dim=latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim,
        terms=("advection", "diffusion", "skew", "forcing"), film=True,
        complex_term=complex_term, block_kernel_size=block_kernel_size,
    ).to(dev)
    operator.load_state_dict(ckpt["operator_state_dict"])

    task_names = args.tasks or list(ckpt["decoder_state_dicts"].keys())
    decoders = {}
    for t in task_names:
        decoders[t] = SharedTrunkFieldHeadsDecoder(
            latent_dim=latent_dim, hidden_dim=decoder_hidden_dim, upsample=2,
            block_kernel_size=block_kernel_size,
            **tasks_cfg[t]["decoder_kwargs"],
        ).to(dev)
        decoders[t].load_state_dict(ckpt["decoder_state_dicts"][t])

    model = FoundationModel(field_embedder, context_cond_encoder, encoder, operator, decoders).to(dev)
    if complex_term:
        saved_complex_proj = ckpt.get("complex_proj_state_dict")
        if saved_complex_proj is not None:
            model.complex_proj.load_state_dict(saved_complex_proj)
            print("Loaded trained complex_proj from checkpoint.")
        else:
            print("NOTE: complex_term=True but no saved complex_proj in this checkpoint -- "
                  "spectral branch is a zero-init no-op here, numbers will understate the real model.")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded {args.checkpoint} (epoch={ckpt['epoch']}, trained rollout_steps={ckpt['rollout_steps']}), "
          f"evaluating horizons t+{horizons} (max t+{max_horizon}, {'within' if max_horizon <= ckpt['rollout_steps'] else 'BEYOND'} training horizon)")

    for t in task_names:
        cfg = tasks_cfg[t]
        param_choices = None
        if cfg.get("param_subset"):
            with open(cfg["param_subset"]) as f:
                param_choices = [tuple(c) for c in json.load(f)["combos"]]

        _, val_loader, _ = dl.create_param_dataloaders(
            cfg["data_dir"], batch_size=args.batch, context_frames=T, predict_frames=max_horizon,
            param_choices=param_choices, train_file_limit=cfg.get("train_file_limit"),
            val_file_limit=cfg.get("val_file_limit"), traj_limit=cfg.get("traj_limit"),
            pair_stride=cfg.get("pair_stride", 1), field_spec=cfg["field_spec"],
            traj_cache_capacity=cfg["traj_cache_capacity"],
        )
        if len(val_loader) == 0:
            print(f"[{t}] SKIPPED -- no val windows fit context={T}+predict={max_horizon} "
                  f"(trajectory too short for this task)")
            continue

        agg = {h: {"vrmse": 0.0, "nrmse_well": 0.0} for h in horizons}
        n_batches = 0
        with torch.no_grad():
            for xb, yb, _bparams in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb, field_spec=cfg["field_spec"], task=t, steps=max_horizon, return_initial_encode=False)
                for h in horizons:
                    stats = step_stats(pred[:, h - 1], yb[:, h - 1])
                    for k in stats:
                        agg[h][k] += stats[k]
                n_batches += 1

        print(f"\n[{t}]  ({n_batches} val batches, context={T})")
        for h in horizons:
            v = agg[h]["vrmse"] / max(1, n_batches)
            n = agg[h]["nrmse_well"] / max(1, n_batches)
            tag = "" if h <= ckpt["rollout_steps"] else "  [beyond training horizon]"
            print(f"  t+{h:<3d} vrmse: {v:.6f}   nrmse(Well): {n:.6f}{tag}")


if __name__ == "__main__":
    main()
