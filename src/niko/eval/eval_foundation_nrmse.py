"""Re-evaluate a trained FoundationModel checkpoint's per-task validation set,
reporting the same rollout predictions under several different error
normalizations side by side -- not just well_style_vrmse (RMSE normalized by
the target's own per-sample spatial std, what every training/val loss number
in this project has used so far), but also:

  - raw RMSE (no normalization)
  - NRMSE_range: RMSE / (target.max() - target.min()), per sample per channel
  - NRMSE_mean: RMSE / mean(|target|), per sample per channel

All four are computed from the exact same predictions in one pass, so they're
directly comparable -- this isn't a different eval run, just a different lens
on the same errors.

Caveat: train_foundation.py's checkpoint save block does not persist
FoundationModel.complex_proj's state (only field_embedder/context_cond_encoder/
encoder/operator/decoders). For a --complex-term checkpoint, this script's
freshly-constructed complex_proj is therefore zero-init (FoundationModel's own
init default) rather than the checkpoint's actual trained weights, making the
spectral branch a true no-op here regardless of what it contributed during
training. The script prints a recomputed well_style_vrmse next to the
checkpoint's own stored val_loss_by_task so you can see directly how much that
gap matters for this checkpoint.
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


def error_stats(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> dict:
    """pred/target: [B, K, C, H, W] -> dict of scalars, each averaged over
    B, K, C (matching well_style_vrmse's own reduction: per-sample-per-channel
    stat computed over the spatial dims first, then averaged)."""
    spatial_dims = (-2, -1)
    mse = (pred - target).float().pow(2).mean(dim=spatial_dims)  # [B,K,C]
    rmse = mse.sqrt()

    var = target.float().std(dim=spatial_dims, unbiased=False).pow(2)
    vrmse = torch.sqrt(mse / (var + eps))

    tmax = target.float().amax(dim=spatial_dims)
    tmin = target.float().amin(dim=spatial_dims)
    nrmse_range = rmse / (tmax - tmin).clamp_min(eps)

    tmean_abs = target.float().abs().mean(dim=spatial_dims)
    nrmse_mean = rmse / tmean_abs.clamp_min(eps)

    # The Well's own published NRMSE (Eq. 6): per channel, ||u - u_hat||_2 /
    # (||u||_2 + eps), with ||.||_2 = RMS averaged over space -- note the
    # denominator is RMS of the RAW target (uncentered, includes its own
    # mean/DC term), not std like well_style_vrmse's var-normalization. For
    # any channel with a nonzero mean offset, RMS(target) > std(target), so
    # this reads systematically smaller than our vrmse there -- they are NOT
    # the same metric even though both are called "normalized RMSE".
    target_rms = (target.float().pow(2).mean(dim=spatial_dims) + eps).sqrt()
    nrmse_well = rmse / (target_rms + eps)

    return {
        "rmse": rmse.mean().item(),
        "vrmse": vrmse.mean().item(),
        "nrmse_range": nrmse_range.mean().item(),
        "nrmse_mean": nrmse_mean.mean().item(),
        "nrmse_well": nrmse_well.mean().item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", action="append", dest="tasks", default=None,
                    help="restrict to specific task(s); repeatable. Default: all tasks in the checkpoint.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="/app/data/datasets")
    p.add_argument("--config-root", default="/app/configs")
    p.add_argument("--task-set", default="core3", choices=["core3", "all14"])
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    tasks_cfg = build_tasks(args.data_root, args.config_root, task_set=args.task_set)
    channel_specs = union_channel_specs(tasks_cfg)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    T, K = ckpt["context_frames"], ckpt["rollout_steps"]
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
            print("NOTE: complex_term=True, but this checkpoint predates complex_proj being saved "
                  "(train_foundation.py's save block used to skip it -- now fixed). Reconstructed "
                  "FoundationModel.complex_proj falls back to its own zero-init, i.e. a no-op spectral "
                  "branch. The 'recomputed vrmse' column below versus the checkpoint's own stored "
                  "val_loss_by_task shows exactly how much that gap moves the numbers.")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded {args.checkpoint} (epoch={ckpt['epoch']}, stored val_loss_by_task={ckpt['val_loss_by_task']})")

    for t in task_names:
        cfg = tasks_cfg[t]
        param_choices = None
        if cfg.get("param_subset"):
            with open(cfg["param_subset"]) as f:
                param_choices = [tuple(c) for c in json.load(f)["combos"]]

        _, val_loader, _ = dl.create_param_dataloaders(
            cfg["data_dir"], batch_size=args.batch, context_frames=T, predict_frames=K,
            param_choices=param_choices, train_file_limit=cfg.get("train_file_limit"),
            val_file_limit=cfg.get("val_file_limit"), traj_limit=cfg.get("traj_limit"),
            pair_stride=cfg.get("pair_stride", 1), field_spec=cfg["field_spec"],
            traj_cache_capacity=cfg["traj_cache_capacity"],
        )

        agg = {"rmse": 0.0, "vrmse": 0.0, "nrmse_range": 0.0, "nrmse_mean": 0.0, "nrmse_well": 0.0}
        n_batches = 0
        with torch.no_grad():
            for xb, yb, _bparams in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb, field_spec=cfg["field_spec"], task=t, steps=K, return_initial_encode=False)
                stats = error_stats(pred, yb)
                for k in agg:
                    agg[k] += stats[k]
                n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in agg.items()}
        stored = ckpt["val_loss_by_task"].get(t)
        print(f"\n[{t}]  ({n_batches} val batches)")
        print(f"  recomputed vrmse:  {avg['vrmse']:.6f}   (checkpoint's stored val_loss_by_task: {stored})")
        print(f"  raw rmse:          {avg['rmse']:.6f}")
        print(f"  nrmse (range):     {avg['nrmse_range']:.6f}")
        print(f"  nrmse (mean):      {avg['nrmse_mean']:.6f}")
        print(f"  nrmse (Well Eq.6): {avg['nrmse_well']:.6f}")


if __name__ == "__main__":
    main()
