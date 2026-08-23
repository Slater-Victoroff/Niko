"""Probe: for a trained FoundationModel checkpoint, how much does the shared
ENCODER LATENT (encoder(field_embedder(x_context)).grid, the actual per-pixel
state the operator rolls forward -- as opposed to context_cond_encoder's
separate pooled cond embedding, see probe_foundation_cond_to_params.py) know
about a given task's own ground-truth physical params?

Sibling of probe_foundation_cond_to_params.py: identical freeze/probe/snap
machinery, only the probed tensor differs (spatially-pooled encoder latent
grid instead of cond). Comparing the two tells you whether any param signal
lost from cond specifically is still present somewhere in the encoder's own
per-pixel state, or is genuinely not represented anywhere in the shared trunk.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import FieldEmbedder, state_dict_uses_batchnorm
from encoders.context_cond import ContextCondEncoder
from encoders.sequence_conv import SequenceConvEncoder
from core.soap import SOAP
import dataloader as dl
from training.train_foundation import build_tasks, union_channel_specs


TASK_TRANSFORMS = {
    "rayleigh_benard": ["log10", "log10"],
    "shear_flow": ["log10", "log10"],
    "active_matter": ["identity", "identity", "identity"],
}


def target_transform_space(bparams: torch.Tensor, transforms: list, dev) -> torch.Tensor:
    cols = []
    for i, t in enumerate(transforms):
        col = bparams[:, i].to(dev).float()
        if t == "identity":
            cols.append(col)
        elif t == "log10":
            cols.append(torch.log10(torch.clamp_min(col, 1e-30)))
        else:
            raise ValueError(f"Unknown transform: {t}")
    return torch.stack(cols, dim=-1)


def invert_transform(pred: torch.Tensor, transforms: list) -> torch.Tensor:
    cols = []
    for i, t in enumerate(transforms):
        col = pred[:, i]
        cols.append(10.0 ** col if t == "log10" else col)
    return torch.stack(cols, dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_TRANSFORMS.keys()))
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-root", default="/app/data/datasets")
    p.add_argument("--config-root", default="/app/configs")
    p.add_argument("--task-set", default="core3", choices=["core3", "all14"])
    p.add_argument("--save-dir", default="/app/checkpoints/foundation_latent_to_params_probe")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    transforms = TASK_TRANSFORMS[args.task]

    tasks = build_tasks(args.data_root, args.config_root, task_set=args.task_set)
    if args.task not in tasks:
        raise ValueError(f"'{args.task}' not in task_set='{args.task_set}': {list(tasks.keys())}")
    cfg = tasks[args.task]
    channel_specs = union_channel_specs(tasks)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    T = ckpt["context_frames"]
    canonical_dim, latent_dim = ckpt["canonical_dim"], ckpt["latent_dim"]
    hidden_dim, cond_dim = ckpt["hidden_dim"], ckpt["cond_dim"]
    block_kernel_size = ckpt.get("block_kernel_size", 7)

    use_bn = state_dict_uses_batchnorm(ckpt["field_embedder_state_dict"])
    field_embedder = FieldEmbedder(channel_specs, canonical_dim=canonical_dim, use_batchnorm=use_bn).to(dev)
    field_embedder.load_state_dict(ckpt["field_embedder_state_dict"])
    # context_cond_encoder isn't needed for this probe (the latent grid comes
    # from encoder alone), loaded only so the checkpoint's completeness is
    # implicitly verified, same rationale as the encoder load in
    # probe_foundation_cond_to_params.py.
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

    for m in (field_embedder, context_cond_encoder, encoder):
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    print(f"Loaded {args.checkpoint} (epoch={ckpt['epoch']}, val_loss_by_task={ckpt['val_loss_by_task']}), "
          f"probing task='{args.task}', latent_dim={latent_dim}, backbone frozen.")

    probe = nn.Sequential(
        nn.Linear(latent_dim, args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, len(transforms)),
    ).to(dev)
    print(f"Probe MLP params: {sum(p.numel() for p in probe.parameters())}")

    param_choices = None
    combo_t = combo_transform = None
    if cfg.get("param_subset"):
        with open(cfg["param_subset"]) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]
        combo_t = torch.tensor(param_choices, dtype=torch.float32, device=dev)
        combo_transform = target_transform_space(combo_t, transforms, dev)
        print(f"Task has a fixed param_subset ({len(param_choices)} combos) -> reporting snap-to-nearest-combo too.")

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        cfg["data_dir"], batch_size=args.batch, context_frames=T, predict_frames=1,
        param_choices=param_choices, train_file_limit=cfg.get("train_file_limit"),
        val_file_limit=cfg.get("val_file_limit"), traj_limit=cfg.get("traj_limit"),
        pair_stride=cfg.get("pair_stride", 1), field_spec=cfg["field_spec"],
        traj_cache_capacity=cfg["traj_cache_capacity"],
    )
    print(f"Using real dataloader from {cfg['data_dir']}")

    opt = SOAP(probe.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    def get_latent(xb):
        with torch.no_grad():
            x_context = field_embedder(xb, cfg["field_spec"])
            z = encoder(x_context).grid  # [B, latent_dim, H', W']
            return z.mean(dim=(-2, -1))  # [B, latent_dim]

    for ep in range(1, args.epochs + 1):
        probe.train()
        running_loss_sum, running_batches = 0.0, 0
        for xb, _yb, bparams in train_loader:
            xb = xb.to(dev)
            target = target_transform_space(bparams, transforms, dev)
            latent = get_latent(xb)

            opt.zero_grad()
            pred = probe(latent)
            loss = mse(pred, target)
            loss.backward()
            opt.step()

            running_loss_sum += float(loss.item())
            running_batches += 1

        avg_loss = running_loss_sum / max(1, running_batches)
        print(f"Epoch {ep}   train_loss: {avg_loss:.6f}")

        probe.eval()
        val_loss_sum, val_batches = 0.0, 0
        vloss_snapped_sum = 0.0
        rel_err_sum = torch.zeros(len(transforms), device=dev)
        rel_err_sum_snapped = torch.zeros(len(transforms), device=dev)
        exact_match, total = 0, 0
        with torch.no_grad():
            for xb, _yb, bparams in val_loader:
                xb = xb.to(dev)
                bparams = bparams.to(dev).float()
                target = target_transform_space(bparams, transforms, dev)
                latent = get_latent(xb)
                pred = probe(latent)
                vloss = mse(pred, target)
                val_loss_sum += float(vloss.item())

                raw_pred = invert_transform(pred, transforms)
                rel_err_sum += ((raw_pred - bparams).abs() / bparams.abs().clamp_min(1e-12)).mean(dim=0)

                if combo_transform is not None:
                    dists = torch.cdist(pred, combo_transform)
                    nearest = dists.argmin(dim=1)
                    snapped_raw = combo_t[nearest]
                    snapped_transform = combo_transform[nearest]
                    rel_err_sum_snapped += ((snapped_raw - bparams).abs() / bparams.abs().clamp_min(1e-12)).mean(dim=0)
                    vloss_snapped_sum += float(mse(snapped_transform, target).item())
                    true_nearest = torch.cdist(target, combo_transform).argmin(dim=1)
                    exact_match += (nearest == true_nearest).sum().item()

                total += xb.shape[0]
                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        rel_err_avg = (rel_err_sum / max(1, val_batches)).tolist()
        print(f"Epoch {ep}   valid_loss (transform-space MSE): {vloss_avg:.6f}")
        print(f"Epoch {ep}   valid_mean_relative_error_by_param (raw): {[f'{e:.4f}' for e in rel_err_avg]}")

        result = {
            "probe_state_dict": probe.state_dict(),
            "source_checkpoint": args.checkpoint,
            "task": args.task,
            "latent_dim": latent_dim,
            "transforms": transforms,
            "epoch": ep,
            "val_loss": vloss_avg,
            "val_mean_relative_error_by_param": rel_err_avg,
        }
        if combo_transform is not None:
            vloss_snapped_avg = vloss_snapped_sum / max(1, val_batches)
            rel_err_snapped_avg = (rel_err_sum_snapped / max(1, val_batches)).tolist()
            acc = exact_match / max(1, total)
            print(f"Epoch {ep}   valid_loss_snapped (transform-space MSE after rounding to nearest combo): {vloss_snapped_avg:.6f}")
            print(f"Epoch {ep}   valid_mean_relative_error_by_param (snapped): {[f'{e:.4f}' for e in rel_err_snapped_avg]}")
            print(f"Epoch {ep}   valid_exact_combo_match_rate (snapped): {acc:.4f}  ({exact_match}/{total})")
            result.update(
                val_loss_snapped=vloss_snapped_avg,
                val_mean_relative_error_by_param_snapped=rel_err_snapped_avg,
                val_exact_combo_match_rate=acc,
            )

        ckpt_out = save_dir / f"probe_ep{ep}_vloss{vloss_avg:.4f}.pt"
        torch.save(result, ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")


if __name__ == "__main__":
    main()
