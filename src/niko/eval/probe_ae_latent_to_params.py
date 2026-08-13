"""Probe: does the frozen single-frame autoencoder's RAW latent already
encode physical params (Rayleigh, Prandtl), before any of the LatentContextCondEncoder
machinery (refine_trunk/head) gets a chance to reshape it?

Loads a train_autoencoder_baseline.py (--context-frames 1) checkpoint, freezes
its encoder, runs it independently over each frame of a normal ctx6 context
window, spatially pools each frame's latent grid to a per-frame vector, mean-
pools over the T frames (same two-stage pool as PooledContextCondEncoder /
LatentContextCondEncoder, minus the learnable refine_trunk), and trains a
tiny linear probe on top regressing to (log10 Rayleigh, log10 Prandtl) --
supervised against the dataset's own ground-truth params. Purely a diagnostic
on the pretrained latent space itself, not part of any model.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.sequence_conv import SequenceConvEncoder
from core.soap import SOAP
import dataloader as dl


def target_transform_space(bparams: torch.Tensor, dev) -> torch.Tensor:
    return torch.log10(bparams.to(dev).float().clamp_min(1e-30))


def invert_transform(pred: torch.Tensor) -> torch.Tensor:
    return 10.0 ** pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae-checkpoint", required=True)
    p.add_argument("--context-frames", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--param-subset", default="/app/configs/param_subsets/broad8.json")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ckpt = torch.load(args.ae_checkpoint, map_location=dev)
    ae_cfg = ckpt["config"]["encoder"]
    ae_latent_dim = ae_cfg["latent_dim"]
    ae_hidden_dim = ae_cfg["hidden_dim"]
    in_channels = ae_cfg["in_channels"]

    frame_encoder = SequenceConvEncoder(
        in_channels=in_channels, context_frames=1, latent_dim=ae_latent_dim, hidden_dim=ae_hidden_dim,
    ).to(dev)
    frame_encoder.load_state_dict(ckpt["encoder_state_dict"])
    frame_encoder.eval()
    for param in frame_encoder.parameters():
        param.requires_grad_(False)
    print(f"Loaded frozen frame_encoder from {args.ae_checkpoint} (val_loss={ckpt['val_loss']:.4f}, epoch={ckpt['epoch']})")

    probe = nn.Sequential(
        nn.Linear(ae_latent_dim, args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, 2),
    ).to(dev)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters())}")

    with open(args.param_subset) as f:
        combos = [tuple(c) for c in json.load(f)["combos"]]
    print(f"Restricting to param subset {args.param_subset}: {combos}")

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir, batch_size=args.batch, context_frames=args.context_frames, predict_frames=1,
        param_choices=combos,
    )
    print(f"Using real dataloader from {args.data_dir}")

    opt = SOAP(probe.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    def get_pooled_latent(xb):
        b, t, c, h, w = xb.shape
        frames = xb.reshape(b * t, 1, c, h, w)
        with torch.no_grad():
            z = frame_encoder(frames).grid  # [B*T, latent_dim, H', W']
        pooled_spatial = z.mean(dim=(-2, -1)).view(b, t, -1)
        return pooled_spatial.mean(dim=1)  # [B, latent_dim]

    for ep in range(1, args.epochs + 1):
        probe.train()
        running_loss_sum, running_batches = 0.0, 0
        for xb, _yb, bparams in train_loader:
            xb = xb.to(dev)
            target = target_transform_space(bparams, dev)
            latent = get_pooled_latent(xb)

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
        rel_err_sum = torch.zeros(2, device=dev)
        with torch.no_grad():
            for xb, _yb, bparams in val_loader:
                xb = xb.to(dev)
                bparams = bparams.to(dev).float()
                target = target_transform_space(bparams, dev)
                latent = get_pooled_latent(xb)
                pred = probe(latent)
                vloss = mse(pred, target)
                val_loss_sum += float(vloss.item())

                raw_pred = invert_transform(pred)
                rel_err_sum += ((raw_pred - bparams).abs() / bparams.abs().clamp_min(1e-12)).mean(dim=0)
                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        rel_err_avg = (rel_err_sum / max(1, val_batches)).tolist()
        print(f"Epoch {ep}   valid_loss (transform-space MSE): {vloss_avg:.6f}")
        print(f"Epoch {ep}   valid_mean_relative_error_by_param [Rayleigh, Prandtl]: {[f'{e:.4f}' for e in rel_err_avg]}")


if __name__ == "__main__":
    main()
