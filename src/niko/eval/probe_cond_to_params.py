"""Probe: how much does the learned context_cond_encoder embedding actually
know about the real physical params (Rayleigh, Prandtl)?

Loads an already-trained context_cond_* checkpoint (e.g.
context_cond_pooled_broad8's best, valid_loss 0.2285 -- see
EXPERIMENT_LOG.md / configs/context_cond_pooled_helmholtz.yaml), freezes the
whole model, runs context_cond_encoder(x_context) to get the learned cond
embedding it actually uses for FiLM conditioning, and trains a small MLP on
top that regresses that embedding back to (log10 Rayleigh, log10 Prandtl) --
supervised against the dataset's own ground-truth params, which were never
used anywhere in training the backbone itself.

This is purely a post-hoc interpretability probe, not part of the model: the
backbone is frozen throughout, only the tiny probe head trains. Since
broad8's ground truth is drawn from exactly 8 known (Rayleigh, Prandtl)
combos (configs/param_subsets/broad8.json), also reports accuracy after
snapping each raw prediction to its nearest neighbor among those 8 combos
(in transform space) -- the "rounding rule" the continuous MLP output alone
can't give you, since we already know real answers land on a small discrete
grid.
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

from models.build import build_model
from core.soap import SOAP
import dataloader as dl


def target_transform_space(bparams: torch.Tensor, transforms: list[str], dev) -> torch.Tensor:
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


def invert_transform(pred: torch.Tensor, transforms: list[str]) -> torch.Tensor:
    cols = []
    for i, t in enumerate(transforms):
        col = pred[:, i]
        cols.append(10.0 ** col if t == "log10" else col)
    return torch.stack(cols, dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="/app/checkpoints/context_cond_pooled_broad8/model_ep7_ctx6_roll6_vloss0.2285_lr1e-03_b8.pt")
    p.add_argument("--transforms", default="log10,log10")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--param-subset", default="/app/configs/param_subsets/broad8.json")
    p.add_argument("--save-dir", default="/app/checkpoints/cond_to_params_probe")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    transforms = args.transforms.split(",")

    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt["config"]
    T = ckpt["context_frames"]
    if "context_cond_encoder" not in cfg:
        raise ValueError(f"{args.checkpoint} was not trained with a context_cond_encoder -- nothing to probe.")

    model = build_model(copy.deepcopy(cfg)).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    cond_dim = cfg["context_cond_encoder"]["cond_dim"]
    print(f"Loaded {args.checkpoint} (val_loss={ckpt['val_loss']:.4f}, context_frames={T}, cond_dim={cond_dim}), backbone frozen.")

    probe = nn.Sequential(
        nn.Linear(cond_dim, args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, len(transforms)),
    ).to(dev)
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"Probe MLP params: {n_probe}")

    with open(args.param_subset) as f:
        combos = [tuple(c) for c in json.load(f)["combos"]]
    print(f"Restricting to param subset {args.param_subset}: {combos}")
    combo_t = torch.tensor(combos, dtype=torch.float32, device=dev)  # [n_combo, 2] raw
    combo_transform = target_transform_space(combo_t, transforms, dev)  # [n_combo, 2] transform space, for snapping

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir, batch_size=args.batch, context_frames=T, predict_frames=1,
        param_choices=combos,
    )
    print(f"Using real dataloader from {args.data_dir}")

    opt = SOAP(probe.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def get_cond(xb):
        with torch.no_grad():
            return model.context_cond_encoder(xb)

    for ep in range(1, args.epochs + 1):
        probe.train()
        running_loss_sum, running_batches = 0.0, 0
        for xb, _yb, bparams in train_loader:
            xb = xb.to(dev)
            target = target_transform_space(bparams, transforms, dev)
            cond = get_cond(xb)

            opt.zero_grad()
            pred = probe(cond)
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
                cond = get_cond(xb)
                pred = probe(cond)
                vloss = mse(pred, target)
                val_loss_sum += float(vloss.item())

                raw_pred = invert_transform(pred, transforms)
                rel_err_sum += ((raw_pred - bparams).abs() / bparams.abs().clamp_min(1e-12)).mean(dim=0)

                # Rounding rule: snap each prediction (in transform space) to its
                # nearest neighbor among the known combos, since ground truth is
                # always exactly one of these 8 points, never anything in between.
                dists = torch.cdist(pred, combo_transform)  # [B, n_combo]
                nearest = dists.argmin(dim=1)  # [B]
                snapped_raw = combo_t[nearest]  # [B, 2]
                snapped_transform = combo_transform[nearest]  # [B, 2]
                rel_err_sum_snapped += ((snapped_raw - bparams).abs() / bparams.abs().clamp_min(1e-12)).mean(dim=0)
                vloss_snapped_sum += float(mse(snapped_transform, target).item())

                true_nearest = torch.cdist(target, combo_transform).argmin(dim=1)
                exact_match += (nearest == true_nearest).sum().item()
                total += xb.shape[0]

                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        vloss_snapped_avg = vloss_snapped_sum / max(1, val_batches)
        rel_err_avg = (rel_err_sum / max(1, val_batches)).tolist()
        rel_err_snapped_avg = (rel_err_sum_snapped / max(1, val_batches)).tolist()
        acc = exact_match / max(1, total)
        print(f"Epoch {ep}   valid_loss (transform-space MSE): {vloss_avg:.6f}")
        print(f"Epoch {ep}   valid_loss_snapped (transform-space MSE after rounding to nearest combo): {vloss_snapped_avg:.6f}")
        print(f"Epoch {ep}   valid_mean_relative_error_by_param (raw MLP output): {[f'{e:.4f}' for e in rel_err_avg]}")
        print(f"Epoch {ep}   valid_mean_relative_error_by_param (snapped to nearest combo): {[f'{e:.4f}' for e in rel_err_snapped_avg]}")
        print(f"Epoch {ep}   valid_exact_combo_match_rate (snapped): {acc:.4f}  ({exact_match}/{total})")

        ckpt_out = save_dir / f"probe_ep{ep}_vloss{vloss_avg:.4f}_acc{acc:.4f}.pt"
        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "source_checkpoint": args.checkpoint,
                "cond_dim": cond_dim,
                "hidden_dim": args.hidden_dim,
                "transforms": transforms,
                "epoch": ep,
                "val_loss": vloss_avg,
                "val_loss_snapped": vloss_snapped_avg,
                "val_mean_relative_error_by_param": rel_err_avg,
                "val_mean_relative_error_by_param_snapped": rel_err_snapped_avg,
                "val_exact_combo_match_rate": acc,
            },
            ckpt_out,
        )
        print(f"Saved checkpoint: {ckpt_out}")


if __name__ == "__main__":
    main()
