"""Standalone supervised training for ContextParamRegressor: predicts the
physical simulation parameters (Rayleigh/Prandtl, etc.) directly from the raw
context frames, via direct MSE regression against the dataset's own
ground-truth params -- decoupled from the main dynamics model's training
entirely (no encoder/operator/decoder involved here, no rollout).

The point is inference-time convenience: once trained, this replaces the
need to pass ground-truth params into a dynamics model at test time --
splice ContextParamRegressor.to_params(context) in wherever the dataset's
own Params object would otherwise be used.

Each param is regressed in its own *transform* space (see
ContextParamRegressor), matching whichever param_encoder the target
dynamics model uses -- pass --transforms to match that config's own list
(e.g. "log10,log10" for rayleigh_benard/shear_flow's Reynolds+Schmidt,
"identity,identity,identity" for active_matter's L/zeta/alpha).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.param_inference import ContextParamRegressor, PooledSequenceParamRegressor
from core.soap import SOAP
import dataloader as dl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-frames", type=int, required=True,
                    help="frames per sample fed to the model -- for --architecture pooled this can be the "
                         "whole trajectory (e.g. 199 for rayleigh_benard's 200-frame trajectories, since 1 "
                         "frame is reserved as the unused dataloader target)")
    p.add_argument("--architecture", default="stacked", choices=["stacked", "pooled"],
                    help="stacked = ContextParamRegressor (frames stacked as channels, fixed context_frames); "
                         "pooled = PooledSequenceParamRegressor (shared per-frame trunk + mean pool over time, "
                         "works at any context_frames)")
    p.add_argument("--transforms", required=True,
                    help="comma-separated per-param transform list, e.g. log10,log10")
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="soap", choices=["adam", "soap"])
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--param-subset", default=None,
                    help="path to a JSON file like {\"combos\": [[1e7, 1.0], [1e8, 1.0]]}")
    p.add_argument("--field-spec", default=None,
                    help="name of a field_spec constant in dataloader.py (e.g. active_matter, shear_flow); "
                         "omit for rayleigh_benard's default 4-channel layout")
    p.add_argument("--save-dir", default="/app/checkpoints/param_inference")
    p.add_argument("--log-interval", type=int, default=200)
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    transforms = args.transforms.split(",")
    T = args.context_frames

    field_spec = None
    if args.field_spec:
        field_spec_map = {
            "rayleigh_benard": dl.RB_FIELD_SPEC,
            "active_matter": dl.ACTIVE_MATTER_FIELD_SPEC,
            "shear_flow": dl.SHEAR_FLOW_FIELD_SPEC,
        }
        field_spec = field_spec_map[args.field_spec]

    if args.architecture == "pooled":
        model = PooledSequenceParamRegressor(
            in_channels=args.in_channels, transforms=transforms, hidden_dim=args.hidden_dim,
        ).to(dev)
    else:
        model = ContextParamRegressor(
            in_channels=args.in_channels, context_frames=T, transforms=transforms, hidden_dim=args.hidden_dim,
        ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__} params: total={n_params}, transforms={transforms}, context_frames={T}")

    param_choices = None
    combo_t = combo_transform = None
    if args.param_subset:
        with open(args.param_subset) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]
        print(f"Restricting to param subset {args.param_subset}: {param_choices}")

    dl_kwargs = dict(
        batch_size=args.batch,
        context_frames=T,
        predict_frames=1,
        num_workers=args.num_workers,
        param_choices=param_choices,
    )
    if field_spec is not None:
        dl_kwargs["field_spec"] = field_spec

    train_loader, val_loader, _ = dl.create_param_dataloaders(args.data_dir, **dl_kwargs)
    print(f"Using real dataloader from {args.data_dir}")

    if args.optimizer == "soap":
        opt = SOAP(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def target_transform_space(bparams: torch.Tensor) -> torch.Tensor:
        """Ground-truth raw params [B, P] -> each column in its own
        transform space, matching what the model predicts."""
        cols = []
        for i, t in enumerate(transforms):
            col = bparams[:, i].to(dev).float()
            if t == "identity":
                cols.append(col)
            elif t == "log10":
                cols.append(torch.log10(torch.clamp_min(col, 1e-30)))
            elif t == "log":
                cols.append(torch.log(torch.clamp_min(col, 1e-30)))
            elif t == "signed_log10":
                cols.append(torch.sign(col) * torch.log10(1.0 + torch.abs(col)))
            else:
                raise ValueError(f"Unknown transform: {t}")
        return torch.stack(cols, dim=-1)

    mse = nn.MSELoss()

    if param_choices is not None:
        combo_t = torch.tensor(param_choices, dtype=torch.float32, device=dev)
        combo_transform = target_transform_space(combo_t)
        print(f"Also reporting snap-to-nearest-combo diagnostic against {len(param_choices)} combos.")

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss_sum, running_batches = 0.0, 0
        interval_batches, interval_start = 0, time.perf_counter()
        tb = len(train_loader)

        for batch_idx, (xb, _yb, bparams) in enumerate(train_loader):
            xb = xb.to(dev)
            target = target_transform_space(bparams)

            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss_sum += float(loss.item())
            running_batches += 1
            interval_batches += 1

            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - interval_start) * 1000.0
                time_per_batch_ms = elapsed_ms / max(1, interval_batches)
                print(f"Epoch {ep}  batch {batch_idx}/{tb}  loss {loss.item():.6f}  time/batch {time_per_batch_ms:.2f}ms")
                interval_batches, interval_start = 0, time.perf_counter()

        avg_loss = running_loss_sum / max(1, running_batches)
        print(f"Epoch {ep}   train_loss: {avg_loss:.6f}")

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        # Also track raw-space relative error per param -- transform-space
        # MSE alone doesn't tell you "how close in physical units", and raw
        # Rayleigh/Prandtl-scale numbers are what this is ultimately for.
        rel_err_sum = torch.zeros(len(transforms), device=dev)
        vloss_snapped_sum = 0.0
        rel_err_sum_snapped = torch.zeros(len(transforms), device=dev)
        exact_match, total = 0, 0
        with torch.no_grad():
            for xb, _yb, bparams in val_loader:
                xb = xb.to(dev)
                target = target_transform_space(bparams)
                pred = model(xb)
                vloss = mse(pred, target)
                val_loss_sum += float(vloss.item())

                raw_pred = model.invert_transform(pred)
                raw_true = bparams.to(dev).float()
                rel_err_sum += ((raw_pred - raw_true).abs() / raw_true.abs().clamp_min(1e-12)).mean(dim=0)

                if combo_transform is not None:
                    dists = torch.cdist(pred, combo_transform)
                    nearest = dists.argmin(dim=1)
                    snapped_raw = combo_t[nearest]
                    snapped_transform = combo_transform[nearest]
                    rel_err_sum_snapped += ((snapped_raw - raw_true).abs() / raw_true.abs().clamp_min(1e-12)).mean(dim=0)
                    vloss_snapped_sum += float(mse(snapped_transform, target).item())
                    true_nearest = torch.cdist(target, combo_transform).argmin(dim=1)
                    exact_match += (nearest == true_nearest).sum().item()
                total += xb.shape[0]

                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        rel_err_avg = (rel_err_sum / max(1, val_batches)).tolist()
        print(f"Epoch {ep}   valid_loss (transform-space MSE): {vloss_avg:.6f}")
        print(f"Epoch {ep}   valid_mean_relative_error_by_param: {[f'{e:.4f}' for e in rel_err_avg]}")

        result = {
            "model_state_dict": model.state_dict(),
            "architecture": args.architecture,
            "in_channels": args.in_channels,
            "context_frames": T,
            "transforms": transforms,
            "hidden_dim": args.hidden_dim,
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

        ckpt = save_dir / f"paraminfer_{args.architecture}_ep{ep}_ctx{T}_vloss{vloss_avg:.4f}.pt"
        torch.save(result, ckpt)
        print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
