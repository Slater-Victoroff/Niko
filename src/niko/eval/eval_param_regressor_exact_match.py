"""One-off post-hoc eval: exact-combo match rate for an already-trained
ContextParamRegressor checkpoint (train_param_inference.py), computed the
same way probe_cond_to_params.py/probe_encoder_latent_to_params.py do it --
train_param_inference.py itself never computed this diagnostic, so this
reruns just the val set through the frozen trained regressor and adds the
snapping-to-nearest-known-combo comparison after the fact. No training here.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.param_inference import ContextParamRegressor
import dataloader as dl


def target_transform_space(bparams: torch.Tensor, transforms: list[str], dev) -> torch.Tensor:
    cols = []
    for i, t in enumerate(transforms):
        col = bparams[:, i].to(dev).float()
        cols.append(torch.log10(torch.clamp_min(col, 1e-30)) if t == "log10" else col)
    return torch.stack(cols, dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--param-subset", default="/app/configs/param_subsets/broad8.json")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    transforms = ckpt["transforms"]
    T = ckpt["context_frames"]

    model = ContextParamRegressor(
        in_channels=ckpt["in_channels"], context_frames=T, transforms=transforms,
        hidden_dim=ckpt["hidden_dim"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint} (val_loss={ckpt['val_loss']:.4f}, epoch={ckpt['epoch']}), frozen.")

    with open(args.param_subset) as f:
        combos = [tuple(c) for c in json.load(f)["combos"]]
    combo_t = torch.tensor(combos, dtype=torch.float32, device=dev)
    combo_transform = target_transform_space(combo_t, transforms, dev)

    _, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir, batch_size=args.batch, context_frames=T, predict_frames=1,
        param_choices=combos,
    )

    exact_match, total = 0, 0
    mse_sum, mse_snapped_sum, batches = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, _yb, bparams in val_loader:
            xb = xb.to(dev)
            bparams = bparams.to(dev).float()
            target = target_transform_space(bparams, transforms, dev)
            pred = model(xb)
            mse_sum += float(((pred - target) ** 2).mean().item())

            dists = torch.cdist(pred, combo_transform)
            nearest = dists.argmin(dim=1)
            snapped_transform = combo_transform[nearest]
            mse_snapped_sum += float(((snapped_transform - target) ** 2).mean().item())

            true_nearest = torch.cdist(target, combo_transform).argmin(dim=1)
            exact_match += (nearest == true_nearest).sum().item()
            total += xb.shape[0]
            batches += 1

    acc = exact_match / max(1, total)
    print(f"valid_loss (transform-space MSE): {mse_sum / max(1, batches):.6f}")
    print(f"valid_loss_snapped (transform-space MSE after rounding to nearest combo): {mse_snapped_sum / max(1, batches):.6f}")
    print(f"valid_exact_combo_match_rate (snapped): {acc:.4f}  ({exact_match}/{total})")


if __name__ == "__main__":
    main()
