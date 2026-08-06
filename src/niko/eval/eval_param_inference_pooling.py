"""Cheap static test of the "average over more of the sequence" hypothesis,
using the ALREADY-TRAINED ContextParamRegressor checkpoint (6-frame windows,
stacked-as-channels) -- no new training. The val dataloader already yields
many overlapping 6-frame windows per trajectory (stride 1, ~194 windows for a
200-frame rayleigh_benard trajectory); this just runs the existing model over
every one of them, groups predictions by their shared ground-truth param
combo, and compares:

  - single-window: relative error of one window's prediction (what
    train_param_inference.py already reports)
  - pooled: relative error of the MEAN prediction across every window that
    shares that combo (mean taken in transform-space, then inverted, plus a
    raw-space mean for comparison)

If pooling helps a lot, that's evidence the 6-frame model's per-window
predictions are noisy-but-unbiased -- averaging more of them (whether via
this static trick or PooledSequenceParamRegressor's learned pooling) should
close much of the gap. If it doesn't help, the error is more likely a
systematic/biased limitation of what's inferable from a short window, not
noise that averages out.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.param_inference import ContextParamRegressor
import dataloader as dl


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    ckpt = Path(checkpoint)
    if ckpt.exists():
        return ckpt.resolve()
    container_ckpt = Path("/app/checkpoints") / checkpoint
    if container_ckpt.exists():
        return container_ckpt.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Also checked {container_ckpt}.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--param-subset", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ckpt = torch.load(_resolve_checkpoint_path(args.checkpoint), map_location=dev)
    T = ckpt["context_frames"]
    transforms = ckpt["transforms"]
    model = ContextParamRegressor(
        in_channels=ckpt["in_channels"], context_frames=T, transforms=transforms, hidden_dim=ckpt["hidden_dim"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint}: context_frames={T}, transforms={transforms}, "
          f"trained val_loss={ckpt.get('val_loss')}")

    param_choices = None
    if args.param_subset:
        with open(args.param_subset) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]

    _, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir,
        batch_size=args.batch,
        context_frames=T,
        predict_frames=1,
        num_workers=args.num_workers,
        param_choices=param_choices,
    )

    # Collect every window's raw-space prediction, grouped by its ground-truth
    # combo (rounded to identify the combo; doesn't affect the actual values
    # used for error computation).
    preds_by_combo = defaultdict(list)
    n_windows = 0
    with torch.no_grad():
        for xb, _yb, bparams in val_loader:
            xb = xb.to(dev)
            pred = model(xb)  # transform-space
            bparams = bparams.to(dev).float()
            for i in range(xb.shape[0]):
                key = tuple(round(v, 6) for v in bparams[i].tolist())
                preds_by_combo[key].append(pred[i].detach().cpu())
            n_windows += xb.shape[0]

    print(f"Collected predictions for {len(preds_by_combo)} combo(s) from {n_windows} total windows.")

    def invert(t: torch.Tensor) -> torch.Tensor:
        return model.invert_transform(t.unsqueeze(0)).squeeze(0)

    single_rel_err_sum = torch.zeros(len(transforms))
    pooled_transform_rel_err_sum = torch.zeros(len(transforms))
    pooled_raw_rel_err_sum = torch.zeros(len(transforms))
    n_combos = 0

    # Transform-space MSE, directly comparable to train_param_inference.py's
    # own "valid_loss (transform-space MSE)" number (0.0435 at epoch 10) --
    # that number is nn.MSELoss() over EVERY window individually, so the
    # single-window figure here should reproduce it; the pooled figure is
    # the same metric computed on the one pooled prediction per combo instead.
    single_sq_err_sum = 0.0
    single_sq_err_n = 0
    pooled_sq_err_sum = 0.0

    for combo, preds in preds_by_combo.items():
        true_raw = torch.tensor(combo)
        log_target = torch.log10(true_raw)  # both transforms here are log10
        stacked = torch.stack(preds, dim=0)  # [n_windows_for_combo, n_params], transform-space

        # single-window baseline: mean relative error across all individual windows
        single_raw = torch.stack([invert(p) for p in stacked], dim=0)
        single_rel_err = ((single_raw - true_raw).abs() / true_raw.abs().clamp_min(1e-12)).mean(dim=0)

        single_sq_err_sum += ((stacked - log_target) ** 2).sum().item()
        single_sq_err_n += stacked.numel()

        # pooled, mean taken in transform-space then inverted once
        pooled_transform_mean = stacked.mean(dim=0)
        pooled_transform_raw = invert(pooled_transform_mean)
        pooled_transform_rel_err = (pooled_transform_raw - true_raw).abs() / true_raw.abs().clamp_min(1e-12)
        pooled_sq_err_sum += ((pooled_transform_mean - log_target) ** 2).sum().item()

        # pooled, mean taken directly in raw space (for comparison)
        pooled_raw_mean = single_raw.mean(dim=0)
        pooled_raw_rel_err = (pooled_raw_mean - true_raw).abs() / true_raw.abs().clamp_min(1e-12)

        print(f"combo={combo}  n_windows={len(preds)}  "
              f"single_window_rel_err={single_rel_err.tolist()}  "
              f"pooled(transform-space)_rel_err={pooled_transform_rel_err.tolist()}  "
              f"pooled(raw-space)_rel_err={pooled_raw_rel_err.tolist()}")

        single_rel_err_sum += single_rel_err
        pooled_transform_rel_err_sum += pooled_transform_rel_err
        pooled_raw_rel_err_sum += pooled_raw_rel_err
        n_combos += 1

    print()
    print(f"Mean over {n_combos} combos:")
    print(f"  single-window relative error:            {(single_rel_err_sum / n_combos).tolist()}")
    print(f"  pooled (transform-space mean) rel error: {(pooled_transform_rel_err_sum / n_combos).tolist()}")
    print(f"  pooled (raw-space mean) rel error:       {(pooled_raw_rel_err_sum / n_combos).tolist()}")
    print()
    print(f"Transform-space MSE (directly comparable to training's valid_loss={ckpt.get('val_loss')}):")
    print(f"  single-window MSE (every window individually): {single_sq_err_sum / single_sq_err_n:.6f}")
    print(f"  pooled MSE (one pooled prediction per combo):  {pooled_sq_err_sum / (n_combos * len(transforms)):.6f}")


if __name__ == "__main__":
    main()
