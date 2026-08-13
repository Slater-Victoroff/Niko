import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure `src/niko` is on sys.path so this works from project root:
# python src/niko/eval/eval_rb_checkpoint.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model
from training.losses import well_style_vrmse
from core.states import Params
import dataloader as dl


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    ckpt = Path(checkpoint)
    if ckpt.exists():
        return ckpt.resolve()

    # Convenience: allow passing only filename for /app/checkpoints in container.
    container_ckpt = Path("/app/checkpoints") / checkpoint
    if container_ckpt.exists():
        return container_ckpt.resolve()

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}. Also checked {container_ckpt}."
    )


def _resolve_test_subdir(base_dir: Path, requested_subdir: str) -> str:
    requested_path = base_dir / requested_subdir
    if requested_path.is_dir():
        return requested_subdir

    # Friendly fallback for setups that use valid/ instead of test/
    if requested_subdir == "test" and (base_dir / "valid").is_dir():
        print("Requested test split not found; falling back to valid split.")
        return "valid"

    raise FileNotFoundError(
        f"Test split folder not found: {requested_path}. "
        f"Available under {base_dir}: {[p.name for p in base_dir.iterdir() if p.is_dir()]}"
    )


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    explicit_config_path: Path | None = None,
):
    ckpt_obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
        ckpt_cfg = ckpt_obj.get("config")
        metadata = {k: v for k, v in ckpt_obj.items() if k != "model_state_dict"}
    else:
        state_dict = ckpt_obj
        ckpt_cfg = None
        metadata = {}

    # Explicit --config always wins.
    if explicit_config_path is not None:
        print(f"Loading config from: {explicit_config_path}")
        with open(explicit_config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt_cfg
        if cfg is not None and "name" not in cfg.get("param_encoder", {}):
            # Config was saved after build_model mutated it — reload from disk.
            cfg = None

        if cfg is None:
            # Try to find the original YAML from the path stored in the checkpoint.
            config_path_str = metadata.get("config_path")
            if config_path_str is not None:
                config_path = Path(config_path_str)
                search_roots = [
                    Path("/app"),
                    checkpoint_path.parents[2],
                    checkpoint_path.parents[3],
                ]
                resolved = None
                if config_path.exists():
                    resolved = config_path
                else:
                    for root in search_roots:
                        for candidate in (root / config_path.name, root / "configs" / config_path.name):
                            if candidate.exists():
                                resolved = candidate
                                break
                        if resolved:
                            break
                if resolved:
                    print(f"Reloading config from disk: {resolved}")
                    with open(resolved) as f:
                        cfg = yaml.safe_load(f)

        if cfg is None:
            raise ValueError(
                "No usable model config found. Pass --config /path/to/your.yaml to specify it explicitly."
            )

    # build_model mutates nested dicts via pop("name"), so use a copy
    try:
        model = build_model(copy.deepcopy(cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to build model from config: {e}") from e

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg, metadata


def evaluate_checkpoint(
    model,
    test_loader,
    device: torch.device,
    rollout_steps: int,
    max_batches: int | None = None,
    log_interval: int = 20,
):
    total_vrmse = 0.0
    num_batches = 0
    channel_vrmse_sum = torch.zeros(4, dtype=torch.float64)

    total_batches = None
    try:
        total_batches = len(test_loader)
    except Exception:
        total_batches = None

    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                xb, yb, bparams = batch
            else:
                xb, yb = batch
                bparams = None

            xb = xb.to(device)
            yb = yb.to(device)

            if bparams is None:
                # RB uses 2 scalar params (Rayleigh, Prandtl).
                bparams = torch.zeros((xb.shape[0], 2), dtype=torch.float32, device=device)
            else:
                bparams = bparams.to(device=device, dtype=torch.float32)

            params = Params(values=bparams)
            pred = model(xb, steps=rollout_steps, params=params, return_initial_encode=False)

            vrmse_bkc = well_style_vrmse(pred, yb)  # [B, K, C]
            batch_vrmse = vrmse_bkc.mean()
            batch_channel_vrmse = vrmse_bkc.mean(dim=(0, 1))

            total_vrmse += float(batch_vrmse.item())
            channel_vrmse_sum += batch_channel_vrmse.detach().cpu().to(torch.float64)
            num_batches += 1

            if log_interval > 0 and (num_batches % log_interval == 0 or (total_batches is not None and num_batches == total_batches)):
                elapsed = time.time() - start_time
                running_avg = total_vrmse / max(1, num_batches)
                avg_batch_s = elapsed / max(1, num_batches)
                if total_batches is not None:
                    remaining = max(total_batches - num_batches, 0)
                    eta_s = remaining * avg_batch_s
                    print(
                        f"Eval batch {num_batches}/{total_batches} | "
                        f"running_vrmse {running_avg:.6f} | elapsed {elapsed:.1f}s | ETA {eta_s:.1f}s"
                    )
                else:
                    print(
                        f"Eval batch {num_batches} | "
                        f"running_vrmse {running_avg:.6f} | elapsed {elapsed:.1f}s"
                    )

    if num_batches == 0:
        return float("inf"), [float("inf")] * 4

    avg_vrmse = total_vrmse / num_batches
    avg_channel_vrmse = (channel_vrmse_sum / num_batches).tolist()
    return avg_vrmse, avg_channel_vrmse


def main():
    p = argparse.ArgumentParser(
        description="Evaluate an RB checkpoint against the RB test split.",
    )
    p.add_argument("checkpoint", help="Checkpoint path, or checkpoint filename under /app/checkpoints")
    p.add_argument(
        "--config",
        default=None,
        help="Path to the training YAML. Required for checkpoints that don't embed config_path.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device like cuda:0 or cpu. Default is cuda:0 if available else cpu.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after this many eval batches instead of the full split -- for a quick smoke test",
    )

    args = p.parse_args()

    if args.device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    explicit_config = Path(args.config) if args.config else None

    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg, meta = _load_model_from_checkpoint(ckpt_path, dev, explicit_config)

    context_frames = int(meta.get("context_frames", cfg["encoder"].get("context_frames", 1)))
    rollout_steps = int(meta.get("rollout_steps", cfg.get("rollout_steps", 1)))

    data_dir = Path("/app/data/datasets/rayleigh_benard/data")
    split = _resolve_test_subdir(data_dir, "test")
    print(f"Evaluating split: {split} (base: {data_dir})")

    _, test_loader, _ = dl.create_param_dataloaders(
        str(data_dir),
        batch_size=8,
        num_workers=0,
        context_frames=context_frames,
        predict_frames=rollout_steps,
        train_file_limit=None,
        val_file_limit=None,
        train_subdir=split,
        valid_subdir=split,
        shuffle_train=False,
        shuffle_val=False,
        use_all_params=True,
        return_params=True,
    )

    avg_vrmse, avg_channel_vrmse = evaluate_checkpoint(
        model=model,
        test_loader=test_loader,
        device=dev,
        rollout_steps=rollout_steps,
        max_batches=args.max_batches,
        log_interval=5,
    )

    print("=== Evaluation Results ===")
    print(f"checkpoint: {ckpt_path}")
    if "epoch" in meta:
        print(f"epoch: {meta['epoch']}")
    if "val_loss" in meta:
        print(f"saved_val_loss: {meta['val_loss']}")
    print(f"test_vrmse_mean: {avg_vrmse:.6f}")
    print(
        "test_vrmse_by_channel "
        "[pressure, buoyancy, velocity_x, velocity_y]: "
        f"[{avg_channel_vrmse[0]:.6f}, {avg_channel_vrmse[1]:.6f}, {avg_channel_vrmse[2]:.6f}, {avg_channel_vrmse[3]:.6f}]"
    )


if __name__ == "__main__":
    main()
