import argparse
import sys
import time
from pathlib import Path
import yaml
import torch

# Ensure `src` is on sys.path so `python src/training/train.py` works from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model
from training.losses import vnmse_rollout_loss, well_style_vrmse
from core.states import Params
import dataloader as dl


def train(
    cfg_path: str,
    data_dir: str,
    epochs: int,
    batch: int,
    device: str,
    lr: float = 1e-4,
    save_dir: str = "checkpoints",
    num_workers: int = 0,
    train_file_limit: int | None = None,
    steps_per_epoch: int | None = None,
    val_file_limit: int | None = None,
    log_interval: int = 10,
    debug_timing: bool = False,
):
    torch.cuda.set_device(device)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    dev = torch.device(device)
    model.to(dev)

    enc = cfg["encoder"]
    in_ch = enc.get("in_channels", 2)
    T = enc.get("context_frames", 3)
    K = cfg.get("rollout_steps", 1)

    if in_ch != 4:
        raise ValueError(
            "Rayleigh-Benard training requires encoder.in_channels=4 for physical channels "
            "[pressure, buoyancy, velocity_x, velocity_y]. "
            f"Got in_channels={in_ch}."
        )

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        data_dir,
        batch_size=batch,
        context_frames=T,
        predict_frames=K,
        num_workers=num_workers,
        train_file_limit=train_file_limit,
        val_file_limit=val_file_limit,
    )
    print(f"Using real dataloader from {data_dir}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    total_batches = len(train_loader)

    initial_anchor = cfg.get("anchor_target", False)

    for ep in range(1, epochs + 1):
        model.train()
        running_loss_sum, running_batches = 0.0, 0
        interval_batches = 0
        interval_start = time.perf_counter()
        tb = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            iter_wall_start = time.perf_counter()

            batch_start = torch.cuda.Event(enable_timing=True) if debug_timing else None
            if batch_start is not None:
                batch_start.record()

            xb, yb, bparams = batch_data
            xb = xb.to(dev)
            yb = yb.to(dev)
            params_obj = Params(values=bparams.to(dev))

            if debug_timing:
                batch_to_device_end = torch.cuda.Event(enable_timing=True)
                batch_to_device_end.record()
                torch.cuda.synchronize()
                batch_load_time = batch_start.elapsed_time(batch_to_device_end) if batch_start is not None else 0.0

                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_start.record()
            else:
                fwd_start = None
                batch_load_time = 0.0

            opt.zero_grad()
            pred = model(xb, steps=K, params=params_obj, debug_timing=debug_timing, return_initial_encode=initial_anchor)
            
            if initial_anchor:
                initial, pred = pred
                initial_target = xb[:, -1, ...]
                initial_loss = well_style_vrmse(initial.unsqueeze(1), initial_target.unsqueeze(1)).mean()
                rollout_loss = well_style_vrmse(pred, yb).mean()
                initial_weight = 1 / K
                loss = initial_weight * initial_loss + rollout_loss
            else:
                loss = well_style_vrmse(pred, yb).mean()

            if debug_timing:
                loss_start = torch.cuda.Event(enable_timing=True)
                loss_start.record()

            loss.backward()
            
            if debug_timing:
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_start.record()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if debug_timing:
                opt_end = torch.cuda.Event(enable_timing=True)
                opt_end.record()
                torch.cuda.synchronize()

            iter_wall_ms = (time.perf_counter() - iter_wall_start) * 1000.0

            loss_item = float(loss.item())
            running_loss_sum += loss_item
            running_batches += 1
            interval_batches += 1

            if batch_idx > 0 and batch_idx % log_interval == 0:
                sample_vrmse = well_style_vrmse(pred.detach(), yb.detach()).mean().item()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                interval_elapsed_ms = (time.perf_counter() - interval_start) * 1000.0
                time_per_batch_ms = interval_elapsed_ms / max(1, interval_batches)

                if debug_timing:
                    fwd_time_ms = fwd_start.elapsed_time(loss_start)
                    bwd_time_ms = loss_start.elapsed_time(backward_start)
                    opt_time_ms = backward_start.elapsed_time(opt_end)
                    total_iter_ms = fwd_start.elapsed_time(opt_end)
                    non_model_ms = max(0.0, iter_wall_ms - total_iter_ms)
                    print(
                        f"Epoch {ep}  batch {batch_idx}/{tb}  "
                        f"loss {loss_item:.6f}  "
                        f"vrmse {sample_vrmse:.6f}  "
                        f"[to_dev={batch_load_time:.1f}ms  fwd+loss={fwd_time_ms:.1f}ms  bwd={bwd_time_ms:.1f}ms  "
                        f"opt={opt_time_ms:.1f}ms  model_total={total_iter_ms:.1f}ms  other={non_model_ms:.1f}ms  "
                        f"iter={iter_wall_ms:.1f}ms  wall/batch={time_per_batch_ms:.1f}ms]"
                    )
                else:
                    print(
                        f"Epoch {ep}  batch {batch_idx}/{tb}  "
                        f"loss {loss_item:.6f}  "
                        f"vrmse {sample_vrmse:.6f}  time/batch {time_per_batch_ms:.2f}ms"
                    )
                interval_batches = 0
                interval_start = time.perf_counter()

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

        avg_loss = running_loss_sum / max(1, running_batches)
        print(f"Epoch {ep}   train_loss: {avg_loss:.6f}")

        # validation pass (no grad): average vrmse over validation loader
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for vbatch in val_loader:
                vxb, vyb, vparams = vbatch

                vxb = vxb.to(dev)
                vyb = vyb.to(dev)
                vp = Params(values=vparams.to(dev))

                vpred = model(vxb, steps=K, params=vp, return_initial_encode=False)
                # use variance-normalized MSE (VRMSE) as validation metric
                vloss = well_style_vrmse(vpred, vyb.to(dev))
                val_loss_sum += float(vloss.mean())
                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        print(f"Epoch {ep}   valid_loss: {vloss_avg:.6f}")

        ckpt = save_dir / f"model_ep{ep}_ctx{T}_roll{K}_vloss{vloss_avg:.4f}_lr{lr:.0e}_b{batch}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--train-file-limit", type=int, default=None)
    p.add_argument("--val-file-limit", type=int, default=None)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=20, help="batches between progress logs")
    p.add_argument("--debug-timing", action="store_true", help="enable detailed timing logs for debugging")
    args = p.parse_args()
    train(
        args.config,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr=args.lr,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        steps_per_epoch=args.steps_per_epoch,
        num_workers=args.num_workers,
        train_file_limit=args.train_file_limit,
        val_file_limit=args.val_file_limit,
        log_interval=args.log_interval,
        debug_timing=args.debug_timing,
    )


if __name__ == "__main__":
    main()
