"""Reference baseline: param_encoder + encoder + decoder only, no operator,
no rollout. Trains the encode/decode pair to reconstruct the last context
frame directly -- the operator's parameters are never constructed, so
there's no wasted memory or optimizer state for them.

This measures the information bottleneck's ceiling: how good can single-
frame reconstruction ever be through a given encoder/decoder pair, decoupled
from the harder job of predicting forward in time. Useful as a reference
number to interpret the full model's rollout loss against -- if the full
model's rollout error is close to this floor, the operator is doing about
as well as it can given what the encoder/decoder can represent; if there's
a big gap, the operator itself is the bottleneck, not the latent space.

Same config format as train.py (encoder/decoder/param_encoder sections
reused as-is; the operator section is present in the YAML but ignored here).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_param_encoder, build_encoder, build_decoder
from training.losses import well_style_vrmse
from core.states import Params
from core.soap import SOAP
import dataloader as dl

OUT_DIR = Path("/app/checkpoints")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--context-frames", type=int, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="soap", choices=["adam", "soap"])
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--param-subset", default=None,
                    help="path to a JSON file like {\"combos\": [[1e7, 1.0], [1e8, 1.0]]}")
    p.add_argument("--save-dir", default="/app/checkpoints/autoencoder_baseline")
    p.add_argument("--log-interval", type=int, default=200)
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    T = args.context_frames
    enc_cfg = copy.deepcopy(cfg["encoder"])
    enc_cfg["context_frames"] = T
    param_encoder = build_param_encoder(copy.deepcopy(cfg["param_encoder"])).to(dev)
    encoder = build_encoder(enc_cfg).to(dev)
    decoder = build_decoder(copy.deepcopy(cfg["decoder"])).to(dev)

    for name, m in [("param_encoder", param_encoder), ("encoder", encoder), ("decoder", decoder)]:
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name} params: total={n_params}")

    param_choices = None
    if args.param_subset:
        with open(args.param_subset) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]
        print(f"Restricting to param subset {args.param_subset}: {param_choices}")

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir,
        batch_size=args.batch,
        context_frames=T,
        predict_frames=1,
        num_workers=args.num_workers,
        param_choices=param_choices,
    )
    print(f"Using real dataloader from {args.data_dir}")

    params_to_train = list(param_encoder.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    if args.optimizer == "soap":
        opt = SOAP(params_to_train, lr=args.lr)
    else:
        opt = torch.optim.Adam(params_to_train, lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def encode_decode(xb, bparams):
        cond_input = Params(values=bparams.to(dev))
        cond = param_encoder(cond_input)
        if hasattr(encoder, "cond_dim") and encoder.cond_dim is not None:
            z0 = encoder(xb, cond=cond)
        else:
            z0 = encoder(xb)
        recon = decoder(z0.grid, cond=cond)
        return recon

    for ep in range(1, args.epochs + 1):
        param_encoder.train()
        encoder.train()
        decoder.train()
        running_loss_sum, running_batches = 0.0, 0
        interval_batches, interval_start = 0, time.perf_counter()
        tb = len(train_loader)

        for batch_idx, (xb, _yb, bparams) in enumerate(train_loader):
            xb = xb.to(dev)
            opt.zero_grad()
            recon = encode_decode(xb, bparams)
            target = xb[:, -1, ...]
            loss = well_style_vrmse(recon.unsqueeze(1), target.unsqueeze(1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
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

        param_encoder.eval()
        encoder.eval()
        decoder.eval()
        val_loss_sum, val_batches = 0.0, 0
        mse_sum = torch.zeros(4, device=dev)
        with torch.no_grad():
            for xb, _yb, bparams in val_loader:
                xb = xb.to(dev)
                recon = encode_decode(xb, bparams)
                target = xb[:, -1, ...]
                vloss = well_style_vrmse(recon.unsqueeze(1), target.unsqueeze(1))
                val_loss_sum += float(vloss.mean())
                mse_sum += ((recon - target) ** 2).mean(dim=(0, 2, 3))
                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        mse_avg = (mse_sum / max(1, val_batches)).tolist()
        print(f"Epoch {ep}   valid_loss (vrmse): {vloss_avg:.6f}")
        print(f"Epoch {ep}   valid_mse_by_channel [pressure, buoyancy, velocity_x, velocity_y]: "
              f"[{mse_avg[0]:.6f}, {mse_avg[1]:.6f}, {mse_avg[2]:.6f}, {mse_avg[3]:.6f}]")

        ckpt = save_dir / f"ae_ep{ep}_ctx{T}_vloss{vloss_avg:.4f}.pt"
        torch.save(
            {
                "param_encoder_state_dict": param_encoder.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "config": cfg,
                "config_path": str(args.config),
                "epoch": ep,
                "val_loss": vloss_avg,
                "val_mse_by_channel": mse_avg,
                "context_frames": T,
            },
            ckpt,
        )
        print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
