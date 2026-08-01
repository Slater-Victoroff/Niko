"""JEPA-style operator training: reuse a pretrained (frozen) param_encoder +
encoder from an autoencoder checkpoint (scripts/train_autoencoder_baseline.py),
and train ONLY a new operator to predict forward in latent space -- loss is
computed entirely in latent space, never decoded to physical units during
training. No stop-gradient/EMA target network needed: the encoder is frozen
and was pretrained via pixel reconstruction (autoencoding), which already
rules out representation collapse structurally (a collapsed/constant latent
can't reconstruct anything) -- unlike from-scratch two-tower JEPA, which
needs those tricks specifically because the target encoder is free to
co-adapt with the predictor.

Target latents for each future step t+k: since the frozen encoder expects a
context_frames-length window (context_frames=6 here, baked into its input
conv's channel count), a "target latent for frame t+k" means re-encoding a
full 6-frame window ENDING at t+k, not the single frame in isolation (which
would be out-of-distribution for a network that never saw repeated/degenerate
frame stacks during training). A standard batch already provides exactly
enough contiguous frames for this: xb (context, t-5..t) concatenated with yb
(rollout targets, t+1..t+6) gives 12 consecutive frames (t-5..t+6), and the
window ending at t+k is simply combined[:, k:k+6] for k=1..rollout_steps --
no dataloader change needed.

Validation also reports a physical-space VRMSE (decoding the final rollout
latent through the frozen decoder) purely for comparability against every
other recipe in this project -- this number plays no role in the training
loss itself.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_param_encoder, build_encoder, build_operator, build_decoder
from training.losses import well_style_vrmse
from core.states import Params
from core.soap import SOAP
import dataloader as dl


def load_frozen_autoencoder(ae_checkpoint_path, dev, context_frames):
    ckpt = torch.load(ae_checkpoint_path, map_location=dev)
    cfg = ckpt["config"]

    if ckpt.get("context_frames", context_frames) != context_frames:
        raise ValueError(
            f"--context-frames={context_frames} doesn't match the autoencoder checkpoint's own "
            f"context_frames={ckpt.get('context_frames')} -- its input conv is sized for that "
            f"exact window length, a mismatch will fail to load or silently misinterpret the input."
        )

    param_encoder = build_param_encoder(copy.deepcopy(cfg["param_encoder"])).to(dev)
    param_encoder.load_state_dict(ckpt["param_encoder_state_dict"])

    enc_cfg = copy.deepcopy(cfg["encoder"])
    enc_cfg["context_frames"] = context_frames
    encoder = build_encoder(enc_cfg).to(dev)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    decoder = build_decoder(copy.deepcopy(cfg["decoder"])).to(dev)
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    for m in (param_encoder, encoder, decoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    return param_encoder, encoder, decoder, cfg, ckpt["val_loss"]


def encode(encoder, x, cond):
    """Only pass cond through if the encoder actually declared a cond_dim --
    mirrors LatentDynamicsModel.forward's own guard. Several encoders in this
    project (e.g. SplitEncoder) accept a cond_dim constructor arg but never
    assign self.cond_dim, so they silently don't support conditioning; calling
    them with cond=... unconditionally raises inside validate_input."""
    if hasattr(encoder, "cond_dim") and encoder.cond_dim is not None:
        return encoder(x, cond=cond)
    return encoder(x)


def latent_mse(pred_latent, target_latent):
    """MSE on LatentState.grid (real_grid + irfft2(spectral_grid)) -- the
    same combined real-space representation the decoder itself would
    consume, so real and complex parts are compared through one principled
    view instead of an arbitrarily-weighted sum of two separate losses."""
    return (pred_latent.grid - target_latent.grid).pow(2).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--operator-config", required=True,
                    help="YAML with the operator section to train (e.g. configs/streamheads.yaml); "
                         "its encoder/decoder sections are ignored -- those come from --autoencoder-checkpoint")
    p.add_argument("--autoencoder-checkpoint", required=True,
                    help="checkpoint from train_autoencoder_baseline.py -- supplies the frozen "
                         "param_encoder/encoder/decoder and their exact architecture")
    p.add_argument("--context-frames", type=int, required=True)
    p.add_argument("--rollout-steps", type=int, required=True)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="soap", choices=["adam", "soap"])
    p.add_argument("--soap-max-precond-dim", type=int, default=10000)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--param-subset", default=None)
    p.add_argument("--save-dir", default="/app/checkpoints/jepa_operator")
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                    help="truncate each training epoch after this many batches -- for quick smoke tests")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    T, K = args.context_frames, args.rollout_steps

    param_encoder, encoder, decoder, ae_cfg, ae_val_loss = load_frozen_autoencoder(
        args.autoencoder_checkpoint, dev, T
    )
    print(f"Loaded frozen autoencoder from {args.autoencoder_checkpoint} "
          f"(encoder={ae_cfg['encoder']['name']}, its own reconstruction val_loss={ae_val_loss:.4f})")

    with open(args.operator_config) as f:
        op_full_cfg = yaml.safe_load(f)
    operator = build_operator(copy.deepcopy(op_full_cfg["operator"])).to(dev)

    for name, m in [("param_encoder (frozen)", param_encoder), ("encoder (frozen)", encoder),
                     ("decoder (frozen, eval-only)", decoder), ("operator (trainable)", operator)]:
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
        predict_frames=K,
        num_workers=args.num_workers,
        param_choices=param_choices,
    )
    print(f"Using real dataloader from {args.data_dir}")

    if args.optimizer == "soap":
        opt = SOAP(operator.parameters(), lr=args.lr, max_precond_dim=args.soap_max_precond_dim)
    else:
        opt = torch.optim.Adam(operator.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def encode_target_windows(combined, cond):
        """combined: [B, T+K, C, H, W] (context + rollout, consecutive). Returns
        a list of K frozen target LatentStates, one per rollout step, each
        encoded from the T-frame window ending at that step."""
        targets = []
        with torch.no_grad():
            for k in range(1, K + 1):
                window = combined[:, k:k + T]
                targets.append(encode(encoder, window, cond))
        return targets

    for ep in range(1, args.epochs + 1):
        operator.train()
        running_loss_sum, running_batches = 0.0, 0
        interval_batches, interval_start = 0, time.perf_counter()
        tb = len(train_loader)

        for batch_idx, (xb, yb, bparams) in enumerate(train_loader):
            xb, yb = xb.to(dev), yb.to(dev)
            cond_input = Params(values=bparams.to(dev))
            with torch.no_grad():
                cond = param_encoder(cond_input)
                z = encode(encoder, xb, cond)

            combined = torch.cat([xb, yb], dim=1)
            target_latents = encode_target_windows(combined, cond)

            opt.zero_grad()
            loss = 0.0
            for k in range(K):
                z = operator(z, cond=cond)
                loss = loss + latent_mse(z, target_latents[k])
            loss = loss / K
            loss.backward()
            torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)
            opt.step()

            running_loss_sum += float(loss.item())
            running_batches += 1
            interval_batches += 1

            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - interval_start) * 1000.0
                time_per_batch_ms = elapsed_ms / max(1, interval_batches)
                print(f"Epoch {ep}  batch {batch_idx}/{tb}  jepa_loss {loss.item():.6f}  "
                      f"time/batch {time_per_batch_ms:.2f}ms")
                interval_batches, interval_start = 0, time.perf_counter()

            if args.steps_per_epoch is not None and batch_idx >= args.steps_per_epoch:
                break

        avg_loss = running_loss_sum / max(1, running_batches)
        print(f"Epoch {ep}   train_jepa_loss: {avg_loss:.6f}")

        operator.eval()
        val_loss_sum, val_batches = 0.0, 0
        val_vrmse_sum = 0.0
        with torch.no_grad():
            for xb, yb, bparams in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                cond_input = Params(values=bparams.to(dev))
                cond = param_encoder(cond_input)
                z = encode(encoder, xb, cond)

                combined = torch.cat([xb, yb], dim=1)
                target_latents = encode_target_windows(combined, cond)

                vloss = 0.0
                decoded_steps = []
                for k in range(K):
                    z = operator(z, cond=cond)
                    vloss = vloss + latent_mse(z, target_latents[k])
                    decoded_steps.append(decoder(z.grid, cond=cond))
                vloss = vloss / K
                val_loss_sum += float(vloss.item())

                pred_stack = torch.stack(decoded_steps, dim=1)  # [B, K, C, H, W]
                vrmse = well_style_vrmse(pred_stack, yb).mean()
                val_vrmse_sum += float(vrmse.item())
                val_batches += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        vrmse_avg = val_vrmse_sum / max(1, val_batches)
        print(f"Epoch {ep}   valid_jepa_loss: {vloss_avg:.6f}   "
              f"valid_vrmse (decoded, comparison-only, not the training loss): {vrmse_avg:.6f}")

        ckpt = save_dir / f"jepa_op_ep{ep}_ctx{T}_roll{K}_vloss{vloss_avg:.4f}_vrmse{vrmse_avg:.4f}.pt"
        torch.save(
            {
                "operator_state_dict": operator.state_dict(),
                "operator_config": op_full_cfg["operator"],
                "autoencoder_checkpoint": args.autoencoder_checkpoint,
                "epoch": ep,
                "val_jepa_loss": vloss_avg,
                "val_vrmse": vrmse_avg,
                "context_frames": T,
                "rollout_steps": K,
            },
            ckpt,
        )
        print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
