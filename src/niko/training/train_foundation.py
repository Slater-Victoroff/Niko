"""Foundation-model test: one shared trunk (field_embedder + context_cond_encoder
+ encoder + operator) trained jointly across rayleigh_benard/shear_flow/
active_matter, with a separate decoder per task. Tests whether joint training
helps the shared trunk vs. each task's own from-scratch trunk (this session's
single-task baselines: plain_helmholtz_broad8 0.229, shear_flow_broad8 0.385,
active_matter 0.446-0.567 -- see EXPERIMENT_LOG.md).

The three tasks are at different native resolutions (rayleigh_benard
128x512, shear_flow 512x256, active_matter 256x256), so a single batch
tensor can't literally mix samples from different tasks. Instead, every
optimizer step pulls one batch from ALL THREE tasks (each task's own
shuffled loader), runs forward+backward for each without zeroing grad in
between -- so gradients from all three accumulate together into the shared
trunk every single step (each task's own decoder only ever gets gradient
from its own batch, since decoders are disjoint per task) -- then a single
opt.step() applies the combined update.

Earlier version of this script let a task drop out once ITS OWN loader was
exhausted for the epoch, so the last ~third of every epoch (after
active_matter's ~254 batches and rayleigh_benard's ~7560 ran out) was
shear_flow-only for ~4500 straight steps -- the shared trunk drifted toward
shear_flow's optimum with no rayleigh_benard/active_matter gradient at all
for that whole stretch, and validation (run right after) caught the drifted
trunk: rayleigh_benard/active_matter val loss was 30-60x worse than their
own single-task baselines despite normal training loss all epoch, while
shear_flow (never without gradient) came out fine. Fixed by having the two
shorter loaders restart (fresh shuffle) whenever exhausted mid-epoch, so all
three ALWAYS co-occur in every step -- shear_flow (the longest, ~12096
batches) is still seen exactly once per epoch with no repeats and defines
the epoch length; rayleigh_benard cycles ~1.6x and active_matter ~47x within
that same epoch to keep pace. This restarts via a fresh iter(loader) each
pass, not itertools.cycle (which caches every batch it's ever yielded --
the actual cause of this script's two earlier OOM crashes), so memory stays
bounded regardless of how many times a shorter loader repeats.

Conditioning is context-inferred (shared context_cond_encoder), not
ground-truth params: rayleigh_benard/shear_flow use 2 log10 params,
active_matter uses 3 (one signed) -- context-inferred cond sidesteps that
mismatch entirely, and this session's own probes found it matches
ground-truth-param conditioning quality on rayleigh_benard.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import FieldEmbedder, canonical_field_name
from encoders.context_cond import ContextCondEncoder
from encoders.sequence_conv import SequenceConvEncoder
from operators.transport_operator import TransportOperator
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder
from models.foundation_model import FoundationModel
from training.losses import well_style_vrmse
from core.soap import SOAP
import dataloader as dl


TASKS = {
    "rayleigh_benard": dict(
        field_spec=dl.RB_FIELD_SPEC,
        data_dir="/app/data/datasets/rayleigh_benard/data",
        param_subset="/app/configs/param_subsets/broad8.json",
        batch=8,
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True),
    ),
    "shear_flow": dict(
        field_spec=dl.SHEAR_FLOW_FIELD_SPEC,
        data_dir="/app/data/datasets/shear_flow/data",
        param_subset="/app/configs/param_subsets/shear_flow_broad8.json",
        # shear_flow's 512x256 grid has ~2x rayleigh_benard's pixel count --
        # EXPERIMENT_LOG hit a real OOM at batch=8 for this task specifically.
        batch=4,
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True,
                             scalar_field_names=["pressure", "tracer"]),
    ),
    "active_matter": dict(
        field_spec=dl.ACTIVE_MATTER_FIELD_SPEC,
        data_dir="/app/data/datasets/active_matter/data",
        param_subset=None,
        train_file_limit=8,  # no existing named param_subset for this task; match broad8's size
        val_file_limit=8,  # unset originally -> pulled all 16 val files uncapped, a real driver of the OOM below
        batch=8,
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["concentration"], tensor_field_names=["D", "E"]),
    ),
}

# create_param_dataloaders defaults traj_cache_capacity to max(2*batch_size, 64)
# PER DATASET -- fine for a single task (2 datasets: train+val), but this script
# holds 6 simultaneously (train+val x 3 tasks), each caching FULL 200-frame
# trajectories (not just context windows). At the default that's 100+GB combined
# (shear_flow's grid alone is ~2x the others' pixel count) -- caused a real host
# OOM kill (docker inspect confirmed OOMKilled=true, no container memory limit,
# so this was genuine system RAM exhaustion) on the first attempt. A small
# explicit cap keeps total resident trajectory data in the single-digit GB
# range across all 6 loaders combined.
TRAJ_CACHE_CAPACITY = 8


def repeat_forever(loader):
    """Re-iterate `loader` indefinitely, starting a fresh (re-shuffled) pass
    each time the previous one is exhausted. NOT itertools.cycle(loader):
    cycle() caches every item it has ever yielded internally so it can
    replay them once exhausted -- catastrophic for large batch tensors
    repeated across a long run (this was the direct cause of this script's
    two earlier OOM crashes). This only ever holds the current pass's
    DataLoader iterator; a finished pass's batches are freed normally as
    soon as the training loop moves past them.
    """
    while True:
        for batch in loader:
            yield batch


def soft_cap_loss(loss: torch.Tensor, threshold: float) -> torch.Tensor:
    """Huber-style soft cap: identity below `threshold` (every normal batch is
    completely unaffected, full gradient, nothing discarded), sqrt-growth
    above it (an anomalous batch still pulls the model away from whatever
    produced it -- the signal isn't thrown away -- it just can't dominate the
    shared trunk's gradient the way a raw 10^11-scale loss would). No hard
    skip, no fixed clamp: this only reduces marginal influence past the
    threshold, continuously (value-continuous at the boundary; derivative
    has a kink there, same as ordinary Huber loss).

    Root cause (see EXPERIMENT_LOG.md): a rare, non-physical anomaly in the
    raw simulation data (a real solver artifact, confirmed by direct
    inspection -- not something this pipeline introduces) drove one batch's
    loss to ~10^11. Gradient-norm clipping alone caps the resulting step
    *size* but not its *direction* -- this caps the loss itself, before that
    direction is ever computed.
    """
    if loss <= threshold:
        return loss
    excess = loss - threshold
    return threshold + torch.sqrt(excess + 1.0) - 1.0


def union_channel_specs(tasks: dict) -> dict:
    specs = {}
    for cfg in tasks.values():
        for spec in cfg["field_spec"]:
            name = canonical_field_name(spec["key"])
            n = spec["n_components"]
            if name in specs and specs[name] != n:
                raise ValueError(f"Field '{name}' has conflicting n_components across tasks: {specs[name]} vs {n}")
            specs[name] = n
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-frames", type=int, required=True)
    p.add_argument("--rollout-steps", type=int, required=True)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--canonical-dim", type=int, default=12)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--cond-dim", type=int, default=32)
    p.add_argument("--decoder-hidden-dim", type=int, default=48)
    p.add_argument("--complex-term", action="store_true",
                    help="add the FFT-fed complex-rotation branch (complex_proj + ComplexAmplitude/RotationTerm) "
                         "on top of the real Helmholtz terms -- see EXPERIMENT_LOG.md for the single-task history")
    p.add_argument("--amplitude-scale", type=float, default=1.0,
                    help="tanh ceiling on the complex branch's per-step log-amplitude factor, exp(+-this)")
    p.add_argument("--loss-cap-threshold", type=float, default=25.0,
                    help="soft_cap_loss threshold -- comfortably above any observed normal-batch loss across "
                         "all three tasks (including active_matter's noisier variance-normalized spikes), well "
                         "below where a genuinely anomalous batch's raw loss lands (thousands+)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-dir", default="/app/checkpoints/foundation_model")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    T, K = args.context_frames, args.rollout_steps

    channel_specs = union_channel_specs(TASKS)
    print(f"Canonical field registry: {channel_specs}")

    field_embedder = FieldEmbedder(channel_specs, canonical_dim=args.canonical_dim).to(dev)
    context_cond_encoder = ContextCondEncoder(
        in_channels=args.canonical_dim, context_frames=T, cond_dim=args.cond_dim, hidden_dim=args.hidden_dim,
    ).to(dev)
    encoder = SequenceConvEncoder(
        in_channels=args.canonical_dim, context_frames=T, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
    ).to(dev)
    operator = TransportOperator(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
        terms=("advection", "diffusion", "skew", "forcing"), film=True,
        complex_term=args.complex_term, amplitude_scale=args.amplitude_scale,
    ).to(dev)

    decoders = {}
    train_loaders, val_loaders = {}, {}
    for task, cfg in TASKS.items():
        decoders[task] = SharedTrunkFieldHeadsDecoder(
            latent_dim=args.latent_dim, hidden_dim=args.decoder_hidden_dim, upsample=2,
            **cfg["decoder_kwargs"],
        ).to(dev)

        param_choices = None
        if cfg.get("param_subset"):
            with open(cfg["param_subset"]) as f:
                param_choices = [tuple(c) for c in json.load(f)["combos"]]

        train_loader, val_loader, _ = dl.create_param_dataloaders(
            cfg["data_dir"], batch_size=cfg["batch"], context_frames=T, predict_frames=K,
            num_workers=args.num_workers, param_choices=param_choices,
            train_file_limit=cfg.get("train_file_limit"), val_file_limit=cfg.get("val_file_limit"),
            field_spec=cfg["field_spec"], traj_cache_capacity=TRAJ_CACHE_CAPACITY,
        )
        train_loaders[task] = train_loader
        val_loaders[task] = val_loader
        print(f"[{task}] {len(train_loader)} train batches/epoch, {len(val_loader)} val batches, batch={cfg['batch']}")

    model = FoundationModel(field_embedder, context_cond_encoder, encoder, operator, decoders).to(dev)
    opt = SOAP(model.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(TASKS.keys())
    # Every step draws from all three; the longest task's own loader length
    # defines the epoch (it's seen exactly once, no repeats), the shorter
    # two restart mid-epoch as needed to keep supplying a batch every step.
    steps_per_epoch = max(len(train_loaders[t]) for t in task_names)
    task_iters = {t: repeat_forever(train_loaders[t]) for t in task_names}

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        task_loss_sum = {t: 0.0 for t in task_names}
        interval_start = time.perf_counter()

        for step in range(1, steps_per_epoch + 1):
            opt.zero_grad()
            step_loss = 0.0
            for task in task_names:
                xb, yb, bparams = next(task_iters[task])
                xb, yb = xb.to(dev), yb.to(dev)
                initial, pred = model(xb, field_spec=TASKS[task]["field_spec"], task=task, steps=K,
                                       return_initial_encode=True)
                initial_target = xb[:, -1, ...]
                initial_loss = well_style_vrmse(initial.unsqueeze(1), initial_target.unsqueeze(1)).mean()
                rollout_loss = well_style_vrmse(pred, yb).mean()
                raw_task_loss = (1.0 / K) * initial_loss + rollout_loss

                task_loss = soft_cap_loss(raw_task_loss, args.loss_cap_threshold)
                if task_loss.item() > args.loss_cap_threshold:
                    print(f"  [soft-cap engaged] epoch={ep} step={step} task={task} "
                          f"raw_loss={raw_task_loss.item():.6g} capped_loss={task_loss.item():.6g} "
                          f"bparams[0]={bparams[0].tolist()}")
                task_loss.backward()  # accumulates into .grad -- no zero_grad() between tasks this step

                loss_item = float(task_loss.item())
                step_loss += loss_item
                task_loss_sum[task] += loss_item

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += step_loss

            if step % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - interval_start) * 1000.0
                print(f"Epoch {ep}  step {step}/{steps_per_epoch}  "
                      f"combined_loss {step_loss:.6f}  time/step {elapsed_ms / args.log_interval:.2f}ms")
                interval_start = time.perf_counter()

        print(f"Epoch {ep}   train_loss (avg combined-step loss): {loss_sum / steps_per_epoch:.6f}  ({steps_per_epoch} steps)")
        for t in task_names:
            print(f"Epoch {ep}   train_loss[{t}]: {task_loss_sum[t] / steps_per_epoch:.6f}  ({steps_per_epoch} batches)")

        # Per-task validation, kept separate -- not comparable across tasks
        # (different channel composition / normalization distribution each
        # time, per EXPERIMENT_LOG.md).
        model.eval()
        val_results = {}
        with torch.no_grad():
            for t in task_names:
                vloss_sum, vbatches = 0.0, 0
                for xb, yb, _bparams in val_loaders[t]:
                    xb, yb = xb.to(dev), yb.to(dev)
                    pred = model(xb, field_spec=TASKS[t]["field_spec"], task=t, steps=K,
                                 return_initial_encode=False)
                    vloss_sum += float(well_style_vrmse(pred, yb).mean().item())
                    vbatches += 1
                vloss_avg = vloss_sum / max(1, vbatches)
                val_results[t] = vloss_avg
                print(f"Epoch {ep}   valid_loss[{t}]: {vloss_avg:.6f}")

        tag = "_".join(f"{t}{v:.4f}" for t, v in val_results.items())
        ckpt_out = save_dir / f"foundation_ep{ep}_ctx{T}_roll{K}_{tag}.pt"
        torch.save({
            "field_embedder_state_dict": field_embedder.state_dict(),
            "context_cond_encoder_state_dict": context_cond_encoder.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "operator_state_dict": operator.state_dict(),
            "decoder_state_dicts": {t: decoders[t].state_dict() for t in task_names},
            "canonical_dim": args.canonical_dim,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "cond_dim": args.cond_dim,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "context_frames": T,
            "rollout_steps": K,
            "epoch": ep,
            "val_loss_by_task": val_results,
        }, ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")


if __name__ == "__main__":
    main()
