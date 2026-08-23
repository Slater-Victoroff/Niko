import argparse
import copy
import json
import sys
import time
from pathlib import Path
import yaml
import torch

# Ensure `src` is on sys.path so `python src/training/train.py` works from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model
from training.losses import vnmse_rollout_loss, well_style_vrmse, physics_consistency_loss
from training.physics_diagnostics import compute_physics_diagnostics, format_diagnostics
from training.log_utils import print_gns
from core.states import Params
from core.soap import SOAP
import dataloader as dl


def _curriculum_k(epoch: int, total_epochs: int, k_start: int, k_end: int) -> int:
    """Linear ramp from k_start (epoch 1) to k_end (final epoch), one step
    per epoch -- e.g. 7 epochs, k_start=2, k_end=16 gives 2,4,7,9,11,14,16.
    Motivation: a fixed rollout_steps the whole time asks the model to solve
    short-horizon transition accuracy AND long-horizon coherence
    simultaneously from scratch; training on short rollouts first (where the
    immediate next-state transition is the only thing being scored, not
    diluted 1/K across a whole trajectory) then growing the horizon lets the
    model establish accurate per-step dynamics before being asked to hold
    them stable over many steps. See EXPERIMENT_LOG.md.
    """
    if total_epochs <= 1:
        return k_end
    frac = (epoch - 1) / (total_epochs - 1)
    k = round(k_start + frac * (k_end - k_start))
    return max(1, min(k_end, k))


def train(
    cfg_path: str,
    data_dir: str,
    epochs: int,
    batch: int,
    device: str,
    lr: float,
    save_dir: str,
    num_workers: int,
    log_interval: int,
    train_file_limit: int | None,
    steps_per_epoch: int | None,
    val_file_limit: int | None,
    context_frames: int,
    rollout_steps: int,
    param_subset: str | None = None,
    soap_max_precond_dim: int = 10000,
    debug_timing: bool = False,
    log_physics: bool = False,
    log_grad_norms: bool = False,
    diag_interval: int = 500,
    diag_kappa: float = 0.1,
    diag_nu: float | None = None,
    diag_g: float = 1.0,
    rollout_curriculum_start: int | None = None,
    resume_from: str | None = None,
    start_epoch: int = 1,
    n_substeps: int = 1,
):
    torch.cuda.set_device(device)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    T = context_frames
    K = rollout_steps

    is_direct = cfg.get("model_type") == "direct"

    # encoder.in_channels must match field_spec's total channel count (4 for RB/shear_flow,
    # 11 for active_matter's concentration+velocity+D+E, etc.) -- no longer asserted to a
    # fixed "4" here; a real mismatch is still caught loudly, just by the dataloader's own
    # field_spec-driven channel assertion in H5RayleighBenardFields.__getitem__, which knows
    # the actual expected count for whatever field_spec is configured, rather than this
    # function hardcoding RB's specific case.
    if not is_direct:
        # Patch cfg before build_model so the encoder (and context_cond_encoder, if
        # configured in place of param_encoder) receive the correct context_frames.
        cfg["encoder"]["context_frames"] = T
        if "context_cond_encoder" in cfg:
            cfg["context_cond_encoder"]["context_frames"] = T

    cfg["rollout_steps"] = K

    model = build_model(copy.deepcopy(cfg))
    dev = torch.device(device)
    model.to(dev)

    if resume_from is not None:
        # Only model_state_dict is ever saved (see the checkpoint block below) -- no
        # optimizer state, so SOAP's preconditioner starts cold again here rather than
        # picking up its prior running estimate. Fine for "keep training this checkpoint
        # a few more epochs to see if it's still improving" (the actual use case this was
        # added for); if warm-restarting the optimizer state ever matters, that's a
        # separate change (save/load opt.state_dict() too).
        ckpt = torch.load(resume_from, map_location=dev, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"--resume-from {resume_from}: checkpoint mismatch, "
                                f"missing={missing}, unexpected={unexpected}")
        print(f"Resumed weights from {resume_from} (checkpoint epoch {ckpt.get('epoch') if isinstance(ckpt, dict) else '?'})")

    param_choices = None
    if param_subset:
        with open(param_subset, "r") as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]
        print(f"Restricting to param subset {param_subset}: {param_choices}")

    # field_spec (which HDF5 datasets to load and how to stack them into channels) lives in
    # the training config itself, same as encoder/operator/decoder -- either a named preset
    # (e.g. "active_matter", resolved below) or an inline list of {"key":..., "n_components":...}
    # dicts, for a dataset with no named preset yet. None (the default, RB's own configs never
    # set this key) keeps the original RB_FIELD_SPEC default inside create_param_dataloaders.
    field_spec = cfg.get("field_spec")
    if isinstance(field_spec, str):
        field_spec = {
            "rayleigh_benard": dl.RB_FIELD_SPEC,
            "active_matter": dl.ACTIVE_MATTER_FIELD_SPEC,
            "shear_flow": dl.SHEAR_FLOW_FIELD_SPEC,
        }[field_spec]

    train_loader, val_loader, _ = dl.create_param_dataloaders(
        data_dir,
        batch_size=batch,
        context_frames=T,
        predict_frames=K,
        num_workers=num_workers,
        train_file_limit=train_file_limit,
        val_file_limit=val_file_limit,
        param_choices=param_choices,
        field_spec=field_spec,
    )
    print(f"Using real dataloader from {data_dir}")

    opt = SOAP(model.parameters(), lr=lr, max_precond_dim=soap_max_precond_dim)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    total_batches = len(train_loader)

    initial_anchor = cfg.get("anchor_target", False)

    for ep in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss_sum, running_batches = 0.0, 0
        interval_batches = 0
        interval_start = time.perf_counter()
        tb = len(train_loader)

        # k_train: this epoch's training rollout length. Fixed at K (unchanged behavior)
        # unless rollout_curriculum_start is set, in which case it ramps K_start -> K
        # linearly over the epochs (see _curriculum_k) -- train_loader/val_loader are
        # always built at the full K (predict_frames=K), so early-curriculum epochs just
        # use a PREFIX of yb and a shorter model rollout, not a different dataloader.
        # Validation always evaluates at the full K regardless of k_train, so valid_loss
        # stays comparable epoch to epoch and across curriculum vs non-curriculum runs.
        if rollout_curriculum_start is not None:
            k_train = _curriculum_k(ep, epochs, rollout_curriculum_start, K)
        else:
            k_train = K
        print(f"Epoch {ep}   k_train: {k_train}" + (f" (curriculum, target {K})" if rollout_curriculum_start is not None else "")
              + (f"   n_substeps: {n_substeps}" if n_substeps != 1 else ""))

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

            yb_k = yb[:, :k_train]

            opt.zero_grad()
            pred = model(xb, steps=k_train, params=params_obj, debug_timing=debug_timing, return_initial_encode=initial_anchor, n_substeps=n_substeps)

            if initial_anchor:
                initial, pred = pred
                initial_target = xb[:, -1, ...]
                initial_loss = well_style_vrmse(initial.unsqueeze(1), initial_target.unsqueeze(1)).mean()
                rollout_loss = well_style_vrmse(pred, yb_k).mean()
                initial_weight = 1 / k_train
                loss = initial_weight * initial_loss + rollout_loss
            else:
                loss = well_style_vrmse(pred, yb_k).mean()
            if debug_timing:
                loss_start = torch.cuda.Event(enable_timing=True)
                loss_start.record()

            loss.backward()

            if log_grad_norms and batch_idx > 0 and batch_idx % log_interval == 0:
                global_step = (ep - 1) * total_batches + batch_idx
                print_gns(model, step=global_step)

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

            if log_physics and diag_interval > 0 and batch_idx > 0 and batch_idx % diag_interval == 0:
                global_step = (ep - 1) * total_batches + batch_idx
                diag = compute_physics_diagnostics(pred.detach(), yb_k.detach(), kappa=diag_kappa, nu=diag_nu, g=diag_g)
                print(format_diagnostics(diag, step=global_step, prefix="TRAIN_DIAG"))

            if batch_idx > 0 and batch_idx % log_interval == 0:
                sample_vrmse = well_style_vrmse(pred.detach(), yb_k.detach()).mean().item()
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
        val_diag_accum: dict[str, float] = {}
        val_diag_count = 0
        with torch.no_grad():
            for vbatch in val_loader:
                vxb, vyb, vparams = vbatch

                vxb = vxb.to(dev)
                vyb = vyb.to(dev)
                vp = Params(values=vparams.to(dev))

                vpred = model(vxb, steps=K, params=vp, return_initial_encode=False, n_substeps=n_substeps)
                # use variance-normalized MSE (VRMSE) as validation metric
                vloss = well_style_vrmse(vpred, vyb.to(dev))
                val_loss_sum += float(vloss.mean())
                val_batches += 1

                if log_physics and diag_interval > 0:
                    vdiag = compute_physics_diagnostics(vpred.detach(), vyb.detach(), kappa=diag_kappa, nu=diag_nu, g=diag_g)
                    for k, v in vdiag.items():
                        val_diag_accum[k] = val_diag_accum.get(k, 0.0) + v
                    val_diag_count += 1

        vloss_avg = val_loss_sum / max(1, val_batches)
        print(f"Epoch {ep}   valid_loss: {vloss_avg:.6f}")

        if log_physics and diag_interval > 0 and val_diag_count > 0:
            val_diag_avg = {k: v / val_diag_count for k, v in val_diag_accum.items()}
            print(format_diagnostics(val_diag_avg, step=ep, prefix="VAL_DIAG  "))

        substep_tag = f"_sub{n_substeps}" if n_substeps != 1 else ""
        ckpt = save_dir / f"model_ep{ep}_ctx{T}_roll{K}{substep_tag}_vloss{vloss_avg:.4f}_lr{lr:.0e}_b{batch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "config_path": str(cfg_path),
                "epoch": ep,
                "val_loss": vloss_avg,
                "lr": lr,
                "batch": batch,
                "context_frames": T,
                "rollout_steps": K,
            },
            ckpt,
        )
        print(f"Saved checkpoint: {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--context-frames", type=int, required=True, help="Number of context frames fed to the encoder.")
    p.add_argument("--rollout-steps", type=int, required=True, help="Number of autoregressive rollout steps.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="/app/data/datasets/rayleigh_benard/data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--soap-max-precond-dim", type=int, default=10000,
                    help="SOAP skips preconditioning any parameter axis wider than this (falls back to "
                         "plain per-axis scaling on that axis instead). Default (10000) matches SOAP's "
                         "own default and is fine for every recipe so far; lower it (e.g. 1024) for "
                         "parameters with a very wide axis (e.g. LocallyConnected1x1's per-position "
                         "weight tensor) -- SOAP's eigh-based preconditioner update crashes cusolver "
                         "on large-enough covariance matrices otherwise, rather than just being slow.")
    p.add_argument("--train-file-limit", type=int, default=None)
    p.add_argument("--val-file-limit", type=int, default=None)
    p.add_argument("--param-subset", default=None,
                    help="path to a JSON file like {\"combos\": [[1e7, 1.0], [1e8, 1.0]]} restricting "
                         "training to just those (rayleigh, prandtl) combos, instead of all 35 -- "
                         "for quick iteration without paying for the full ~300GB dataset")
    p.add_argument("--save-dir", default="/app/checkpoints")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=20, help="batches between progress logs")
    p.add_argument("--debug-timing", action="store_true", help="enable detailed timing logs for debugging")
    p.add_argument("--log-physics", action="store_true",
                   help="enable physics diagnostics logging (pred vs target physical quantities, gated by --diag-interval)")
    p.add_argument("--log-grad-norms", action="store_true",
                   help="print component-wise gradient norms (param_encoder/encoder/operator/decoder) every --log-interval steps")
    p.add_argument("--diag-interval", type=int, default=500,
                   help="steps between physics diagnostics logs (0 = disabled)")
    p.add_argument("--diag-kappa", type=float, default=0.1,
                   help="buoyancy diffusivity kappa=1/Pr for PDE residual (default Pr=10)")
    p.add_argument("--diag-nu", type=float, default=None,
                   help="kinematic viscosity nu for momentum residual (default: same as kappa)")
    p.add_argument("--diag-g", type=float, default=1.0,
                   help="gravity coefficient g in vertical momentum forcing term g*b")
    p.add_argument("--rollout-curriculum-start", type=int, default=None,
                   help="if set, ramps the TRAINING rollout length linearly from this value "
                        "(epoch 1) up to --rollout-steps (final epoch), one step per epoch -- "
                        "validation always uses the full --rollout-steps regardless. "
                        "Default: unset, fixed rollout_steps every epoch (unchanged behavior).")
    p.add_argument("--resume-from", default=None,
                   help="path to a checkpoint .pt to load model weights from before training "
                        "starts -- e.g. to keep training a run that hadn't converged yet. Only "
                        "the weights carry over (checkpoints don't save optimizer state), so "
                        "SOAP's preconditioner starts cold. Pair with --start-epoch so logging/ "
                        "checkpoint filenames reflect the true cumulative epoch instead of "
                        "restarting at 1.")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="epoch number to start counting from (only affects logging/checkpoint "
                        "filenames and, if --rollout-curriculum-start is also set, its ramp -- "
                        "does not affect training when curriculum is unset). Use with --resume-from.")
    p.add_argument("--n-substeps", type=int, default=1,
                   help="calls the operator this many times per output frame at dt=1/n_substeps "
                        "instead of once at dt=1 -- DISCO-inspired sub-step integration (see "
                        "EXPERIMENT_LOG.md §20). 1 (default) is the original, unchanged behavior. "
                        "Applies to both training and validation rollouts (same value for both).")
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
        context_frames=args.context_frames,
        rollout_steps=args.rollout_steps,
        param_subset=args.param_subset,
        soap_max_precond_dim=args.soap_max_precond_dim,
        log_interval=args.log_interval,
        debug_timing=args.debug_timing,
        log_physics=args.log_physics,
        log_grad_norms=args.log_grad_norms,
        diag_interval=args.diag_interval,
        diag_kappa=args.diag_kappa,
        diag_nu=args.diag_nu,
        diag_g=args.diag_g,
        rollout_curriculum_start=args.rollout_curriculum_start,
        resume_from=args.resume_from,
        start_epoch=args.start_epoch,
        n_substeps=args.n_substeps,
    )


if __name__ == "__main__":
    main()
