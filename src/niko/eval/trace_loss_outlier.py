"""Deep-dive on a specific outlier batch found by find_loss_outlier.py:
replays the 6-step rollout with the frozen epoch-4 checkpoint, logging each
transport term's individual contribution magnitude at every step (not just
the summed operator output), plus the raw learned advection velocity field's
own magnitude before it's combined with the input's spatial gradient --
FiLM-conditioned terms (_FiLMConvNet) have no output-bounding tanh, unlike
the concat-conditioned path (_ConcatConvNet), so this checks whether that
gap is where the blowup actually originates.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import FieldEmbedder, canonical_field_name
from encoders.context_cond import ContextCondEncoder
from encoders.sequence_conv import SequenceConvEncoder
from operators.transport_operator import TransportOperator
from operators.transport_terms import dx_central, dy_central
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder
from models.foundation_model import FoundationModel
import dataloader as dl


TASKS = {
    "rayleigh_benard": dict(
        field_spec=dl.RB_FIELD_SPEC,
        data_dir="/app/data/datasets/rayleigh_benard/data",
        param_subset="/app/configs/param_subsets/broad8.json",
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True),
    ),
    "shear_flow": dict(
        field_spec=dl.SHEAR_FLOW_FIELD_SPEC,
        data_dir="/app/data/datasets/shear_flow/data",
        param_subset="/app/configs/param_subsets/shear_flow_broad8.json",
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True,
                             scalar_field_names=["pressure", "tracer"]),
    ),
    "active_matter": dict(
        field_spec=dl.ACTIVE_MATTER_FIELD_SPEC,
        data_dir="/app/data/datasets/active_matter/data",
        param_subset=None,
        train_file_limit=8,
        val_file_limit=8,
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["concentration"], tensor_field_names=["D", "E"]),
    ),
}


def union_channel_specs(tasks: dict) -> dict:
    specs = {}
    for cfg in tasks.values():
        for spec in cfg["field_spec"]:
            specs[canonical_field_name(spec["key"])] = spec["n_components"]
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="rayleigh_benard")
    p.add_argument("--batch-indices", type=int, nargs="+", required=True,
                    help="which batch-1 dataset indices to trace, e.g. 31358 31359 31360")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    T, K = ckpt["context_frames"], ckpt["rollout_steps"]
    canonical_dim, latent_dim = ckpt["canonical_dim"], ckpt["latent_dim"]
    hidden_dim, cond_dim = ckpt["hidden_dim"], ckpt["cond_dim"]
    decoder_hidden_dim = ckpt["decoder_hidden_dim"]

    channel_specs = union_channel_specs(TASKS)
    field_embedder = FieldEmbedder(channel_specs, canonical_dim=canonical_dim).to(dev)
    field_embedder.load_state_dict(ckpt["field_embedder_state_dict"])
    context_cond_encoder = ContextCondEncoder(
        in_channels=canonical_dim, context_frames=T, cond_dim=cond_dim, hidden_dim=hidden_dim,
    ).to(dev)
    context_cond_encoder.load_state_dict(ckpt["context_cond_encoder_state_dict"])
    encoder = SequenceConvEncoder(
        in_channels=canonical_dim, context_frames=T, latent_dim=latent_dim, hidden_dim=hidden_dim,
    ).to(dev)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    operator = TransportOperator(
        latent_dim=latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim,
        terms=("advection", "diffusion", "skew", "forcing"), film=True,
    ).to(dev)
    operator.load_state_dict(ckpt["operator_state_dict"])

    cfg = TASKS[args.task]
    decoder = SharedTrunkFieldHeadsDecoder(
        latent_dim=latent_dim, hidden_dim=decoder_hidden_dim, upsample=2, **cfg["decoder_kwargs"],
    ).to(dev)
    decoder.load_state_dict(ckpt["decoder_state_dicts"][args.task])
    model = FoundationModel(field_embedder, context_cond_encoder, encoder, operator, {args.task: decoder}).to(dev)
    model.eval()

    param_choices = None
    if cfg.get("param_subset"):
        with open(cfg["param_subset"]) as f:
            param_choices = [tuple(c) for c in json.load(f)["combos"]]

    train_loader, _val_loader, _ = dl.create_param_dataloaders(
        cfg["data_dir"], batch_size=1, context_frames=T, predict_frames=K,
        num_workers=0, param_choices=param_choices,
        train_file_limit=cfg.get("train_file_limit"), field_spec=cfg["field_spec"],
        traj_cache_capacity=8, shuffle_train=False,
    )

    target_indices = set(args.batch_indices)
    max_idx = max(target_indices)

    with torch.no_grad():
        for i, (xb, yb, bparams) in enumerate(train_loader):
            if i > max_idx:
                break
            if i not in target_indices:
                continue

            xb = xb.to(dev)
            print(f"\n{'='*70}\nbatch {i}  bparams={bparams[0].tolist()}")

            x_context = field_embedder(xb, cfg["field_spec"])
            cond = context_cond_encoder(x_context)
            z = encoder(x_context)
            print(f"  z0.real_grid: norm={z.real_grid.norm().item():.4f} max_abs={z.real_grid.abs().max().item():.4f}")

            for step in range(K):
                contributions = {}
                for name, term in operator.real_terms.items():
                    contributions[name] = term(z, cond)

                # AdvectionTerm internals specifically: the raw learned velocity
                # field `a`, before combining with dx/dy -- the unbounded
                # (no-tanh) FiLM path's own output magnitude.
                adv_term = operator.real_terms["advection"]
                a = adv_term.net(z.real_grid, cond)
                dx = dx_central(z.real_grid)
                dy = dy_central(z.real_grid)

                print(f"  step {step}:")
                print(f"    input z.real_grid: norm={z.real_grid.norm().item():.4f} max_abs={z.real_grid.abs().max().item():.4f}")
                print(f"    advection velocity field 'a' (unbounded FiLM output): "
                      f"norm={a.norm().item():.4f} max_abs={a.abs().max().item():.4f}")
                print(f"    dx_central(z): max_abs={dx.abs().max().item():.4f}   dy_central(z): max_abs={dy.abs().max().item():.4f}")
                for name, c in contributions.items():
                    print(f"    term[{name}] contribution: norm={c.norm().item():.4f} max_abs={c.abs().max().item():.4f} "
                          f"finite={torch.isfinite(c).all().item()}")

                real_out = z.real_grid
                for c in contributions.values():
                    real_out = real_out + c
                z = z.replace_state(real_grid=real_out)
                print(f"    -> z.real_grid AFTER step {step}: norm={z.real_grid.norm().item():.4f} "
                      f"max_abs={z.real_grid.abs().max().item():.4f} finite={torch.isfinite(z.real_grid).all().item()}")

                if not torch.isfinite(z.real_grid).all():
                    print("    !! non-finite state reached, stopping trace for this batch")
                    break


if __name__ == "__main__":
    main()
