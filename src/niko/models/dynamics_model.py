from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor
from contextlib import contextmanager
import time

from core.modules import ParamEncoderBase, EncoderBase, OperatorBase, DecoderBase
from core.states import LatentState, Params
import logging


logger = logging.getLogger(__name__)


def _tensor_summary(name: str, t: Optional[Tensor]) -> str:
    if t is None:
        return f"{name}: None"
    try:
        # reduce stats to scalars
        nm = torch.isnan(t).any().item()
        inf = torch.isinf(t).any().item()
        finite = torch.isfinite(t).all().item()
        t_min = float(t.min())
        t_max = float(t.max())
        t_mean = float(t.mean())
        return (
            f"{name}: shape={tuple(t.shape)} device={t.device} dtype={t.dtype} "
            f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} "
            f"has_nan={nm} has_inf={inf} all_finite={finite}"
        )
    except Exception as e:
        return f"{name}: (could not summarize: {e})"


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderBase,
        operator: OperatorBase,
        decoder: DecoderBase,
        param_encoder: Optional[ParamEncoderBase] = None,
        context_cond_encoder: Optional[nn.Module] = None,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        if (param_encoder is None) == (context_cond_encoder is None):
            raise ValueError(
                "LatentDynamicsModel requires exactly one of param_encoder (cond from ground-truth "
                "Params) or context_cond_encoder (cond inferred directly from x_context), not both/neither."
            )
        self.param_encoder = param_encoder
        self.context_cond_encoder = context_cond_encoder
        self.encoder = encoder
        self.operator = operator
        self.decoder = decoder
        # Gradient-checkpoint each rollout step instead of keeping its activations
        # resident for the whole rollout -- trades ~20-30% more compute for memory that
        # no longer scales with rollout length, so long rollouts (roll16+) can train at
        # the same batch size as short ones. False (default) changes nothing about
        # existing configs/checkpoints -- same math either way, just when the graph gets
        # freed. See rollout_latent/_checkpointed_step.
        self.use_checkpointing = use_checkpointing

        # log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print overall and per-component parameter counts
        component_names = ["param_encoder", "context_cond_encoder", "encoder", "operator", "decoder"]
        for name in component_names:
            m = getattr(self, name, None)
            if m is None:
                print(f"{name} params: total=0, trainable=0")
            else:
                total_c = sum(p.numel() for p in m.parameters())
                trainable_c = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"{name} params: total={total_c}, trainable={trainable_c}")

        print(f"LatentDynamicsModel params: total={total_params}, trainable={trainable_params}")

    def rollout_latent(
        self,
        z0: LatentState,
        steps: int,
        cond: Optional[Tensor] = None,
        n_substeps: int = 1,
    ) -> List[LatentState]:
        # n_substeps=1 (default) is exactly the original behavior, unchanged for
        # every existing config/checkpoint: one full-dt operator call per output
        # frame. n_substeps>1 instead calls the operator n_substeps times per
        # output frame at dt=1/n_substeps each -- a DISCO-inspired experiment
        # (see EXPERIMENT_LOG.md, and TransportOperator.forward's own dt docstring)
        # giving each term smaller, more numerically-stable sub-steps instead of
        # one large discrete jump. Only the intermediate substeps are hidden from
        # `preds` -- one entry per REQUESTED output frame (`steps`), same contract
        # as before, decoded at the same points regardless of n_substeps.
        dt = 1.0 / n_substeps
        preds = []
        z = z0
        for i in range(steps):
            for _ in range(n_substeps):
                if self.use_checkpointing and self.training:
                    z = self._checkpointed_step(z, cond, dt=dt)
                else:
                    z = self.operator(z, cond=cond, dt=dt)
            preds.append(z)
        return preds

    def _checkpointed_step(self, z: LatentState, cond: Optional[Tensor], dt: float = 1.0) -> LatentState:
        """One gradient-checkpointed operator call: torch.utils.checkpoint
        needs a plain-tensor-in/plain-tensor-out function, so this
        reconstructs a LatentState around real_grid inside the checkpointed
        closure and unpacks it back out after. Only handles the
        real_grid-only case -- spectral_grid is always None for
        LatentDynamicsModel currently (its complex-branch wiring was removed
        in the consolidation, see EXPERIMENT_LOG.md) -- and falls back to an
        uncheckpointed call if it's ever not, rather than silently dropping
        a complex branch.
        """
        if z.spectral_grid is not None:
            return self.operator(z, cond=cond, dt=dt)

        aux, meta = z.aux, z.meta

        def fn(real_grid: Tensor, cond: Tensor) -> Tensor:
            z_in = LatentState(real_grid=real_grid, spectral_grid=None, aux=aux, meta=meta)
            return self.operator(z_in, cond=cond, dt=dt).real_grid

        real_grid_out = torch.utils.checkpoint.checkpoint(fn, z.real_grid, cond, use_reentrant=False)
        return LatentState(real_grid=real_grid_out, spectral_grid=None, aux=aux, meta=meta)

    def forward(
        self,
        x_context: Tensor,
        steps: int,
        params: Optional[Params] = None,
        return_initial_encode: bool = True,
        debug_timing: bool = False,
        n_substeps: int = 1,
    ) -> Tensor:
        @contextmanager
        def time_block(timings: dict, key: str):
            if debug_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if debug_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings[key] = time.perf_counter() - t0

        if self.context_cond_encoder is not None:
            cond = self.context_cond_encoder(x_context)
        else:
            cond = self.param_encoder(params)
        timings: dict[str, float] = {}

        # encoder
        with time_block(timings, "encoder"):
            if hasattr(self.encoder, "cond_dim") and self.encoder.cond_dim is not None:
                assert cond.shape[1] == self.encoder.cond_dim, f"param_encoder output dim {cond.shape[1]} does not match encoder cond_dim {self.encoder.cond_dim}"
                z0 = self.encoder(x_context, cond=cond)
            else:
                z0 = self.encoder(x_context)

        # rollout
        with time_block(timings, "rollout"):
            zs = self.rollout_latent(z0, steps=steps, cond=cond, n_substeps=n_substeps)

        # decode each latent into a raw tensor [B, C, H, W]
        with time_block(timings, "decoder"):
            # Stack latent states and decode in one batch
            zs_stacked = torch.stack([z.grid for z in zs], dim=1)
            # Merge batch and steps dimensions for decoder: [B, T, C, H, W] -> [B*T, C, H, W]
            B, T = zs_stacked.shape[:2]
            zs_flat = zs_stacked.reshape(B * T, *zs_stacked.shape[2:])
            cond_flat = cond[:, None, :].expand(B, T, cond.shape[-1]).reshape(B * T, cond.shape[-1]) if cond is not None else None
            if self.use_checkpointing and self.training:
                # Same exact-gradient checkpointing as rollout_latent's operator steps (see
                # _checkpointed_step). Chunked along the flattened B*T batch, not one single
                # checkpointed call over all of it -- checkpointing only changes WHEN
                # activations exist (freed after the forward pass, recomputed during
                # backward), not their PEAK size for whatever's inside one checkpoint call;
                # a single B*T=128-frame decode call still recomputes all 128 frames at once
                # during backward, so its peak memory is unchanged from the uncheckpointed
                # case. Chunking bounds each checkpointed call (and thus each recompute) to
                # `chunk` frames, which is what actually shrinks the peak for long rollouts.
                chunk = max(1, B)  # one rollout-step's worth of frames per checkpointed call
                response_chunks = [
                    torch.utils.checkpoint.checkpoint(
                        self.decoder, zs_flat[i:i + chunk],
                        cond_flat[i:i + chunk] if cond_flat is not None else None,
                        use_reentrant=False)
                    for i in range(0, zs_flat.shape[0], chunk)
                ]
                response_flat = torch.cat(response_chunks, dim=0)
            else:
                response_flat = self.decoder(zs_flat, cond=cond_flat)
            # Restore original batch and steps dimensions: [B*T, ...] -> [B, T, ...]
            response = response_flat.reshape(B, T, *response_flat.shape[1:])

        if debug_timing:
            total = sum(timings.values())
            print(
                f"Timing (s) encoder={timings.get('encoder', 0.0):0.4f} "
                f"rollout={timings.get('rollout', 0.0):0.4f} "
                f"decoder={timings.get('decoder', 0.0):0.4f} "
                f"total={total:0.4f}"
            )

        if return_initial_encode:
            decoded_last_frame = self.decoder(z0.grid, cond=cond)
            return decoded_last_frame, response
        return response
