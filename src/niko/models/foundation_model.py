from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.modules import EncoderBase, OperatorBase, DecoderBase
from core.states import LatentState
from encoders.field_embedder import FieldEmbedder, rfft2_crop


class FoundationModel(nn.Module):
    """LatentDynamicsModel's shared-trunk / per-task-decoder sibling: one
    field_embedder + context_cond_encoder + encoder + operator (the "trunk",
    trained jointly across every task) feeding into a separate decoder per
    task (kept fully task-specific, matching each task's own field/decoder
    config -- rayleigh_benard and shear_flow's decoders differ in field
    names even though they share a channel count, and active_matter's has
    extra tensor-field heads entirely).

    context_cond_encoder (not param_encoder) is required: ground-truth
    params differ in count/meaning across tasks (2 log10 params for
    rayleigh_benard/shear_flow, 3 mixed-transform params for active_matter),
    so a single shared trunk needs a conditioning source that doesn't depend
    on that -- context-inferred cond sidesteps it entirely, and this
    session's own probes found it matches ground-truth-param conditioning
    quality on rayleigh_benard.

    If `operator.complex_term` is set, also builds a shared `complex_proj`:
    an rfft2 (cropped to the encoder's post-downsample resolution) of the
    embedded canonical context's last frame, projected to a complex
    spectral_grid attached to z0 before rollout. Operating on the embedded
    canonical_dim representation (not raw per-task pixels) keeps complex_proj
    one shared module regardless of task, same as the real encoder path.
    Once attached, LatentState.grid fuses real_grid + irfft2(spectral_grid)
    automatically (see core/states.py) -- nothing else here needs to know
    a complex branch exists at all.

    forward() takes a `task` key selecting which decoder to route through;
    field_embedder/context_cond_encoder/encoder/operator (and complex_proj,
    if present) are always the same shared modules regardless of task.
    """

    def __init__(
        self,
        field_embedder: FieldEmbedder,
        context_cond_encoder: nn.Module,
        encoder: EncoderBase,
        operator: OperatorBase,
        decoders: Dict[str, DecoderBase],
    ):
        super().__init__()
        self.field_embedder = field_embedder
        self.context_cond_encoder = context_cond_encoder
        self.encoder = encoder
        self.operator = operator
        self.decoders = nn.ModuleDict(decoders)

        self.complex_proj = None
        if getattr(operator, "complex_term", False):
            self.complex_proj = nn.Conv2d(
                2 * field_embedder.canonical_dim, 2 * encoder.latent_dim, kernel_size=1,
            )
            # Zero-init: an EXPERIMENT_LOG-documented fix for this exact branch. An
            # un-zero-inited complex_proj feeds an uncontrolled random-scale spectral_grid
            # into the operator's multiplicative exp(amplitude + i*rotation) update every
            # rollout step -- compounding over K steps, this is what caused the original
            # complex-branch attempt to diverge. Starts spectral_grid at exactly zero (a
            # true no-op through the irfft2 fusion in LatentState.grid) and only grows a
            # signal as training finds it useful, matching every other newly-conditioned
            # term's zero-init convention in this codebase.
            nn.init.zeros_(self.complex_proj.weight)
            nn.init.zeros_(self.complex_proj.bias)

        trunk_modules = [field_embedder, context_cond_encoder, encoder, operator]
        if self.complex_proj is not None:
            trunk_modules.append(self.complex_proj)
        trunk_params = sum(p.numel() for m in trunk_modules for p in m.parameters())
        print(f"FoundationModel shared trunk params: {trunk_params}")
        for name, dec in self.decoders.items():
            print(f"  decoder[{name}] params: {sum(p.numel() for p in dec.parameters())}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"FoundationModel total params (trunk + all decoders): {total_params}")

    def rollout_latent(self, z0: LatentState, steps: int, cond: Optional[Tensor] = None) -> List[LatentState]:
        preds = []
        z = z0
        for _ in range(steps):
            z = self.operator(z, cond=cond)
            preds.append(z)
        return preds

    def forward(
        self,
        x_context_raw: Tensor,
        field_spec: List[dict],
        task: str,
        steps: int,
        return_initial_encode: bool = True,
    ) -> Tensor:
        if task not in self.decoders:
            raise ValueError(f"Unknown task '{task}'; known tasks: {list(self.decoders.keys())}")

        x_context = self.field_embedder(x_context_raw, field_spec)  # [B, T, canonical_dim, H, W]
        cond = self.context_cond_encoder(x_context)
        z0 = self.encoder(x_context)

        if self.complex_proj is not None:
            last_frame = x_context[:, -1]  # [B, canonical_dim, H, W]
            spec = rfft2_crop(last_frame, out_h=z0.real_grid.shape[-2], out_w=z0.real_grid.shape[-1])
            spec_ri = torch.cat([spec.real, spec.imag], dim=1)
            proj = self.complex_proj(spec_ri)
            re, im = proj.chunk(2, dim=1)
            z0 = z0.replace_state(real_grid=z0.real_grid, spectral_grid=torch.complex(re, im))

        zs = self.rollout_latent(z0, steps=steps, cond=cond)

        decoder = self.decoders[task]
        zs_stacked = torch.stack([z.grid for z in zs], dim=1)
        B, T = zs_stacked.shape[:2]
        zs_flat = zs_stacked.reshape(B * T, *zs_stacked.shape[2:])
        cond_flat = cond[:, None, :].expand(B, T, cond.shape[-1]).reshape(B * T, cond.shape[-1])
        response_flat = decoder(zs_flat, cond=cond_flat)
        response = response_flat.reshape(B, T, *response_flat.shape[1:])

        if return_initial_encode:
            decoded_last_frame = decoder(z0.grid, cond=cond)
            return decoded_last_frame, response
        return response
