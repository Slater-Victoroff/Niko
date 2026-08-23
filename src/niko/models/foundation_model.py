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
        boundary_geometry: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.field_embedder = field_embedder
        self.context_cond_encoder = context_cond_encoder
        self.encoder = encoder
        self.operator = operator
        self.decoders = nn.ModuleDict(decoders)
        # See encoders/context_cond.py's BoundaryGeometryHead / core/boundary.py.
        # Optional (None disables it entirely, falling back to every spatial
        # derivative/conv's old unconditional-circular-or-zero-pad behavior) so
        # existing call sites/checkpoints that don't pass this keep working
        # unchanged.
        self.boundary_geometry = boundary_geometry

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
        if self.boundary_geometry is not None:
            trunk_modules.append(self.boundary_geometry)
        trunk_params = sum(p.numel() for m in trunk_modules for p in m.parameters())
        print(f"FoundationModel shared trunk params: {trunk_params}")
        for name, dec in self.decoders.items():
            print(f"  decoder[{name}] params: {sum(p.numel() for p in dec.parameters())}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"FoundationModel total params (trunk + all decoders): {total_params}")

    def rollout_latent(
        self, z0: LatentState, steps: int, cond: Optional[Tensor] = None,
        bc_weights: Optional[tuple] = None, n_substeps: int = 1,
    ) -> List[LatentState]:
        # n_substeps=1 (default) is exactly the original behavior, unchanged for
        # every existing config/checkpoint: one full-dt operator call per output
        # frame. n_substeps>1 instead calls the operator n_substeps times per
        # output frame at dt=1/n_substeps each -- ported from
        # LatentDynamicsModel.rollout_latent (models/dynamics_model.py), the
        # single-task path this was originally developed/validated on -- see its
        # docstring and EXPERIMENT_LOG.md §20/§23 for the DISCO-inspired
        # motivation. Only the intermediate substeps are hidden from `preds` --
        # one entry per REQUESTED output frame (`steps`), same contract as
        # before, decoded at the same points regardless of n_substeps.
        dt = 1.0 / n_substeps
        preds = []
        z = z0
        for _ in range(steps):
            for _ in range(n_substeps):
                z = self.operator(z, cond=cond, bc_weights=bc_weights, dt=dt)
            preds.append(z)
        return preds

    @staticmethod
    def _broadcast_cond_bc(cond: Tensor, bc_weights: Optional[tuple], B: int, N: int):
        """cond/bc_weights are one-per-sample ([B, ...]); every per-position call this
        class makes (N rollout steps in forward(), N dense-single-step pairs in
        forward_dense_singlestep()) needs the SAME cond/bc_weights repeated across
        that position axis before flattening position into the batch dim for a
        single decoder/operator call. Shared here so both call sites can't drift
        apart on this broadcasting logic."""
        cond_flat = cond[:, None, :].expand(B, N, cond.shape[-1]).reshape(B * N, cond.shape[-1])
        if bc_weights is not None:
            wx, wy = bc_weights
            bc_weights_flat = (
                wx[:, None, :].expand(B, N, 3).reshape(B * N, 3),
                wy[:, None, :].expand(B, N, 3).reshape(B * N, 3),
            )
        else:
            bc_weights_flat = None
        return cond_flat, bc_weights_flat

    def forward(
        self,
        x_context_raw: Tensor,
        field_spec: List[dict],
        task: str,
        steps: int,
        return_initial_encode: bool = True,
        n_substeps: int = 1,
        return_latents: bool = False,
    ) -> Tensor:
        if task not in self.decoders:
            raise ValueError(f"Unknown task '{task}'; known tasks: {list(self.decoders.keys())}")

        x_context = self.field_embedder(x_context_raw, field_spec, task)  # [B, T, canonical_dim, H, W]
        cond = self.context_cond_encoder(x_context)
        z0 = self.encoder(x_context)
        bc_weights = self.boundary_geometry(x_context, task) if self.boundary_geometry is not None else None

        if self.complex_proj is not None:
            last_frame = x_context[:, -1]  # [B, canonical_dim, H, W]
            spec = rfft2_crop(last_frame, out_h=z0.real_grid.shape[-2], out_w=z0.real_grid.shape[-1])
            spec_ri = torch.cat([spec.real, spec.imag], dim=1)
            proj = self.complex_proj(spec_ri)
            re, im = proj.chunk(2, dim=1)
            z0 = z0.replace_state(real_grid=z0.real_grid, spectral_grid=torch.complex(re, im))

        zs = self.rollout_latent(z0, steps=steps, cond=cond, bc_weights=bc_weights, n_substeps=n_substeps)

        decoder = self.decoders[task]
        zs_stacked = torch.stack([z.grid for z in zs], dim=1)
        B, T = zs_stacked.shape[:2]
        zs_flat = zs_stacked.reshape(B * T, *zs_stacked.shape[2:])
        cond_flat, bc_weights_flat = self._broadcast_cond_bc(cond, bc_weights, B, T)
        response_flat = decoder(zs_flat, cond=cond_flat, bc_weights=bc_weights_flat)
        response = response_flat.reshape(B, T, *response_flat.shape[1:])

        # return_latents: opt-in, off by default (zero cost/shape change on the normal
        # training/eval path) -- exposes the intermediate rollout LatentStates for
        # diagnostics (see training/diagnostics.py's rollout_drift_stats) without
        # changing what any existing caller gets back. Always appended as the LAST
        # element, on top of whatever return_initial_encode's own shape already is,
        # so callers that don't ask for it are completely unaffected.
        if return_initial_encode:
            decoded_last_frame = decoder(z0.grid, cond=cond, bc_weights=bc_weights)
            result = (decoded_last_frame, response)
        else:
            result = response
        if return_latents:
            return (*result, zs) if isinstance(result, tuple) else (result, zs)
        return result

    def forward_dense_singlestep(
        self,
        x_context_raw: Tensor,
        x_target_raw: Tensor,
        field_spec: List[dict],
        task: str,
        n_substeps: int = 1,
    ) -> tuple:
        """Decouples "how much context informs cond" from "how many single-step
        training pairs one window yields" -- context_cond_encoder still pools the
        FULL context window (context_frames=T, e.g. 16) for a well-informed cond
        ("tighter bound on the operator"), but self.encoder here must be a
        SEPARATELY built module with encoder_context_frames=1 (see
        build_foundation_model's encoder_context_frames), since it's applied to
        one raw frame at a time, not the T-frame-stacked input self.encoder gets
        in forward(). Every consecutive pair inside the T-context + K-target
        window becomes its own directly-supervised single-step example (T+K-1 of
        them, e.g. 16 when T=16/K=1) instead of only supervising the one final
        transition -- reuses exactly the same (T-frame-context, K-frame-target)
        batch train_foundation.py's dataloaders already produce; no new
        windowing/data-loading logic needed, just consuming the raw frames that
        were already being loaded and only partially used.

        Motivated by a real, measured result (see EXPERIMENT_LOG.md and this
        session's per-step vrmse breakdown): a K-step-averaged rollout's own
        early steps are meaningfully better than its later ones (compounding
        rollout error), and a model trained head-on for single-step prediction
        (ctx1/roll1) does noticeably better at t+1 specifically than the same
        architecture getting t+1 "for free" from a longer-context/longer-rollout
        model -- but ctx1 also throws away everything a longer context window
        could have told the model about which physical regime it's in. This
        keeps the long-context conditioning benefit while still training
        head-on for single-step accuracy at every available transition, not
        just the final one.

        Returns (pred, target_raw), both [B, T+K-1, C_raw, H, W] -- callers
        compute their own loss the same way every other call site here does,
        typically well_style_vrmse(pred, target_raw).mean().
        """
        if task not in self.decoders:
            raise ValueError(f"Unknown task '{task}'; known tasks: {list(self.decoders.keys())}")

        x_context = self.field_embedder(x_context_raw, field_spec, task)  # [B, T, canonical_dim, H, W]
        x_target = self.field_embedder(x_target_raw, field_spec, task)  # [B, K, canonical_dim, H, W]
        cond = self.context_cond_encoder(x_context)  # from the FULL context window, same as forward()
        bc_weights = self.boundary_geometry(x_context, task) if self.boundary_geometry is not None else None

        all_canonical = torch.cat([x_context, x_target], dim=1)  # [B, T+K, canonical_dim, H, W]
        all_raw = torch.cat([x_context_raw, x_target_raw], dim=1)  # [B, T+K, C_raw, H, W]
        B, TK = all_canonical.shape[:2]
        N = TK - 1  # number of consecutive (frame_i, frame_{i+1}) pairs available in this window

        # Fold position N into the batch dim -- one encoder/operator/decoder call
        # covers every position at once, same "flatten position, run once, reshape
        # back" idiom forward() already uses for its K rollout-step decode.
        inputs_flat = all_canonical[:, :N].reshape(B * N, 1, *all_canonical.shape[2:])
        z0_flat = self.encoder(inputs_flat)  # encoder built with context_frames=1 -- see docstring
        cond_flat, bc_weights_flat = self._broadcast_cond_bc(cond, bc_weights, B, N)

        if self.complex_proj is not None:
            # forward()'s equivalent step uses the LAST frame of a multi-frame context
            # as complex_proj's source; here every position only ever has its own
            # single frame, so that frame IS the natural per-position analog.
            frame_flat = inputs_flat[:, 0]  # [B*N, canonical_dim, H, W]
            spec = rfft2_crop(frame_flat, out_h=z0_flat.real_grid.shape[-2], out_w=z0_flat.real_grid.shape[-1])
            spec_ri = torch.cat([spec.real, spec.imag], dim=1)
            proj = self.complex_proj(spec_ri)
            re, im = proj.chunk(2, dim=1)
            z0_flat = z0_flat.replace_state(real_grid=z0_flat.real_grid, spectral_grid=torch.complex(re, im))

        dt = 1.0 / n_substeps
        z1_flat = z0_flat
        for _ in range(n_substeps):
            z1_flat = self.operator(z1_flat, cond=cond_flat, bc_weights=bc_weights_flat, dt=dt)

        pred_flat = self.decoders[task](z1_flat.grid, cond=cond_flat, bc_weights=bc_weights_flat)
        pred = pred_flat.reshape(B, N, *pred_flat.shape[1:])
        target_raw = all_raw[:, 1:TK]  # each position's own immediate next frame
        return pred, target_raw
