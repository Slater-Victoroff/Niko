from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from core.blocks import ConvNeXtBlock
from encoders.sequence_conv import SequenceConvEncoder


class ContextCondEncoder(nn.Module):
    """Infers a FiLM conditioning vector directly from the raw context
    frames, without ever collapsing through an explicit human-defined
    physical-parameter bottleneck -- contrast with ContextParamRegressor
    (encoders/param_inference.py), which predicts Rayleigh/Prandtl
    explicitly via supervised regression against ground-truth labels.

    Drop-in alternative to param_encoder inside LatentDynamicsModel: same
    output contract ([B, cond_dim]), consumed by the exact same FiLM layers
    in the operator/decoder, but built from x_context instead of a Params
    object, and trained fully jointly with the rest of the model via the
    normal rollout/reconstruction task loss -- no separate supervision, no
    intermediate regression target at all.

    Rationale: forcing all conditioning information through exactly 2 (or
    however many) human-defined scalar parameters may be lossier than
    necessary -- especially given ContextParamRegressor's own finding that
    Prandtl is hard to pin down accurately from a short context window. This
    lets the model use the full cond_dim as a fixed-size interconnect
    between context and FiLM modulation, and learn whatever internal
    representation of "which physical regime is this" is actually useful
    for prediction, not constrained to literally equal (Rayleigh, Prandtl).
    A small MLP mapping this learned cond-space representation back to (and
    from) the literal physical params, for interpretability, is a natural
    follow-up -- not built here. Right now the only goal is prediction
    accuracy.
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        cond_dim: int,
        hidden_dim: int = 64,
        block_kernel_size: int = 7,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_frames = context_frames
        self.cond_dim = cond_dim

        total_in = in_channels * context_frames
        self.trunk = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim, kernel_size=block_kernel_size),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim, kernel_size=block_kernel_size),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C, H, W] -> [B, cond_dim]."""
        reshaped = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        f = self.trunk(reshaped)
        pooled = f.mean(dim=(-2, -1))
        return self.head(pooled)


class PooledContextCondEncoder(nn.Module):
    """Same goal as ContextCondEncoder (infer cond directly from context, no
    param bottleneck), but the same per-frame-trunk-then-pool architecture
    that beat the stacked-as-channels design decisively in the param-
    regression comparison (PooledSequenceParamRegressor vs
    ContextParamRegressor, encoders/param_inference.py): a SHARED conv trunk
    runs on every context frame independently (in_channels = the physical
    channel count only, not channels*T), each frame gets spatially pooled,
    and those per-frame vectors are mean-pooled over time before the cond
    head -- rather than stacking all frames into one big multi-channel
    input. Also sized up (default hidden_dim=128 vs ContextCondEncoder's 64)
    since the per-frame design has real headroom to use it.
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        cond_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_frames = context_frames
        self.cond_dim = cond_dim

        self.per_frame_trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C, H, W] -> [B, cond_dim]."""
        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        f = self.per_frame_trunk(frames)          # [B*T, hidden, H', W']
        pooled_spatial = f.mean(dim=(-2, -1))      # [B*T, hidden]
        pooled_spatial = pooled_spatial.view(b, t, -1)
        pooled_time = pooled_spatial.mean(dim=1)   # [B, hidden]
        return self.head(pooled_time)


class LatentContextCondEncoder(nn.Module):
    """Same shared-trunk-then-pool shape as PooledContextCondEncoder (the
    architecture that won: valid_loss 0.229, tying the ground-truth-param
    baseline), but the per-frame "trunk" is a FROZEN, separately pretrained
    single-frame autoencoder's encoder (SequenceConvEncoder with
    context_frames=1, trained via train_autoencoder_baseline.py to
    reconstruct individual frames -- see checkpoints/autoencoder_ctx1_broad8)
    instead of a from-scratch learnable conv stack operating on raw pixels.

    Each of the T context frames is encoded independently through the frozen
    pretrained encoder into a latent grid, a small NEW learnable conv
    ("refine_trunk") refines each per-frame latent, then the same two-stage
    pool (spatial over the latent grid, then temporal over the T frames) and
    head as PooledContextCondEncoder produce the final cond vector.

    Rationale: PooledContextCondEncoder's from-scratch trunk has to learn
    BOTH a useful per-frame feature extractor AND how to turn that into good
    FiLM conditioning, jointly, from the rollout loss alone. Handing it a
    latent space that's already been trained (via a much simpler, better-
    posed reconstruction objective) to preserve reconstruction-relevant
    per-frame structure might make the second, harder job easier.
    """

    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        ae_checkpoint: str,
        context_frames: int | None = None,  # unused (length-agnostic, like PooledContextCondEncoder) -- accepted so train.py's context_frames patch doesn't error
        ae_latent_dim: int = 16,
        ae_hidden_dim: int = 64,
        hidden_dim: int = 128,
        freeze_ae_encoder: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.freeze_ae_encoder = freeze_ae_encoder

        self.frame_encoder = SequenceConvEncoder(
            in_channels=in_channels, context_frames=1, latent_dim=ae_latent_dim, hidden_dim=ae_hidden_dim,
        )
        ckpt = torch.load(ae_checkpoint, map_location="cpu")
        self.frame_encoder.load_state_dict(ckpt["encoder_state_dict"])
        if freeze_ae_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad_(False)
            self.frame_encoder.eval()

        self.refine_trunk = nn.Sequential(
            nn.Conv2d(ae_latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),
            ConvNeXtBlock(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def train(self, mode: bool = True) -> "LatentContextCondEncoder":
        super().train(mode)
        if self.freeze_ae_encoder:
            self.frame_encoder.eval()  # keep frozen encoder in eval mode even when the rest of the model trains
        return self

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C, H, W] -> [B, cond_dim]."""
        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, 1, c, h, w)  # [B*T, 1, C, H, W] -- one "context frame" each, for the ctx1 encoder
        with torch.set_grad_enabled(not self.freeze_ae_encoder):
            z = self.frame_encoder(frames).grid  # [B*T, ae_latent_dim, H', W']
        f = self.refine_trunk(z)
        pooled_spatial = f.mean(dim=(-2, -1))
        pooled_spatial = pooled_spatial.view(b, t, -1)
        pooled_time = pooled_spatial.mean(dim=1)
        return self.head(pooled_time)


class BoundaryGeometryHead(nn.Module):
    """Infers per-axis boundary-condition-like padding weights from context -- a
    small, deliberately narrow bottleneck (nowhere near cond_dim's width) whose
    only job is "how should the boundary behave," not general-purpose conditioning.
    See core/boundary.py for the motivating problem and how these weights get used.

    2026-08-21: per-task heads, not one head shared across every task -- a single
    shared head was the original (wrong) design, and it showed: inspecting a
    trained 12-task checkpoint found euler_multi_quadrants_openBC (the only fully-
    open-both-axes task in the set) had learned essentially nothing, still sitting
    at ~99.9% circular after 5 epochs, while every other task (even genuinely
    periodic ones like shear_flow) had drifted toward "replicate" regardless of
    their actual BC. One shared Linear was evidently converging toward whatever
    blend helps the aggregate gradient across 12 wildly different tasks, not
    learning each task's own actual geometry -- exactly the same class of problem
    already fixed for FieldEmbedder (per-task copies instead of one shared
    embedder per field name). Same fix here: one independent head per task, each
    identically zero-inited (no warm-start-from-a-shared-prior step needed the way
    FieldEmbedder's per-field copies needed one, since every copy's zero-init here
    is already deterministic/identical regardless of task).

    Each head takes the same field-embedded context every other context-inferred
    module sees, mean-pools over time+space (cheap, no new trunk), maps to 2*3=6
    logits (3 per axis: zero/circular/replicate), softmax per axis independently.

    Zero-inited so every sample starts at exactly [0, 1, 0] per axis (100%
    circular) -- behaviorally identical to the old unconditional-roll/zero-pad
    code at initialization, matching this codebase's established zero-init
    convention for every other newly-added conditioned term (FiLMConvNeXtBlock's
    film layer, DiffusionTerm/SkewTerm's cond heads, complex_proj); only drifts
    away from pure-periodic as training finds reason to, independently per task.
    """

    def __init__(self, in_channels: int, tasks: List[str]):
        super().__init__()

        def _make_head() -> nn.Linear:
            head = nn.Linear(in_channels, 6)
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                bias = head.bias.view(2, 3)
                bias.zero_()
                bias[:, 1] = 8.0  # circular logit raised -- softmax([0,8,0]) ~= [3e-4, 0.9997, 3e-4]
            return head

        self.heads = nn.ModuleDict({name: _make_head() for name in tasks})

    def forward(self, x_context: Tensor, task: str):
        """x_context: [B, T, C, H, W] (field-embedded canonical context) ->
        (weights_x [B, 3], weights_y [B, 3])."""
        if task not in self.heads:
            raise ValueError(f"No boundary-geometry head registered for task '{task}'")
        pooled = x_context.mean(dim=(1, 3, 4))  # [B, C]
        logits = self.heads[task](pooled).view(-1, 2, 3)
        weights = logits.softmax(dim=-1)
        return weights[:, 0], weights[:, 1]
