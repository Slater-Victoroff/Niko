import torch.nn as nn
from torch import Tensor

from core.blocks import ConvNeXtBlock


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_frames = context_frames
        self.cond_dim = cond_dim

        total_in = in_channels * context_frames
        self.trunk = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=4, stride=2, padding=1),
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
