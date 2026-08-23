import torch
import torch.nn as nn
from torch import Tensor

from typing import Literal, Optional

from core.modules import EncoderBase, validate_call
from core.states import LatentState, Params
from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock


class SequenceConvEncoder(EncoderBase):
    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
        block_kernel_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        total_in = in_channels * context_frames

        # 2026-08-23: two changes made together, both motivated by the gray_scott
        # "bubbles" pathology (EXPERIMENT_LOG.md) -- diagnostics on a corrupted
        # checkpoint found the blowup traced to THIS encoder's output (471x the
        # normal gradient norm on a bubbles batch, dwarfing every operator term
        # including the complex branch), not the operator/rollout at all, and not
        # extreme raw input values either (bubbles' raw A/B range is unremarkable
        # next to the other 5 patterns) -- more likely a genuinely harder
        # prediction target producing a large, consistently-directioned backward
        # gradient that survives norm-clipping across several consecutive
        # same-pattern batches (the loader clusters same-file batches together).
        #
        # 1) out_act: a final tanh bounds this encoder's own output for the first
        #    time ever -- nothing here previously capped z0's forward magnitude at
        #    all, unlike every operator term (which all route through
        #    _FiLMConvNet/_ConcatConvNet's own tanh). Saturating tanh's derivative
        #    also shrinks toward zero at the extremes, which directly damps runaway
        #    backward gradient through this exact path, not just forward magnitude.
        #    This is a real behavior change (not a zero-init no-op the way most of
        #    this codebase's other new mechanisms are) -- a freshly-initialized
        #    encoder's pre-tanh output already sits in a similar rough range, but
        #    an OLD checkpoint's saved weights will decode slightly differently
        #    under this code than when it was originally trained.
        #
        # 2) Zero-inited final layer -- unlike every other zero-init in this
        #    codebase (which zeroes an ADDITIVE PERTURBATION on top of an
        #    otherwise-intact base signal, e.g. DiffusionTerm's log_nu_head still
        #    yields a nonzero nu, HelmholtzRotationTerm's phase starts at a genuine
        #    no-op rotation with v itself untouched), this zeroes the base signal
        #    pathway itself: z0.real_grid starts at exactly 0 for every input,
        #    same "no-op is a good prior" logic as complex_proj's own zero-init
        #    (which now means the WHOLE initial LatentState, real and spectral, is
        #    zero at step 0). The model is a real no-op for every input at
        #    initialization -- decoder output doesn't depend on the input at all
        #    until this final layer's own weights move, which happens immediately
        #    (gradient reaches a zero-inited output layer's own weights directly,
        #    same mechanism as every other zero-init here), so this should cost a
        #    handful of steps of "blind" training, not a fundamentally broken
        #    optimization.
        self.net = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim, kernel_size=block_kernel_size),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.out_act = nn.Tanh()

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z = self.out_act(self.net(reshaped_context))
        return LatentState(real_grid=z)
