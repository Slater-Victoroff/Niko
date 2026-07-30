import math

import torch
import torch.nn as nn
from torch import Tensor

from typing import Literal, Optional

from core.modules import EncoderBase, validate_call
from core.states import LatentState, Params
from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock


class SplitEncoder(EncoderBase):
    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = int(hidden_dim // math.sqrt(2))

        total_in = in_channels * context_frames

        self.real_net = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),

            # Fixed k=2 downsample.
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.latent_dim, kernel_size=1),
        )

        self.complex_net = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 2 * self.latent_dim, kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)),
        )

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z = self.real_net(reshaped_context)
        c = self.complex_net(reshaped_context)
        re, im = c.chunk(2, dim=1)
        z_complex = torch.complex(re, im)
        return LatentState(real_grid=z, spectral_grid=z_complex)


class FusedSpectralEncoder(EncoderBase):
    """Real branch (real_net) and complex branch (rfft2 of the last context
    frame, see rfft2_crop) exactly as in FFTSplitEncoder, but fused into a
    single real latent right here in the encoder instead of being kept
    separate for the operator: irfft2 the complex branch back to real space,
    concat with real_net's output, and run a small conv net over the
    concatenation to produce the one output latent.

    Returns LatentState(real_grid=...) only -- no spectral_grid. This is a
    different commitment than FFTSplitEncoder/SplitEncoder: there's no
    persistent complex-space state left for the operator's complex_term
    (ComplexAmplitudeTerm/ComplexRotationTerm) to evolve across rollout
    steps, so this encoder is only meant to pair with a non-complex operator
    (e.g. film_helmholtz -- the same real terms as complex_operator, minus
    the complex rotation term that would have nothing to read).
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = int(hidden_dim // math.sqrt(2))

        total_in = in_channels * context_frames

        self.real_net = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.latent_dim, kernel_size=1),
        )

        self.complex_proj = nn.Conv2d(2 * in_channels, 2 * latent_dim, kernel_size=1)

        # concat(real_net output, irfft'd complex branch) -> latent_dim
        self.fuse_net = nn.Sequential(
            nn.Conv2d(2 * latent_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, latent_dim, kernel_size=1),
        )

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z_real = self.real_net(reshaped_context)

        last_frame = x[:, -1]
        spec = rfft2_crop(last_frame, out_h=z_real.shape[-2], out_w=z_real.shape[-1])
        spec_ri = torch.cat([spec.real, spec.imag], dim=1)
        proj = self.complex_proj(spec_ri)
        re, im = proj.chunk(2, dim=1)
        z_complex = torch.complex(re, im)

        z_complex_real = torch.fft.irfft2(z_complex, s=(z_real.shape[-2], z_real.shape[-1]), norm="ortho")

        fused = torch.cat([z_real, z_complex_real], dim=1)
        z_out = self.fuse_net(fused)

        return LatentState(real_grid=z_out)


def rfft2_crop(x: Tensor, out_h: int, out_w: int) -> Tensor:
    """rfft2(x), cropped to the low-frequency spectrum a real signal of spatial
    size (out_h, out_w) would produce -- i.e. exact Fourier-domain downsampling
    (spectral pooling), not a learned/approximate one.

    The width axis is already a half-spectrum (rfft convention: index 0 = DC,
    increasing to Nyquist, no wraparound), so its low frequencies are a prefix.
    The height axis is a full spectrum (fftfreq order: 0..Nyquist, -Nyquist..-1),
    so its low frequencies sit at *both* ends -- keep the top and bottom blocks.

    norm="ortho" (scales by 1/sqrt(H*W) instead of the unnormalized default) --
    the DC term of an unnormalized rfft2 is the pixel sum, ~H*W times the
    field's spatial mean (65,536x here at 128x512), wildly out of scale with
    the rest of the network's activations. Orthonormal scaling is the standard
    convention when FFT output feeds a learned model.
    """
    X = torch.fft.rfft2(x, norm="ortho")
    out_w_half = out_w // 2 + 1
    X = X[..., :out_w_half]
    h_half = out_h // 2
    X = torch.cat([X[..., :h_half, :], X[..., -h_half:, :]], dim=-2)
    return X


class FFTSplitEncoder(EncoderBase):
    """Same real_net (spatial ConvNeXt) branch as SplitEncoder, but the complex
    branch is fed from an actual rfft2 of the last context frame's raw physical
    channels -- cropped to real_net's output resolution (see rfft2_crop) -- and
    a single learned 1x1 projection from in_channels to latent_dim, instead of a
    from-scratch learned ConvNeXt stack trying to invent a complex latent from
    pixel space. Ablation of SplitEncoder's complex_net only; real_net path is
    identical.
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = int(hidden_dim // math.sqrt(2))

        total_in = in_channels * context_frames

        self.real_net = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.latent_dim, kernel_size=1),
        )

        # operates on [real; imag] stacked as 2*in_channels real channels,
        # per-frequency-bin (1x1) -- projects the raw FFT's channel count up
        # to 2*latent_dim, split back into real/imag halves.
        self.complex_proj = nn.Conv2d(2 * in_channels, 2 * latent_dim, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z = self.real_net(reshaped_context)

        last_frame = x[:, -1]  # [B, in_channels, H, W] -- raw physical channels, most recent context frame only
        spec = rfft2_crop(last_frame, out_h=z.shape[-2], out_w=z.shape[-1])
        spec_ri = torch.cat([spec.real, spec.imag], dim=1)
        proj = self.complex_proj(spec_ri)
        re, im = proj.chunk(2, dim=1)
        z_complex = torch.complex(re, im)

        return LatentState(real_grid=z, spectral_grid=z_complex)


class FFTSplitEncoderWide(EncoderBase):
    """Same as FFTSplitEncoder, but complex_proj gets real learned depth
    instead of a single 1x1 conv: input 1x1 conv -> 2x ConvNeXtBlock ->
    output 1x1 conv, operating on the FFT-cropped spectrum. This mirrors
    SplitEncoder's original complex_net *refinement* depth (it had 2
    ConvNeXtBlocks too) minus its downsample conv, which rfft2_crop already
    does exactly/for free -- so this isn't matching complex_net's parameter
    count for its own sake, it's the same learned-refinement budget applied
    to a real spectrum instead of raw pixels. ~39K complex-branch params
    (vs FFTSplitEncoder's 288 and SplitEncoder's complex_net's ~82K).
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = int(hidden_dim // math.sqrt(2))

        total_in = in_channels * context_frames

        self.real_net = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            ConvNeXtBlock(self.hidden_dim),

            nn.Conv2d(self.hidden_dim, self.latent_dim, kernel_size=1),
        )

        self.complex_proj = nn.Sequential(
            nn.Conv2d(2 * in_channels, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            ConvNeXtBlock(self.hidden_dim),
            ConvNeXtBlock(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 2 * latent_dim, kernel_size=1),
        )
        # Zero-init the output layer -- matches FiLMConvNeXtBlock's own
        # zero-inited FiLM layer and DirectFieldOperator's zero-inited head
        # elsewhere in this codebase: start as a no-op (spectral_grid ~ 0)
        # and let training grow the complex branch's contribution gradually,
        # instead of injecting an uncontrolled random-scale perturbation
        # into the operator's multiplicative amplitude/rotation terms from
        # step one.
        nn.init.zeros_(self.complex_proj[-1].weight)
        nn.init.zeros_(self.complex_proj[-1].bias)

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z = self.real_net(reshaped_context)

        last_frame = x[:, -1]
        spec = rfft2_crop(last_frame, out_h=z.shape[-2], out_w=z.shape[-1])
        spec_ri = torch.cat([spec.real, spec.imag], dim=1)
        proj = self.complex_proj(spec_ri)
        re, im = proj.chunk(2, dim=1)
        z_complex = torch.complex(re, im)

        return LatentState(real_grid=z, spectral_grid=z_complex)
