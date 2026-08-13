from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


def canonical_field_name(key: str) -> str:
    """"t0_fields/pressure" -> "pressure" -- the field_spec key's basename,
    shared across tasks whose raw HDF5 layout differs but whose physical
    field is the same (e.g. "velocity" appears under t1_fields/velocity in
    rayleigh_benard, shear_flow, and active_matter alike)."""
    return key.rsplit("/", 1)[-1]


class FieldEmbedder(nn.Module):
    """Maps a task's heterogeneous raw field channels into one canonical,
    task-agnostic per-pixel latent, via a shared registry of small per-field
    embedding MLPs (1x1-conv stacks) keyed by canonical field name --
    the same "velocity" embedder is reused by every task whose field_spec
    includes a velocity field, rather than each task getting its own
    from-scratch input embedding. A field with multiple raw components
    (e.g. velocity's 2, or active_matter's D/E tensors' 4) is embedded by
    ONE MLP taking all of its components at once, not one embedder per
    component.

    Fields present in a given forward call's field_spec are embedded
    independently then SUMMED into one canonical_dim-wide latent per pixel
    per frame -- keeps this fixed-width regardless of which task/how many
    field groups are present (unlike concatenation, which would make the
    shared trunk's input width vary by task).
    """

    def __init__(self, channel_specs: Dict[str, int], canonical_dim: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.canonical_dim = canonical_dim
        self.embedders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(n_components, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, canonical_dim, kernel_size=1),
            )
            for name, n_components in channel_specs.items()
        })

    def forward(self, x: Tensor, field_spec: List[dict]) -> Tensor:
        """x: [B, T, C_raw, H, W] (channels ordered per field_spec) ->
        [B, T, canonical_dim, H, W]."""
        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)

        total = None
        offset = 0
        for spec in field_spec:
            name = canonical_field_name(spec["key"])
            n = spec["n_components"]
            chunk = frames[:, offset:offset + n]
            offset += n
            if name not in self.embedders:
                raise ValueError(f"No embedder registered for field '{name}' (from key '{spec['key']}')")
            embedded = self.embedders[name](chunk)  # [B*T, canonical_dim, H, W]
            total = embedded if total is None else total + embedded

        if offset != c:
            raise ValueError(f"field_spec accounts for {offset} channels but x has {c}")

        return total.view(b, t, self.canonical_dim, h, w)


def rfft2_crop(x: Tensor, out_h: int, out_w: int) -> Tensor:
    """rfft2(x), cropped to the low-frequency spectrum a real signal of
    spatial size (out_h, out_w) would produce -- exact Fourier-domain
    downsampling (spectral pooling), not a learned/approximate one.

    The width axis is already a half-spectrum (rfft convention: index 0 =
    DC, increasing to Nyquist, no wraparound), so its low frequencies are a
    prefix. The height axis is a full spectrum (fftfreq order: 0..Nyquist,
    -Nyquist..-1), so its low frequencies sit at *both* ends -- keep the top
    and bottom blocks.

    norm="ortho" (1/sqrt(H*W) scaling instead of the unnormalized default)
    -- the DC term of an unnormalized rfft2 is the pixel sum, tens of
    thousands of times the field's spatial mean, wildly out of scale with
    the rest of the network's activations.
    """
    X = torch.fft.rfft2(x, norm="ortho")
    out_w_half = out_w // 2 + 1
    X = X[..., :out_w_half]
    h_half = out_h // 2
    X = torch.cat([X[..., :h_half, :], X[..., -h_half:, :]], dim=-2)
    return X
