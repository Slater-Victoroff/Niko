import copy
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

    def __init__(self, tasks: Dict[str, dict], canonical_dim: int = 12, hidden_dim: int = 32,
                 use_batchnorm: bool = True):
        super().__init__()
        self.canonical_dim = canonical_dim
        self.use_batchnorm = use_batchnorm

        # Discover each canonical field's n_components (consistency-checked across
        # tasks, same check this file used to do via train_foundation.py's
        # union_channel_specs) and which tasks actually use it.
        field_n_components: Dict[str, int] = {}
        field_tasks: Dict[str, List[str]] = {}
        for task_name, cfg in tasks.items():
            for spec in cfg["field_spec"]:
                name = canonical_field_name(spec["key"])
                n = spec["n_components"]
                if name in field_n_components and field_n_components[name] != n:
                    raise ValueError(
                        f"Field '{name}' has conflicting n_components across tasks: "
                        f"{field_n_components[name]} vs {n} (task '{task_name}')"
                    )
                field_n_components[name] = n
                field_tasks.setdefault(name, []).append(task_name)

        def _make_embedder(n_components: int) -> nn.Sequential:
            layers = []
            if use_batchnorm:
                # Adaptive per-field input scaling, not a fixed external one (a
                # precomputed stats.yaml z-score caused a real divergence -- its
                # fixed epsilon floor blew up a near-constant field's noise).
                # affine=False: the following Conv2d already has its own bias.
                layers.append(nn.BatchNorm2d(n_components, affine=False))
            layers += [
                nn.Conv2d(n_components, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, canonical_dim, kernel_size=1),
            ]
            return nn.Sequential(*layers)

        # 2026-08-20: per-(task, field) embedder copies instead of one embedder
        # shared identically across every task using that field name.
        # EXPERIMENT_LOG-documented finding: "velocity"/"pressure" were being
        # funneled through one shared embedder despite meaning genuinely different
        # things per task (incompressible streamfunction-derived flow vs.
        # linearized acoustic particle velocity vs. compressible NS velocity;
        # gauge-free fluctuation pressure vs. real-mean compressible pressure --
        # the decoder side already treats these differently per zero_mean_pressure/
        # use_streamfunction, the encoder side didn't). Every task's copy is
        # weight-IDENTICAL at initialization (deep-copied from one shared prior per
        # field, not independently random-initialized) -- a task starts with
        # "the pressure embedding we already have," then specializes
        # independently as its own data demands, rather than being permanently
        # forced to share representation with every other task that has a field
        # of the same name.
        self.embedders = nn.ModuleDict()
        for name, n_components in field_n_components.items():
            prior = _make_embedder(n_components)
            per_task = nn.ModuleDict({
                task_name: copy.deepcopy(prior) for task_name in field_tasks[name]
            })
            self.embedders[name] = per_task

    def forward(self, x: Tensor, field_spec: List[dict], task: str) -> Tensor:
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
            if task not in self.embedders[name]:
                raise ValueError(f"No embedder for field '{name}' registered for task '{task}'")
            embedded = self.embedders[name][task](chunk)  # [B*T, canonical_dim, H, W]
            total = embedded if total is None else total + embedded

        if offset != c:
            raise ValueError(f"field_spec accounts for {offset} channels but x has {c}")

        return total.view(b, t, self.canonical_dim, h, w)


def state_dict_uses_batchnorm(field_embedder_state_dict: Dict[str, Tensor]) -> bool:
    """A checkpoint's field_embedder_state_dict predates FieldEmbedder's
    BatchNorm2d addition iff it has no running_mean/running_var buffers --
    lets probes auto-select use_batchnorm to match whichever checkpoint
    they're loading, rather than hardcoding an assumption that goes stale
    the next time this module's architecture changes again."""
    return any(k.endswith("running_mean") for k in field_embedder_state_dict)


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
