"""Boundary-aware padding: a per-sample, learned soft blend of zero/circular/
replicate padding, instead of this codebase's previous hard-coded choice (either
unconditional torch.roll -- always circular -- in transport_terms.py's
dx_central/dy_central/laplacian and core/states.py's velocity_from_streamfunction,
or nn.Conv2d's default zero-padding everywhere else).

2026-08-19/20 investigation found this dataset spans at least 6 distinct boundary-
condition regimes per axis (periodic, wall-Dirichlet, wall-no-slip, open,
open-Neumann, mixed-asymmetric -- see EXPERIMENT_LOG.md), while every spatial
derivative/conv in the model treated every task identically regardless. This module
is the shared mechanism fixing that: see encoders/context_cond.py's
BoundaryGeometryHead for how the blend weights are produced (a small,
deliberately-bottlenecked head inferred from context, zero-inited so every sample
starts at exactly [0,1,0] -- 100% circular -- behaviorally identical to the old
code at initialization).

zeros ~ wall/Dirichlet (field pinned toward zero at the boundary), circular ~
periodic (true wraparound), replicate ~ open/Neumann (boundary value extends
outward rather than hitting an artificial wall). Not a perfect physical BC solver
for every regime in this dataset (e.g. the mixed-asymmetric acoustic_scattering
BCs aren't fully expressible as one blend), but a large improvement over "assume
every task is periodic" or "assume every task is zero-padded," and continuous/
differentiable so training can find whatever blend actually helps per task rather
than committing to a hard per-task lookup table.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def blended_pad(x: Tensor, weights_x: Tensor, weights_y: Tensor, pad: int = 1) -> Tensor:
    """Pad x ([B, C, H, W]) by `pad` pixels each spatial side, blending zero/circular/
    replicate padding per-sample instead of committing to one scheme.

    weights_x, weights_y: each [B, 3] softmax weights (order: zero, circular,
    replicate), independent per axis -- this dataset's x/y boundary conditions
    routinely differ (periodic-x/wall-y is the majority pattern here).

    Implementation: sequential 1D pads (x-axis then y-axis) -- standard for
    correct 2D corner behavior, and means the y-axis padding step naturally
    operates on already x-padded data (matches how joint circular/replicate 2D
    padding is normally built). At each step, all three padding variants are
    cheap to compute (F.pad is bandwidth-only; `pad` is small, e.g. 1 or 3) and
    blending them is a single weighted sum -- no need to avoid the "3x" cost
    some sketchier designs worry about, since it's 3x of a barely-larger-than-x
    tensor, not 3x of x itself.
    """
    b = x.shape[0]
    assert weights_x.shape == (b, 3), f"weights_x expected [{b},3], got {tuple(weights_x.shape)}"
    assert weights_y.shape == (b, 3), f"weights_y expected [{b},3], got {tuple(weights_y.shape)}"

    def _blend(t: Tensor, pad_arg, weights: Tensor) -> Tensor:
        zero = F.pad(t, pad_arg, mode="constant", value=0.0)
        circ = F.pad(t, pad_arg, mode="circular")
        repl = F.pad(t, pad_arg, mode="replicate")
        w = weights.to(dtype=t.dtype)[:, :, None, None, None]  # [B,3,1,1,1]
        stacked = torch.stack([zero, circ, repl], dim=1)  # [B,3,C,H,W']
        return (stacked * w).sum(dim=1)

    x_padded = _blend(x, (pad, pad, 0, 0), weights_x)  # pad W (last dim)
    return _blend(x_padded, (0, 0, pad, pad), weights_y)  # pad H (second-to-last)
