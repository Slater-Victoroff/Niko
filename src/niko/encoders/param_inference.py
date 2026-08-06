import torch
import torch.nn as nn
from torch import Tensor

from core.blocks import ConvNeXtBlock
from core.states import Params


class _ParamTransformMixin:
    """Shared invert_transform()/to_params() for any module that predicts
    physical params in their own transform space (see ContextParamRegressor's
    docstring) -- factored out so PooledSequenceParamRegressor doesn't
    duplicate it verbatim.
    """

    transforms: list[str]

    def invert_transform(self, pred: Tensor) -> Tensor:
        """[B, n_params] in transform space -> raw physical values, inverting
        each column with its own param's transform (mirrors
        ScalarParamHead.transform_input in encoders/param_encoder.py)."""
        cols = []
        for i, t in enumerate(self.transforms):
            col = pred[:, i]
            if t == "identity":
                cols.append(col)
            elif t == "log10":
                cols.append(10.0 ** col)
            elif t == "log":
                cols.append(torch.exp(col))
            elif t == "signed_log10":
                cols.append(torch.sign(col) * (10.0 ** col.abs() - 1.0))
            else:
                raise ValueError(f"Unknown transform: {t}")
        return torch.stack(cols, dim=-1)

    def to_params(self, x: Tensor, names: list[str] | None = None) -> Params:
        """Drop-in replacement for a dataset's ground-truth Params object at
        inference time: infers raw param values from context alone."""
        raw = self.invert_transform(self.forward(x))
        return Params(values=raw, names=names)


class ContextParamRegressor(_ParamTransformMixin, nn.Module):
    """Infers physical simulation parameters directly from the raw context
    frames, so they don't have to be passed in explicitly at test time --
    trained as a standalone supervised regression (predicted vs ground-truth
    params) via train_param_inference.py, fully decoupled from the main
    dynamics model's own training: this is a well-posed problem with free
    ground-truth labels, so there's no reason to route the learning signal
    indirectly through the operator/decoder's rollout loss.

    Predicts in each param's own *transform* space (matching whichever
    param_encoder the target dynamics model uses -- e.g. log10 for
    rayleigh_benard's Rayleigh/Prandtl) rather than raw physical units:
    Rayleigh alone spans 4 orders of magnitude, so a plain MSE on raw values
    would be dominated by scale. invert_transform()/to_params() map back to
    raw values for splicing into a Params object at inference time.

    Stacks all context frames into the input channel dim (like
    SequenceConvEncoder) -- fast, but ties the trained weights to a fixed
    context_frames. See PooledSequenceParamRegressor for a variant that
    isn't.
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        transforms: list[str],
        hidden_dim: int = 64,
    ):
        super().__init__()
        if len(transforms) == 0:
            raise ValueError("ContextParamRegressor requires at least one transform.")

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.transforms = transforms

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
            nn.Linear(hidden_dim, len(transforms)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C, H, W] -> [B, n_params], each column in that param's
        own transform space (e.g. log10(Rayleigh), not Rayleigh itself)."""
        reshaped = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        f = self.trunk(reshaped)
        pooled = f.mean(dim=(-2, -1))
        return self.head(pooled)


class PooledSequenceParamRegressor(_ParamTransformMixin, nn.Module):
    """Same goal as ContextParamRegressor, but built to consume an entire
    trajectory (or any number of frames) instead of a fixed 6-frame window:
    a SHARED per-frame conv trunk (in_channels = the physical channel count
    only, not channels*T) runs on every frame independently, each frame gets
    spatially pooled to a feature vector, and those per-frame vectors are
    then mean-pooled over time before the regression head.

    Rayleigh/Prandtl are constant for an entire trajectory, so this is
    exactly the situation temporal averaging helps with: a single 6-frame
    window's estimate is one noisy sample, while pooling over dozens of
    frames should average that noise down. Also, unlike
    ContextParamRegressor's stack-as-channels design, nothing here is tied
    to a fixed sequence length -- train and infer on however many frames are
    available.
    """

    def __init__(
        self,
        in_channels: int,
        transforms: list[str],
        hidden_dim: int = 64,
    ):
        super().__init__()
        if len(transforms) == 0:
            raise ValueError("PooledSequenceParamRegressor requires at least one transform.")

        self.in_channels = in_channels
        self.transforms = transforms

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
            nn.Linear(hidden_dim, len(transforms)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C, H, W], T can be any length -> [B, n_params]."""
        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        f = self.per_frame_trunk(frames)          # [B*T, hidden, H', W']
        pooled_spatial = f.mean(dim=(-2, -1))      # [B*T, hidden]
        pooled_spatial = pooled_spatial.view(b, t, -1)
        pooled_time = pooled_spatial.mean(dim=1)   # [B, hidden]
        return self.head(pooled_time)
