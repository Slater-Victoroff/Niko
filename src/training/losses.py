import torch
from torch import Tensor


def mse_rollout_loss(pred: Tensor, target: Tensor) -> Tensor:
    err = (pred - target) ** 2
    return err.mean()


def divergence_loss(velocity: Tensor) -> Tensor:
    """
    velocity: [B, 2, H, W]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]

    dudx = 0.5 * (torch.roll(u, -1, -1) - torch.roll(u, 1, -1))
    dvdy = 0.5 * (torch.roll(v, -1, -2) - torch.roll(v, 1, -2))

    div = dudx + dvdy
    return (div ** 2).mean()


def vnmse_rollout_loss(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Variance-normalized MSE.

    pred/target: [B, K, C, H, W]
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    if pred.ndim == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    if pred.ndim != 5:
        raise ValueError(f"Expected [B, K, C, H, W], got {tuple(pred.shape)}")

    err = (pred - target).float().pow(2)

    var_ch = target.float().var(dim=(0, 1, 3, 4), unbiased=False)
    err = err / torch.clamp_min(var_ch[None, None, :, None, None], eps)

    return err.mean()


def well_style_vrmse(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """
    WELL-style VRMSE for [B, K, C, H, W].

    Returns [B, K, C], because WELL reduces spatial dims only.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    if pred.ndim != 5:
        raise ValueError(f"Expected [B, K, C, H, W], got {tuple(pred.shape)}")

    spatial_dims = (-2, -1)

    mse = (pred - target).float().pow(2).mean(dim=spatial_dims)
    var = target.float().std(dim=spatial_dims, unbiased=False).pow(2)

    return torch.sqrt(mse / (var + eps))
