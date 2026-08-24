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


def well_style_nrmse(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """The Well's own published NRMSE (Eq. 6): RMSE / (target RMS + eps), per
    channel -- same [B, K, C, H, W] -> [B, K, C] contract as well_style_vrmse
    above, so it's a drop-in replacement anywhere that's used (train_foundation.py's
    --loss-fn flag selects between them for both the training objective and
    validation).

    2026-08-23: built after well_style_vrmse's std-normalization was found (via
    eval/eval_foundation_nrmse.py, then confirmed against a real worst-case
    gray_scott validation sample -- see EXPERIMENT_LOG.md) to blow up ~100x+ on
    windows where the target has genuinely converged to a spatially-constant
    state (std -> 0) even though the model's prediction was within a few percent
    in absolute terms -- a metric-normalization artifact on physically-boring
    states, not a real prediction failure. RMS (this function's denominator)
    captures the target's MAGNITUDE, not its spread, so a converged-but-nonzero
    field (e.g. gray_scott's A channel saturating to ~1.0) still gets a normal,
    well-behaved denominator here even though std-based normalization breaks
    down for it. Confirmed via that same worst-case sample: vrmse 35.8 -> this
    metric 4.8.

    NOT immune to the same class of problem when the target's magnitude ALSO
    goes to zero (not just its spread) -- gray_scott's other channel (B) can
    decay toward genuinely near-zero magnitude in the same converged windows,
    which is why eps appears twice here (once inside the sqrt, once in the
    final division) rather than once: a single un-square-rooted eps=1e-7 floor
    (what nrmse_range/nrmse_mean in eval_foundation_nrmse.py use) was found to
    be too thin for that case specifically (blew up to ~15000 on the same
    worst-case sample, worse than plain vrmse) -- the double, partially-
    square-rooted floor here degrades far more gracefully.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    if pred.ndim != 5:
        raise ValueError(f"Expected [B, K, C, H, W], got {tuple(pred.shape)}")

    spatial_dims = (-2, -1)

    mse = (pred - target).float().pow(2).mean(dim=spatial_dims)
    rmse = mse.sqrt()
    target_rms = (target.float().pow(2).mean(dim=spatial_dims) + eps).sqrt()

    return rmse / (target_rms + eps)


def compute_channel_scale(loader, n_batches: int = 50, eps: float = 1e-7) -> Tensor:
    """Pools RAW target values across n_batches real batches into one stable
    per-channel RMS reference -- used by well_style_nrmse_fixed instead of
    well_style_nrmse's per-sample-per-batch dynamic normalization.

    Deliberately pools every sampled batch's values into ONE running
    sum-of-squares (not "average each batch's own RMS together") -- averaging
    per-batch RMS estimates would still let one unusually-quiet batch pull the
    average down; pooling raw sums-of-squares means the estimate is dominated
    by the SAME total volume of data regardless of how any individual batch
    happens to look, exactly the stability property a per-batch estimate
    lacks. n_batches=50 (default) is enough to average out ordinary batch-to-
    batch variation without scanning a whole epoch -- see EXPERIMENT_LOG.md
    for the well_style_nrmse investigation this was built to fix.

    Returns a [C] tensor (one scale per channel), meant to be computed once
    per task at training start and reused for every batch/epoch after --
    NOT recomputed per batch, which is the entire point.
    """
    sq_sum = None
    n_elements = None
    seen = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            _, yb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            _, yb, _ = next(it)
        # yb: [B, K, C, H, W] -- pool over every dim except C.
        y = yb.float()
        c = y.shape[2]
        y_c = y.movedim(2, 0).reshape(c, -1)  # [C, B*K*H*W]
        batch_sq_sum = y_c.pow(2).sum(dim=1)
        if sq_sum is None:
            sq_sum = batch_sq_sum
            n_elements = torch.full((c,), y_c.shape[1], dtype=torch.float64)
        else:
            sq_sum += batch_sq_sum
            n_elements += y_c.shape[1]
        seen += 1
    return (sq_sum.double() / n_elements + eps).sqrt().float()


def well_style_nrmse_fixed(pred: Tensor, target: Tensor, channel_scale: Tensor, eps: float = 1e-7) -> Tensor:
    """Like well_style_nrmse, but normalizes by a FIXED, precomputed per-channel
    RMS (see compute_channel_scale) instead of recomputing the reference fresh
    from each batch's own target.

    2026-08-24: built after tracing a real, chronic (not rare-outlier) problem
    with well_style_nrmse's per-batch normalization -- traced directly via
    per-decoder-head gradient norms on shear_flow (see EXPERIMENT_LOG.md):
    shear_flow's pressure channel (RMS~0.066) is ~5x smaller than its siblings
    (vx~0.344, tracer~0.325), and because the loss's normalization denominator
    is recomputed from each batch's OWN local statistics, the gradient
    contribution from pressure varied wildly batch to batch (15-300x its
    siblings', not a stable ratio) -- decoder.shear_flow's pre-clip gradient
    norm hit 1000+ on some batches (vs. a ~10-50 typical), forcing the SAME
    global clip to rescale the other two tasks' perfectly healthy gradients
    down by the same huge, unpredictable factor every time it happened
    (80%+ of logged steps in one real run).

    This does NOT remove the cross-channel weighting itself -- a channel that's
    naturally 5x smaller than its siblings still gets proportionally more
    gradient weight per unit of relative error, which is the whole legitimate
    reason a normalized loss exists in the first place (so heterogeneous-unit
    channels combine into one meaningful scalar). What changes is that the
    weighting is now a STABLE, predictable ratio (computed once, from a large
    pooled sample) instead of one that swings 15-300x depending on which
    specific window happened to be in this batch -- removing the noise/
    instability without silently changing what "good performance" means.

    Also inherits the same near-zero-target robustness well_style_nrmse has
    over well_style_vrmse (RMS-normalized, not std-normalized) -- and is
    actually MORE robust to it than well_style_nrmse, since a single
    degenerate (near-constant or fully-decayed) window can no longer drag its
    own denominator down at all; the reference is fixed regardless of what
    that specific window looks like.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 5:
        raise ValueError(f"Expected [B, K, C, H, W], got {tuple(pred.shape)}")

    spatial_dims = (-2, -1)
    mse = (pred - target).float().pow(2).mean(dim=spatial_dims)  # [B, K, C]
    rmse = mse.sqrt()
    scale = channel_scale.to(rmse.device).view(1, 1, -1)
    return rmse / (scale + eps)


def _phys_dx(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, -1, -1) - torch.roll(x, 1, -1))


def _phys_dy(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, -1, -2) - torch.roll(x, 1, -2))


def _phys_lap(x: Tensor) -> Tensor:
    return (
        torch.roll(x, 1, -1) + torch.roll(x, -1, -1)
        + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        - 4.0 * x
    )


def _var_normed_mse(pred_field: Tensor, target_field: Tensor, eps: float) -> Tensor:
    """mean((pred - target)^2) / var(target) -- the per-quantity vRMSE building block."""
    err = (pred_field - target_field).pow(2).mean()
    var = target_field.var(unbiased=False)
    return err / torch.clamp_min(var, eps)


def physics_consistency_loss(
    pred: Tensor,
    target: Tensor,
    kappa: float = 0.1,
    nu: float | None = None,
    g: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 5:
        raise ValueError(f"Expected [B, K, C, H, W], got {tuple(pred.shape)}")

    B, K, C, H, W = pred.shape
    if C != 4:
        raise ValueError(f"Expected C=4 [p, b, u, v], got C={C}")

    if nu is None:
        nu = kappa

    pred_f = pred.float()
    target_f = target.float()

    pf = pred_f.reshape(B * K, C, H, W)
    tf = target_f.reshape(B * K, C, H, W)
    u_p, v_p = pf[:, 2], pf[:, 3]
    u_t, v_t = tf[:, 2], tf[:, 3]

    div_p = _phys_dx(u_p) + _phys_dy(v_p)
    div_t = _phys_dx(u_t) + _phys_dy(v_t)
    terms = [_var_normed_mse(div_p, div_t, eps)]

    if K >= 2:
        M = B * (K - 1)
        pp,  pp1 = pred_f[:, :-1].reshape(M, C, H, W), pred_f[:, 1:].reshape(M, C, H, W)
        tp,  tp1 = target_f[:, :-1].reshape(M, C, H, W), target_f[:, 1:].reshape(M, C, H, W)

        p0_p, b0_p, u0_p, v0_p = pp[:, 0],  pp[:, 1],  pp[:, 2],  pp[:, 3]
        p0_t, b0_t, u0_t, v0_t = tp[:, 0],  tp[:, 1],  tp[:, 2],  tp[:, 3]
        b1_p, u1_p, v1_p       = pp1[:, 1], pp1[:, 2], pp1[:, 3]
        b1_t, u1_t, v1_t       = tp1[:, 1], tp1[:, 2], tp1[:, 3]

        res_b_p = (b1_p - b0_p) + u0_p * _phys_dx(b0_p) + v0_p * _phys_dy(b0_p) - kappa * _phys_lap(b0_p)
        res_b_t = (b1_t - b0_t) + u0_t * _phys_dx(b0_t) + v0_t * _phys_dy(b0_t) - kappa * _phys_lap(b0_t)
        terms.append(_var_normed_mse(res_b_p, res_b_t, eps))

        res_u_p = (u1_p - u0_p) + u0_p * _phys_dx(u0_p) + v0_p * _phys_dy(u0_p) + _phys_dx(p0_p) - nu * _phys_lap(u0_p)
        res_u_t = (u1_t - u0_t) + u0_t * _phys_dx(u0_t) + v0_t * _phys_dy(u0_t) + _phys_dx(p0_t) - nu * _phys_lap(u0_t)
        terms.append(_var_normed_mse(res_u_p, res_u_t, eps))

        res_v_p = (v1_p - v0_p) + u0_p * _phys_dx(v0_p) + v0_p * _phys_dy(v0_p) + _phys_dy(p0_p) - nu * _phys_lap(v0_p) - g * b0_p
        res_v_t = (v1_t - v0_t) + u0_t * _phys_dx(v0_t) + v0_t * _phys_dy(v0_t) + _phys_dy(p0_t) - nu * _phys_lap(v0_t) - g * b0_t
        terms.append(_var_normed_mse(res_v_p, res_v_t, eps))

    return torch.stack(terms).mean()
