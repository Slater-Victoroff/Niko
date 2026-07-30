"""
Physics diagnostics for Rayleigh-Bénard rollout quality.

Fields: [B, K, C, H, W] with C=4: [pressure, buoyancy, vel_x, vel_y].
Grid spacing dx = dy = 1 (normalized). No gradients are created or modified.

PDEs (dimensionless Boussinesq, residual = LHS of each equation):
  Buoyancy:   ∂t b  +  u ∂x b  +  v ∂y b  −  κ Δb             = 0
  Momentum u: ∂t u  +  u ∂x u  +  v ∂y u  +  ∂x p  −  ν Δu    = 0
  Momentum v: ∂t v  +  u ∂x v  +  v ∂y v  +  ∂y p  −  ν Δv  −  g·b = 0
  Continuity: ∂x u  +  ∂y v                                     = 0

All spatial derivatives are second-order central differences with periodic BCs.
Time derivatives use a first-order forward difference: ∂t f ≈ f[t+1] − f[t].
Neither the simulator nor the model will have zero PDE residual under this FD
approximation; always compare pred vs target ratios, not absolute values.

Ratio convention
----------------
  ratio < 1  →  model has LESS of this quantity than ground truth (diffusion, smoothing)
  ratio ≈ 1  →  model matches ground-truth statistics
  ratio > 1  →  model has MORE of this quantity (noise, constraint violation)
"""

import torch
from torch import Tensor
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Finite-difference operators (inline — keeps this module self-contained)
# Convention: dim=-1 is x (columns), dim=-2 is y (rows)
# ---------------------------------------------------------------------------

def _dx(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, -1, -1) - torch.roll(x, 1, -1))


def _dy(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, -1, -2) - torch.roll(x, 1, -2))


def _lap(x: Tensor) -> Tensor:
    return (
        torch.roll(x, 1, -1) + torch.roll(x, -1, -1)
        + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        - 4.0 * x
    )


# ---------------------------------------------------------------------------
# Scalar statistics and storage helpers
# ---------------------------------------------------------------------------

def _rms(x: Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def _mae(x: Tensor) -> float:
    return float(x.float().abs().mean())


def _ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return a / (abs(b) + eps)


def _rec(m: Dict, key: str, pred_val: float, tgt_val: float) -> None:
    """Store pred / target / ratio triple into m."""
    m[f"{key}_pred"] = pred_val
    m[f"{key}_target"] = tgt_val
    m[f"{key}_ratio"] = _ratio(pred_val, tgt_val)


# ---------------------------------------------------------------------------
# Diagnostic groups
# Each function takes pre-flattened frames and returns a partial metrics dict.
# ---------------------------------------------------------------------------

def _incompressibility(pf: Tensor, tf: Tensor) -> Dict[str, float]:
    """Continuity: ∂x u + ∂y v."""
    div_p = _dx(pf[:, 2]) + _dy(pf[:, 3])
    div_t = _dx(tf[:, 2]) + _dy(tf[:, 3])
    m: Dict[str, float] = {}
    _rec(m, "incomp_div_rms", _rms(div_p), _rms(div_t))
    _rec(m, "incomp_div_mae", _mae(div_p), _mae(div_t))
    return m


def _buoyancy_terms(
    pf:  Tensor,            # [N, 4, H, W]  all pred frames
    tf:  Tensor,            # [N, 4, H, W]  all target frames
    pp:  Optional[Tensor],  # [M, 4, H, W]  pred at time t in each consecutive pair
    pp1: Optional[Tensor],  # [M, 4, H, W]  pred at time t+1
    tp:  Optional[Tensor],  # [M, 4, H, W]  target at time t
    tp1: Optional[Tensor],  # [M, 4, H, W]  target at time t+1
    kappa: float,
) -> Dict[str, float]:
    """
    Buoyancy transport: ∂t b + u ∂x b + v ∂y b − κ Δb.

    Per-frame spatial terms are averaged over all N frames.
    ∂t b and the full residual require consecutive pairs (K >= 2).
    """
    m: Dict[str, float] = {}

    b_p, u_p, v_p = pf[:, 1], pf[:, 2], pf[:, 3]
    b_t, u_t, v_t = tf[:, 1], tf[:, 2], tf[:, 3]

    _rec(m, "buoy_u_db_dx",    _rms(u_p * _dx(b_p)),      _rms(u_t * _dx(b_t)))
    _rec(m, "buoy_v_db_dy",    _rms(v_p * _dy(b_p)),      _rms(v_t * _dy(b_t)))
    _rec(m, "buoy_kappa_lap_b", _rms(kappa * _lap(b_p)),   _rms(kappa * _lap(b_t)))

    if pp is not None:
        b0_p, b1_p = pp[:, 1], pp1[:, 1]
        b0_t, b1_t = tp[:, 1], tp1[:, 1]
        u0_p, v0_p = pp[:, 2], pp[:, 3]
        u0_t, v0_t = tp[:, 2], tp[:, 3]

        db_dt_p = b1_p - b0_p
        db_dt_t = b1_t - b0_t
        _rec(m, "buoy_db_dt", _rms(db_dt_p), _rms(db_dt_t))

        res_p = db_dt_p + u0_p * _dx(b0_p) + v0_p * _dy(b0_p) - kappa * _lap(b0_p)
        res_t = db_dt_t + u0_t * _dx(b0_t) + v0_t * _dy(b0_t) - kappa * _lap(b0_t)
        _rec(m, "buoy_residual", _rms(res_p), _rms(res_t))

    return m


def _momentum_terms(
    pf:  Tensor,
    tf:  Tensor,
    pp:  Optional[Tensor],
    pp1: Optional[Tensor],
    tp:  Optional[Tensor],
    tp1: Optional[Tensor],
    nu: float,
    g:  float,
) -> Dict[str, float]:
    """
    NS momentum:
      u: ∂t u + u ∂x u + v ∂y u + ∂x p − ν Δu  = 0
      v: ∂t v + u ∂x v + v ∂y v + ∂y p − ν Δv − g·b = 0

    forcing_x = 0  (identically, RBC has no horizontal body force)
    forcing_y = g·b  (buoyancy drives vertical momentum)
    """
    m: Dict[str, float] = {}

    p_p, b_p, u_p, v_p = pf[:, 0], pf[:, 1], pf[:, 2], pf[:, 3]
    p_t, b_t, u_t, v_t = tf[:, 0], tf[:, 1], tf[:, 2], tf[:, 3]

    # --- momentum u: per-frame spatial terms ---
    _rec(m, "mom_u_u_du_dx",  _rms(u_p * _dx(u_p)),   _rms(u_t * _dx(u_t)))
    _rec(m, "mom_u_v_du_dy",  _rms(v_p * _dy(u_p)),   _rms(v_t * _dy(u_t)))
    _rec(m, "mom_u_dp_dx",    _rms(_dx(p_p)),          _rms(_dx(p_t)))
    _rec(m, "mom_u_nu_lap_u", _rms(nu * _lap(u_p)),    _rms(nu * _lap(u_t)))
    # forcing_x ≡ 0 by problem definition — log it explicitly but ratio is trivially 1
    m["mom_u_forcing_x_pred"]   = 0.0
    m["mom_u_forcing_x_target"] = 0.0
    m["mom_u_forcing_x_ratio"]  = 1.0

    # --- momentum v: per-frame spatial terms ---
    _rec(m, "mom_v_u_dv_dx",  _rms(u_p * _dx(v_p)),   _rms(u_t * _dx(v_t)))
    _rec(m, "mom_v_v_dv_dy",  _rms(v_p * _dy(v_p)),   _rms(v_t * _dy(v_t)))
    _rec(m, "mom_v_dp_dy",    _rms(_dy(p_p)),          _rms(_dy(p_t)))
    _rec(m, "mom_v_nu_lap_v", _rms(nu * _lap(v_p)),    _rms(nu * _lap(v_t)))
    _rec(m, "mom_v_forcing_y", _rms(g * b_p),          _rms(g * b_t))

    # --- time derivatives and residuals (require pairs) ---
    if pp is not None:
        p0_p, b0_p, u0_p, v0_p = pp[:, 0],  pp[:, 1],  pp[:, 2],  pp[:, 3]
        p0_t, b0_t, u0_t, v0_t = tp[:, 0],  tp[:, 1],  tp[:, 2],  tp[:, 3]
        u1_p, v1_p = pp1[:, 2], pp1[:, 3]
        u1_t, v1_t = tp1[:, 2], tp1[:, 3]

        du_dt_p, du_dt_t = u1_p - u0_p, u1_t - u0_t
        dv_dt_p, dv_dt_t = v1_p - v0_p, v1_t - v0_t
        _rec(m, "mom_u_du_dt", _rms(du_dt_p), _rms(du_dt_t))
        _rec(m, "mom_v_dv_dt", _rms(dv_dt_p), _rms(dv_dt_t))

        res_u_p = (du_dt_p + u0_p * _dx(u0_p) + v0_p * _dy(u0_p)
                   + _dx(p0_p) - nu * _lap(u0_p))
        res_u_t = (du_dt_t + u0_t * _dx(u0_t) + v0_t * _dy(u0_t)
                   + _dx(p0_t) - nu * _lap(u0_t))
        _rec(m, "mom_u_residual", _rms(res_u_p), _rms(res_u_t))

        res_v_p = (dv_dt_p + u0_p * _dx(v0_p) + v0_p * _dy(v0_p)
                   + _dy(p0_p) - nu * _lap(v0_p) - g * b0_p)
        res_v_t = (dv_dt_t + u0_t * _dx(v0_t) + v0_t * _dy(v0_t)
                   + _dy(p0_t) - nu * _lap(v0_t) - g * b0_t)
        _rec(m, "mom_v_residual", _rms(res_v_p), _rms(res_v_t))

    return m


def _spectral_group(
    pf: Tensor, tf: Tensor, hf_cutoff_frac: float
) -> Dict[str, float]:
    """FFT power spectrum stats for buoyancy and velocity_x."""
    m: Dict[str, float] = {}
    for name, fp, ft in [("b", pf[:, 1], tf[:, 1]), ("u", pf[:, 2], tf[:, 2])]:
        sp = _spectral_stats_2d(fp, hf_cutoff_frac)
        st = _spectral_stats_2d(ft, hf_cutoff_frac)
        _rec(m, f"spec_{name}_total_energy", sp["total_energy"], st["total_energy"])
        _rec(m, f"spec_{name}_hf_frac",      sp["hf_frac"],      st["hf_frac"])
    return m


def _spectral_stats_2d(field: Tensor, hf_cutoff_frac: float) -> Dict[str, float]:
    """Power spectrum of a [N, H, W] batch: total energy and HF fraction."""
    spec  = torch.fft.rfft2(field.float())
    power = spec.real.pow(2) + spec.imag.pow(2)   # [N, H, W//2+1]

    H, Wh = field.shape[-2], spec.shape[-1]
    W     = (Wh - 1) * 2
    k_nyq = min(H // 2, W // 2)

    ky = torch.arange(H,  device=field.device, dtype=torch.float32)
    kx = torch.arange(Wh, device=field.device, dtype=torch.float32)
    ky_signed = torch.where(ky <= H // 2, ky, ky - H)
    k_mag = (ky_signed[:, None].pow(2) + kx[None, :].pow(2)).sqrt()  # [H, Wh]

    total = power.sum((-2, -1)).mean()
    hf    = (power * (k_mag > hf_cutoff_frac * k_nyq)).sum((-2, -1)).mean()

    return {"total_energy": float(total), "hf_frac": float(hf / (total + 1e-8))}


def _vorticity_group(pf: Tensor, tf: Tensor) -> Dict[str, float]:
    """Vorticity ω = ∂x v − ∂y u."""
    m: Dict[str, float] = {}
    omg_p = _dx(pf[:, 3]) - _dy(pf[:, 2])
    omg_t = _dx(tf[:, 3]) - _dy(tf[:, 2])
    _rec(m, "vort_rms", _rms(omg_p), _rms(omg_t))
    return m


# ---------------------------------------------------------------------------
# Display layout (order and section headers for format_diagnostics)
# ---------------------------------------------------------------------------

_DISPLAY_SECTIONS = [
    (
        "Incompressibility  ∂x u + ∂y v = 0",
        ["incomp_div_rms", "incomp_div_mae"],
    ),
    (
        "Buoyancy Transport  ∂t b + u ∂x b + v ∂y b − κ Δb",
        ["buoy_db_dt", "buoy_u_db_dx", "buoy_v_db_dy", "buoy_kappa_lap_b", "buoy_residual"],
    ),
    (
        "Momentum (u)  ∂t u + u ∂x u + v ∂y u + ∂x p − ν Δu",
        ["mom_u_du_dt", "mom_u_u_du_dx", "mom_u_v_du_dy",
         "mom_u_dp_dx", "mom_u_nu_lap_u", "mom_u_forcing_x", "mom_u_residual"],
    ),
    (
        "Momentum (v)  ∂t v + u ∂x v + v ∂y v + ∂y p − ν Δv − g b",
        ["mom_v_dv_dt", "mom_v_u_dv_dx", "mom_v_v_dv_dy",
         "mom_v_dp_dy", "mom_v_nu_lap_v", "mom_v_forcing_y", "mom_v_residual"],
    ),
    (
        "Spectral  (buoyancy b)",
        ["spec_b_total_energy", "spec_b_hf_frac"],
    ),
    (
        "Spectral  (velocity u)",
        ["spec_u_total_energy", "spec_u_hf_frac"],
    ),
    (
        "Vorticity  ω = ∂x v − ∂y u",
        ["vort_rms"],
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_physics_diagnostics(
    pred:   Tensor,
    target: Tensor,
    kappa:  float = 0.1,
    nu:     Optional[float] = None,
    g:      float = 1.0,
    hf_cutoff_frac: float = 0.5,
) -> Dict[str, float]:
    """Compute physics diagnostics on decoded physical fields.

    No gradients are created or modified.  pred/target are detached internally.

    Args:
        pred:   Model rollout [B, K, C, H, W], C=[pressure, buoyancy, vel_x, vel_y].
        target: Ground-truth rollout, same shape.
        kappa:  Buoyancy diffusivity κ = 1/Pr  (default Pr=10 → κ=0.1).
        nu:     Kinematic viscosity ν  (default: same as kappa).
        g:      Gravity coefficient in buoyancy forcing term of v-momentum.
        hf_cutoff_frac: High-frequency threshold as fraction of Nyquist (0–1).

    Returns:
        Flat dict.  Each physical quantity has _pred, _target, _ratio entries.
    """
    if nu is None:
        nu = kappa

    pred   = pred.detach().float()
    target = target.detach().float()

    B, K, C, H, W = pred.shape
    if C != 4:
        raise ValueError(f"Expected C=4 [p,b,u,v], got C={C}")

    N  = B * K
    pf = pred.reshape(N, C, H, W)
    tf = target.reshape(N, C, H, W)

    # Consecutive-pair slices for time-derivative terms (None when K < 2)
    pp = pp1 = tp = tp1 = None
    if K >= 2:
        M   = B * (K - 1)
        pp  = pred[:, :-1].reshape(M, C, H, W)
        pp1 = pred[:, 1:].reshape(M, C, H, W)
        tp  = target[:, :-1].reshape(M, C, H, W)
        tp1 = target[:, 1:].reshape(M, C, H, W)

    m: Dict[str, float] = {}
    m.update(_incompressibility(pf, tf))
    m.update(_buoyancy_terms(pf, tf, pp, pp1, tp, tp1, kappa))
    m.update(_momentum_terms(pf, tf, pp, pp1, tp, tp1, nu, g))
    m.update(_spectral_group(pf, tf, hf_cutoff_frac))
    m.update(_vorticity_group(pf, tf))
    return m


def format_diagnostics(
    metrics: Dict[str, float],
    step:    int,
    prefix:  str = "DIAG",
) -> str:
    """Format diagnostics as a sectioned table (pred / target / ratio per term).

    Terms requiring K >= 2 (time derivatives, residuals) are silently skipped
    when not present in metrics.

    ratio < 1  →  model has LESS than ground truth (diffusion, smoothing)
    ratio > 1  →  model has MORE than ground truth (noise, constraint violation)
    """
    COL = 22   # metric name column width

    def fmt(key: str) -> str:
        v = metrics.get(key)
        if v is None:
            return "         n/a"
        if abs(v) < 1e-2 or abs(v) >= 1e4:
            return f"{v:>12.3e}"
        return f"{v:>12.6f}"

    lines = [f"{prefix}  step={step}"]

    for section_title, keys in _DISPLAY_SECTIONS:
        # Skip sections where no keys are present (e.g. pair terms when K<2)
        present = [k for k in keys if f"{k}_pred" in metrics]
        if not present:
            continue

        lines.append(f"\n  ── {section_title}")
        lines.append(f"  {'metric':<{COL}}  {'pred':>12}  {'target':>12}  {'ratio':>10}")
        lines.append("  " + "─" * (COL + 40))

        for key in present:
            ratio_v = metrics.get(f"{key}_ratio")
            ratio_s = f"{ratio_v:>10.4f}" if ratio_v is not None else "       n/a"
            lines.append(
                f"  {key:<{COL}}"
                f"  {fmt(key + '_pred')}"
                f"  {fmt(key + '_target')}"
                f"  {ratio_s}"
            )

    return "\n".join(lines)
