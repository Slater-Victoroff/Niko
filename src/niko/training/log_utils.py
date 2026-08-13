"""
Lightweight, opt-in logging utilities for the training loop (gradient norms,
etc.). Mirrors the debug_timing pattern: off by default, cheap to leave wired
in, and only does work when explicitly requested.
"""

import torch
from typing import Dict


@torch.no_grad()
def compute_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    """
    L2 gradient norm grouped by top-level submodule (e.g. param_encoder /
    encoder / operator / decoder for LatentDynamicsModel), plus an overall
    'total'.

    Call after loss.backward() -- and before grad clipping / optimizer.step()
    if you want the model's natural per-component gradient scale rather than
    the post-clip one. Parameters with no .grad (e.g. frozen) are skipped.
    """
    sq_sums: Dict[str, float] = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_sq = float(p.grad.detach().float().pow(2).sum())
        component = name.split(".", 1)[0]
        sq_sums[component] = sq_sums.get(component, 0.0) + g_sq
        total_sq += g_sq

    norms = {component: total**0.5 for component, total in sq_sums.items()}
    norms["total"] = total_sq**0.5
    return norms


def format_grad_norms(norms: Dict[str, float], step: int, prefix: str = "GRAD_NORM") -> str:
    component_order = [k for k in norms if k != "total"]
    parts = "  ".join(f"{k}={norms[k]:.6f}" for k in component_order)
    return f"{prefix}  step={step}  {parts}  total={norms.get('total', 0.0):.6f}"


def print_gns(model: torch.nn.Module, step: int, prefix: str = "GRAD_NORM") -> None:
    """Print component-wise gradient norms (encoder/operator/decoder/...). Call after loss.backward()."""
    print(format_grad_norms(compute_grad_norms(model), step=step, prefix=prefix))
