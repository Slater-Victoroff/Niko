"""Training-health diagnostics for train_foundation.py: per-component gradient
norms, PCGrad inter-task conflict, and latent-state rollout drift -- signals
this project has repeatedly needed and reasoned about from indirect evidence
(final loss curves, ad-hoc checkpoint inspection scripts like
check_euler_bc.py) rather than measuring directly. See EXPERIMENT_LOG.md's
operator-term-ablation entry for the concrete motivating question this
answers: is a given run's instability/noise coming from genuine gradient
conflict or a specific component blowing up, or is it just run-to-run noise
indistinguishable from a stable run at the loss-curve level alone?

Every function here is read-only (no side effects on .grad/parameters) and
cheap relative to a forward/backward pass -- safe to call every logged step
(gated by --log-interval in train_foundation.py, same cadence as the existing
print-based logging) rather than needing its own separate sampling schedule.

Output is a structured JSONL file (one JSON object per logged step) alongside
the existing plain-text stdout log, written by log_diagnostics_step() -- the
stdout log stays exactly as it was (human skimming, tail -f-able); this file
is for loading into pandas/plotting after the fact, which free-text grep
can't give you.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

from core.states import LatentState


def _component_group(param_name: str) -> str:
    """Maps a FoundationModel parameter's full dotted name (from
    model.named_parameters()) to a readable diagnostic group -- one entry per
    operator term (so e.g. "is diffusion's gradient healthy" is answerable
    directly, not buried inside one aggregate "operator" number), one per
    task decoder, and one each for the other trunk submodules. Falls back to
    the top-level attribute name for anything not explicitly special-cased,
    so a newly-added submodule shows up under its own name automatically
    rather than silently vanishing into an "other" bucket."""
    parts = param_name.split(".")
    top = parts[0]
    if top == "operator" and len(parts) > 2 and parts[1] == "real_terms":
        return f"operator.{parts[2]}"
    if top == "operator" and len(parts) > 1 and parts[1] in ("complex_amplitude", "complex_rotation"):
        return f"operator.{parts[1]}"
    if top == "decoders" and len(parts) > 1:
        return f"decoder.{parts[1]}"
    if top == "field_embedder" and len(parts) > 2 and parts[1] == "embedders":
        return f"field_embedder.{parts[2]}"  # per canonical field, not per (field, task) -- keeps the group count sane
    if top == "boundary_geometry" and len(parts) > 2 and parts[1] == "heads":
        return f"boundary_geometry.{parts[2]}"
    return top


def grad_norms_by_component(model) -> Dict[str, float]:
    """L2 grad norm per diagnostic group (see _component_group), computed from
    whatever's currently in every parameter's .grad. Call this AFTER
    _set_grads_from_flat has populated .grad with the (PCGrad-combined, if
    enabled) update for this step, and BEFORE clip_grad_norm_ rescales
    everything -- this is the pre-clip breakdown clip_grad_norm_'s own
    returned total norm doesn't give you (that's the trunk-wide global number
    only; this is where in the model it's actually coming from)."""
    groups: Dict[str, List[Tensor]] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        groups.setdefault(_component_group(name), []).append(p.grad)
    return {
        group: float(torch.cat([g.detach().reshape(-1) for g in grads]).norm().item())
        for group, grads in groups.items()
    }


def pcgrad_conflict_stats(task_grad_vecs: List[Tensor], task_names: List[str]) -> Dict:
    """Pairwise cosine similarity between every pair of this step's per-task flat
    gradient vectors -- the direct measurement behind PCGrad's own premise
    (some task pairs pull the shared trunk in conflicting directions) rather
    than inferring conflict indirectly from "does turning PCGrad on/off change
    the loss curve." Computed on the SAME vectors pcgrad_combine consumes
    (raw per-task grads, pre-projection), so it's accurate regardless of
    whether --pcgrad is actually on for this run -- with pcgrad off, this
    still tells you how much conflict existed that pcgrad_combine isn't
    correcting for.

    Cheap: pairwise dot products on already-materialized flat vectors, no
    extra forward/backward -- safe to compute every logged step.
    """
    n = len(task_grad_vecs)
    if n < 2:
        return {"mean_cosine": None, "min_cosine": None, "conflict_frac": None, "pairs": {}}
    pairs: Dict[str, float] = {}
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = task_grad_vecs[i], task_grad_vecs[j]
            denom = gi.norm() * gj.norm() + 1e-12
            sim = float((torch.dot(gi, gj) / denom).item())
            sims.append(sim)
            pairs[f"{task_names[i]}~{task_names[j]}"] = sim
    return {
        "mean_cosine": sum(sims) / len(sims),
        "min_cosine": min(sims),
        "conflict_frac": sum(1 for s in sims if s < 0) / len(sims),
        "pairs": pairs,
    }


def rollout_drift_stats(zs: List[LatentState]) -> Dict:
    """Per-rollout-step normalized latent drift: ||z_t - z_{t-1}|| / ||z_{t-1}||
    (via .grid, the same real+complex fusion every decoder/loss in this
    codebase already reads), for one representative forward call. A direct
    look at whether the operator's per-step update is shrinking, roughly
    steady, or growing across a rollout -- the thing every long-horizon-
    blowup investigation in EXPERIMENT_LOG.md (§14/§17/§18/§20) has had to
    infer indirectly from decoded loss diverging many steps later, when the
    actual runaway growth in latent space is directly measurable at the
    point it happens instead.

    Requires the caller to have gotten `zs` back from FoundationModel.forward
    via return_latents=True (opt-in, see its docstring) -- not computed by
    default, since it needs the intermediate per-step states a normal forward
    call discards.
    """
    steps = [z.grid for z in zs]
    deltas = []
    for prev, cur in zip(steps[:-1], steps[1:]):
        num = (cur - prev).detach().float().norm()
        den = prev.detach().float().norm() + 1e-8
        deltas.append(float((num / den).item()))
    if not deltas:
        return {"per_step": [], "min": None, "mean": None, "max": None}
    return {"per_step": deltas, "min": min(deltas), "mean": sum(deltas) / len(deltas), "max": max(deltas)}


def boundary_geometry_snapshot(model, task_names: List[str]) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Current learned per-task, per-axis padding-mode blend (softmax over
    [zero, circular, replicate]) straight from BoundaryGeometryHead's own
    zero-inited Linear heads -- the standing version of the one-off
    check_euler_bc.py-style checkpoint inspection this project has done
    manually multiple times (see EXPERIMENT_LOG.md's per-task-BC-fix entry),
    now emitted automatically every epoch instead of needing a separate
    script run against a saved checkpoint after the fact. Returns None if the
    model has no boundary_geometry module (--no-boundary-geometry).

    Reads bias directly rather than running a forward pass: BoundaryGeometryHead's
    weight is zero-inited and stays near-zero early in training, so bias alone
    is a reasonable proxy for "what has this task's head learned so far,
    independent of a specific context batch" -- cheap (no data needed) and
    exactly matches the zero-context assumption the zero-init itself was
    designed around.
    """
    bg = getattr(model, "boundary_geometry", None)
    if bg is None:
        return None
    out = {}
    for task in task_names:
        if task not in bg.heads:
            continue
        bias = bg.heads[task].bias.detach().view(2, 3)
        weights = bias.softmax(dim=-1)  # [2, 3]: axis (x, y) x (zero, circular, replicate)
        out[task] = {
            "x": weights[0].tolist(),
            "y": weights[1].tolist(),
        }
    return out


def log_diagnostics_step(path: Path, record: dict) -> None:
    """Appends one JSON record as a line to `path` -- plain JSONL, no
    dependency beyond the stdlib, loadable with pandas.read_json(path,
    lines=True) or one line of jsonlines-per-line parsing. Append mode so a
    requeued/resumed run's diagnostics accumulate rather than clobbering a
    prior attempt's file, same spirit as the existing stdout log's behavior."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
