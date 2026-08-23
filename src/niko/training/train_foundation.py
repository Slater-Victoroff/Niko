"""Foundation-model test: one shared trunk (field_embedder + context_cond_encoder
+ encoder + operator) trained jointly across rayleigh_benard/shear_flow/
active_matter, with a separate decoder per task. Tests whether joint training
helps the shared trunk vs. each task's own from-scratch trunk (this session's
single-task baselines: plain_helmholtz_broad8 0.229, shear_flow_broad8 0.385,
active_matter 0.446-0.567 -- see EXPERIMENT_LOG.md).

The three tasks are at different native resolutions (rayleigh_benard
128x512, shear_flow 512x256, active_matter 256x256), so a single batch
tensor can't literally mix samples from different tasks. Instead, every
optimizer step pulls one batch from ALL THREE tasks (each task's own
shuffled loader), runs forward+backward for each without zeroing grad in
between -- so gradients from all three accumulate together into the shared
trunk every single step (each task's own decoder only ever gets gradient
from its own batch, since decoders are disjoint per task) -- then a single
opt.step() applies the combined update.

Earlier version of this script let a task drop out once ITS OWN loader was
exhausted for the epoch, so the last ~third of every epoch (after
active_matter's ~254 batches and rayleigh_benard's ~7560 ran out) was
shear_flow-only for ~4500 straight steps -- the shared trunk drifted toward
shear_flow's optimum with no rayleigh_benard/active_matter gradient at all
for that whole stretch, and validation (run right after) caught the drifted
trunk: rayleigh_benard/active_matter val loss was 30-60x worse than their
own single-task baselines despite normal training loss all epoch, while
shear_flow (never without gradient) came out fine. Fixed by having the two
shorter loaders restart (fresh shuffle) whenever exhausted mid-epoch, so all
three ALWAYS co-occur in every step -- shear_flow (the longest, ~12096
batches) is still seen exactly once per epoch with no repeats and defines
the epoch length; rayleigh_benard cycles ~1.6x and active_matter ~47x within
that same epoch to keep pace. This restarts via a fresh iter(loader) each
pass, not itertools.cycle (which caches every batch it's ever yielded --
the actual cause of this script's two earlier OOM crashes), so memory stays
bounded regardless of how many times a shorter loader repeats.

Conditioning is context-inferred (shared context_cond_encoder), not
ground-truth params: rayleigh_benard/shear_flow use 2 log10 params,
active_matter uses 3 (one signed) -- context-inferred cond sidesteps that
mismatch entirely, and this session's own probes found it matches
ground-truth-param conditioning quality on rayleigh_benard.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.field_embedder import FieldEmbedder, canonical_field_name
from encoders.context_cond import PooledContextCondEncoder, BoundaryGeometryHead
from encoders.sequence_conv import SequenceConvEncoder
from operators.transport_operator import TransportOperator
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder
from models.foundation_model import FoundationModel
from training.losses import well_style_vrmse
from core.soap import SOAP
import dataloader as dl


def _param_key(f: str) -> tuple:
    """Sort key for select_representative_files: a file's full simulation_parameters
    tuple, e.g. (Rayleigh, Prandtl) for rayleigh_benard_uniform, (zeta, alpha) for
    active_matter (L is constant, not actually varying). Tuples sort lexicographically
    (primary axis first, secondary within it), so evenly-spacing picks across the full
    sorted list naturally spreads across every axis, not just the first -- e.g.
    active_matter's zeta spans {1..17} and alpha spans {-1..-5}; using the full tuple
    picks 8 of 9 zeta values AND cycles through 4 of 5 alpha values, vs. a
    primary-only/zeta-alone key which would keep landing on the same alpha every time."""
    params = dl.read_params_from_h5(f)
    if not params:
        raise ValueError(f"No simulation_parameters found in {f}")
    return tuple(params)


def _planetswe_ic_key(f: str) -> int:
    """planetswe has no simulation_parameters (empty in every file) -- each file IS one
    very long single trajectory (N=1) for one initial condition, 3 filename-suffixed
    seeds per IC (e.g. planetswe_IC00_s1.hdf5). Parse the IC index straight from the
    filename instead."""
    m = re.search(r"IC(\d+)", os.path.basename(f))
    if not m:
        raise ValueError(f"Could not parse IC index from {f}")
    return int(m.group(1))


def _traj_cache_capacity(frames: int, channels: int, h: int, w: int, budget_mb: float = 512) -> int:
    """Per-task trajectory-cache size (see create_param_dataloaders' traj_cache_capacity)
    picked to hold roughly `budget_mb` of cached trajectories for THIS task's own shape,
    rather than reusing one constant across every task regardless of size. Necessary once
    the task list grows past the original 3 (see TASKS_ALL14 below): per-trajectory size
    varies by ~20x across the 14 2D Well datasets (turbulent_radiative_layer_2D's cached
    trajectory is ~79MB; planetswe's is ~1.6GB, at full resolution/length) -- the original
    single global TRAJ_CACHE_CAPACITY=8 sized for the original 3 tasks would, applied
    uniformly to all 14, reintroduce the exact same class of host-RAM OOM this constant
    was originally added to prevent (EXPERIMENT_LOG.md), just via a different task.
    """
    bytes_per_traj = max(1, frames * channels * h * w * 4)
    return max(2, int(budget_mb * 1024 * 1024 // bytes_per_traj))


def build_tasks(data_root: str, config_root: str, task_set: str = "core3") -> dict:
    """TASKS is a function of (data_root, config_root, task_set) rather than a module-level
    constant so this script can run outside the Docker container (where these are
    always /app/data/datasets and /app/configs) -- e.g. on a bare-metal cluster with
    a different dataset mount point -- and so the original 3-task recipe stays available
    unchanged (task_set="core3", the default) alongside the full 14-task set
    (task_set="all14"). Defaults below reproduce the exact original hardcoded paths, so
    every existing invocation is unaffected.

    task-by-task notes for the all14 additions (field_spec definitions and the physical
    reasoning behind each use_streamfunction/zero_mean_pressure choice live in
    dataloader.py next to the field specs themselves):

    - train_file_limit/val_file_limit/traj_limit/pair_stride are chosen per task to keep
      every task's natural per-epoch batch count in the same rough order of magnitude as
      the original 3 (tens to low thousands) -- NOT proportional to how much data actually
      exists on disk. Without this, euler_multi_quadrants alone (10 files x 400
      trajectories x 90 context/rollout windows each) would produce ~1.6M training pairs
      from a single dataset, vs. shear_flow's already-largest 12096 -- exactly the
      "some datasets are too large" problem being deliberately avoided here, not an
      incidental default.
    - "file" vs "trajectory" meaning differs by dataset: for param-swept datasets
      (euler_multi_quadrants: 1 file = 1 gamma value with 400 realizations;
      gray_scott: 1 file = 1 (F,k) combo with 160 realizations) file diversity is
      physically meaningful and kept generous, while traj_limit does the actual volume
      cutting. acoustic_scattering's files are unlabeled homogeneous chunks
      (simulation_parameters=[]) so there's no physical distinction between limiting
      files vs. trajectories there.
    - gray_scott (T=1001, N=1 traj/file -- one file IS one very long trajectory) uses
      pair_stride instead of/alongside traj_limit, since a single trajectory alone
      yields ~990 context/rollout windows -- file_limit and traj_limit can't touch
      that axis at all.
    - traj_cache_capacity is computed per task via _traj_cache_capacity() rather than
      reusing one global constant -- see that function's docstring.
    - 2026-08-17: planetswe and euler_multi_quadrants_periodicBC excluded from all14
      (both still fully supported for a future task_set that wants them back --
      _planetswe_ic_key and the field specs are untouched, only their all14 entries
      are removed here). Two independent, unresolved concerns raised on the all14 run
      that finished 2026-08-16, not confirmed root causes:
        - planetswe is the only lat-lon-sphere dataset (dimensions/theta,phi, not x,y)
          in the whole set, but nothing in field_embedder/encoder/operator treats it as
          anything other than another Cartesian raster -- no pole handling, no
          latitude-dependent cell-size correction, ordinary zero-padded conv2d exactly
          like every x/y task. Suspected but unverified source of trouble specifically
          for the shared trunk (which planetswe's polar grid feeds into like everyone
          else's Cartesian one).
        - euler_multi_quadrants_openBC and _periodicBC share one field_spec and the
          same shared trunk with zero explicit boundary-condition signal anywhere (no
          BC-type channel/scalar in field_spec or cond) -- the trunk can only infer
          which regime it's in from raw pixel context, if at all. openBC showed a 2x
          train/valid gap (train 2.12 / valid 4.22) vs. periodicBC's much tighter gap
          (train 1.11 / valid 1.31) on identical data volume (810 train / 540 val
          batches each) in that run -- dropping periodicBC here on the theory that
          having both variants share one undifferentiated trunk may be actively
          confusing it, though openBC (the visibly worse performer) is the one being
          kept, since periodicBC's own generalization looked fine on its own.
    """
    core3 = {
        "rayleigh_benard": dict(
            field_spec=dl.RB_FIELD_SPEC,
            data_dir=f"{data_root}/rayleigh_benard/data",
            param_subset=f"{config_root}/param_subsets/broad8.json",
            batch=8,
            traj_cache_capacity=8,  # literal original TRAJ_CACHE_CAPACITY value -- unchanged from before
            decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True),
        ),
        "shear_flow": dict(
            field_spec=dl.SHEAR_FLOW_FIELD_SPEC,
            data_dir=f"{data_root}/shear_flow/data",
            param_subset=f"{config_root}/param_subsets/shear_flow_broad8.json",
            # shear_flow's 512x256 grid has ~2x rayleigh_benard's pixel count --
            # EXPERIMENT_LOG hit a real OOM at batch=8 for this task specifically.
            batch=4,
            traj_cache_capacity=8,  # literal original TRAJ_CACHE_CAPACITY value -- unchanged from before
            decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True,
                                 scalar_field_names=["pressure", "tracer"]),
        ),
        "active_matter": dict(
            field_spec=dl.ACTIVE_MATTER_FIELD_SPEC,
            data_dir=f"{data_root}/active_matter/data",
            param_subset=None,
            train_file_limit=8,  # no existing named param_subset for this task; match broad8's size
            val_file_limit=8,  # unset originally -> pulled all 16 val files uncapped, a real driver of the OOM below
            # plain alphabetical order picks zeta={1,11} only (9 values span 1-17) -- a
            # known bad selection from before this file existed. Span zeta AND alpha.
            file_select_fn=lambda files, k: dl.select_representative_files(files, k, _param_key),
            batch=8,
            traj_cache_capacity=8,  # literal original TRAJ_CACHE_CAPACITY value -- unchanged from before
            decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                                 scalar_field_names=["concentration"], tensor_field_names=["D", "E"]),
        ),
    }

    if task_set == "core3":
        return core3

    if task_set != "all14":
        raise ValueError(f"Unknown task_set '{task_set}'; expected 'core3' or 'all14'")

    all14 = dict(core3)
    all14["rayleigh_benard_uniform"] = dict(
        # identical field layout/physics to rayleigh_benard -- reuse RB_FIELD_SPEC and
        # RB's exact decoder_kwargs (legacy pressure/buoyancy path) directly.
        field_spec=dl.RB_FIELD_SPEC,
        data_dir=f"{data_root}/rayleigh_benard_uniform/data",
        param_subset=None,
        train_file_limit=8,
        val_file_limit=8,
        # plain alphabetical order clusters on Rayleigh=1e10 ("1e10" < "1e6" as a
        # string) -- a file_limit=8 head-slice picks 7 of 7 Rayleigh=1e10 combos plus
        # one 1e6, entirely missing 1e7/1e8/1e9. Span Rayleigh's actual range instead.
        file_select_fn=lambda files, k: dl.select_representative_files(files, k, _param_key),
        batch=8,
        traj_cache_capacity=_traj_cache_capacity(200, 4, 512, 128),
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True),
    )
    for name in ("acoustic_scattering_discontinuous", "acoustic_scattering_inclusions", "acoustic_scattering_maze"):
        all14[name] = dict(
            field_spec=dl.ACOUSTIC_SCATTERING_FIELD_SPEC,
            data_dir=f"{data_root}/{name}/data",
            param_subset=None,
            train_file_limit=4,
            val_file_limit=2,
            traj_limit=15,  # of ~100 trajectories/file; files are unlabeled homogeneous chunks, not param combos
            batch=8,
            traj_cache_capacity=_traj_cache_capacity(102, 3, 256, 256),
            # pressure = acoustic perturbation from the (separately-stored, static,
            # currently-unused -- see dataloader.py) ambient medium -- naturally
            # zero-mean, same gauge-freedom reasoning as rayleigh_benard's pressure.
            # Not incompressible viscous flow (this is wave propagation) -> no streamfunction.
            decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=False,
                                 scalar_field_names=["pressure"]),
        )
    for name in ("euler_multi_quadrants_openBC",):  # periodicBC excluded -- see build_tasks docstring
        all14[name] = dict(
            field_spec=dl.EULER_MULTI_QUADRANTS_FIELD_SPEC,
            data_dir=f"{data_root}/{name}/data",
            param_subset=None,
            train_file_limit=6,  # of 10 -- each file is a distinct gamma value, keep most of them
            val_file_limit=4,
            file_select_fn=lambda files, k: dl.select_representative_files(files, k, _param_key),
            traj_limit=3,  # of 400 realizations/file -- the real volume lever here
            batch=2,  # 512x512 x 5 channels, largest per-sample footprint of any task here
            traj_cache_capacity=_traj_cache_capacity(101, 5, 512, 512),
            # compressible Euler: density/energy/pressure all vary independently, this is
            # NOT incompressible flow -> no streamfunction, and pressure has a real
            # physically-meaningful nonzero mean (background pressure) -> don't zero it.
            decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                                 scalar_field_names=["density", "energy", "pressure"]),
        )
    all14["gray_scott_reaction_diffusion"] = dict(
        field_spec=dl.GRAY_SCOTT_FIELD_SPEC,
        data_dir=f"{data_root}/gray_scott_reaction_diffusion/data",
        param_subset=None,
        train_file_limit=6,  # all of them -- each is a distinct (F,k) combo, real physical diversity
        val_file_limit=6,
        traj_limit=2,  # of 160 realizations/file
        pair_stride=15,  # T=1001 -- a single trajectory alone yields ~990 windows otherwise
        batch=8,
        traj_cache_capacity=_traj_cache_capacity(1001, 2, 128, 128),
        # pure reaction-diffusion: no velocity field in the data at all.
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["A", "B"], predict_velocity=False),
    )
    all14["helmholtz_staircase"] = dict(
        field_spec=dl.HELMHOLTZ_STAIRCASE_FIELD_SPEC,
        data_dir=f"{data_root}/helmholtz_staircase/data",
        param_subset=None,
        train_file_limit=8,  # of 16 omega values
        val_file_limit=8,
        file_select_fn=lambda files, k: dl.select_representative_files(files, k, _param_key),
        traj_limit=5,  # of 26 trajectories/file
        batch=4,  # 1024x256 grid -- same pixel count as euler's 512x512
        traj_cache_capacity=_traj_cache_capacity(50, 2, 1024, 256),
        # frequency-domain acoustics: no velocity field, pressure_re/pressure_im are a
        # complex pressure field's real/imaginary parts, not a "pressure" with a
        # physically meaningful sign convention to zero-mean.
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["pressure_re", "pressure_im"], predict_velocity=False),
    )
    # planetswe excluded from all14 -- see build_tasks docstring. The entry (field_spec,
    # file/traj limits, decoder_kwargs) lived here unchanged; restore verbatim from
    # git history/EXPERIMENT_LOG if a future task_set wants it back.
    all14["turbulent_radiative_layer_2D"] = dict(
        field_spec=dl.TURBULENT_RADIATIVE_LAYER_2D_FIELD_SPEC,
        data_dir=f"{data_root}/turbulent_radiative_layer_2D/data",
        param_subset=None,
        # small dataset already (9 files, N=8 traj/file) -- no limits needed
        batch=8,
        traj_cache_capacity=_traj_cache_capacity(101, 4, 128, 384),
        # compressible (density varies) -> no streamfunction, pressure has a real
        # background mean -> don't zero it.
        decoder_kwargs=dict(zero_mean_pressure=False, use_streamfunction=False,
                             scalar_field_names=["density", "pressure"]),
    )
    all14["viscoelastic_instability"] = dict(
        field_spec=dl.VISCOELASTIC_INSTABILITY_FIELD_SPEC,
        data_dir=f"{data_root}/viscoelastic_instability/data",
        param_subset=None,
        # small dataset already (7 files, N=16 traj/file, T=20) -- no limits needed
        batch=2,  # 512x512 x 8 channels (c_zz, pressure, velocity x2, C x4)
        traj_cache_capacity=_traj_cache_capacity(20, 8, 512, 512),
        # Oldroyd-B/FENE-P-style viscoelastic flow (per the Re/Wi/beta/epsilon/Lmax
        # params): classic formulation carries an incompressible Newtonian-plus-polymer-
        # stress momentum equation, same family as shear_flow -> streamfunction +
        # zero-mean pressure, same justification.
        decoder_kwargs=dict(zero_mean_pressure=True, use_streamfunction=True,
                             scalar_field_names=["c_zz", "pressure"], tensor_field_names=["C"]),
    )
    return all14


def repeat_forever(loader):
    """Re-iterate `loader` indefinitely, starting a fresh (re-shuffled) pass
    each time the previous one is exhausted. NOT itertools.cycle(loader):
    cycle() caches every item it has ever yielded internally so it can
    replay them once exhausted -- catastrophic for large batch tensors
    repeated across a long run (this was the direct cause of this script's
    two earlier OOM crashes). This only ever holds the current pass's
    DataLoader iterator; a finished pass's batches are freed normally as
    soon as the training loop moves past them.
    """
    while True:
        for batch in loader:
            yield batch


def _flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Current .grad of every param, concatenated into one flat vector (params with no
    grad this step -- e.g. a decoder belonging to a different task -- contribute zeros,
    so every task's vector has the same shape and a plain dot product/projection between
    two tasks' vectors is well-defined)."""
    return torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in params
    ])


def _set_grads_from_flat(params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p).clone()
        offset += n


def pcgrad_combine(task_grads: List[torch.Tensor], rng: random.Random) -> torch.Tensor:
    """PCGrad (Yu et al. 2020, "Gradient Surgery for Multi-Task Learning"): the plain
    sum-of-gradients this script used before this function existed has no way to tell
    "two tasks pull the shared trunk in genuinely different but compatible directions"
    (fine) apart from "two tasks pull it in directly opposing directions" (one
    partially cancels/corrupts the other's update). For each task's gradient, project
    away the component that conflicts (negative dot product) with each other task's
    gradient, in random order, before summing -- the shared, non-conflicting part of
    every task's gradient is kept in full; only the conflicting part is removed, and
    only from the side that's doing the conflicting (not symmetrically zeroed).

    task_grads: one flat gradient vector per task, same length (see _flatten_grads).
    Returns the single combined flat gradient to actually apply.
    """
    n = len(task_grads)
    pc_grads = [g.clone() for g in task_grads]
    for i in range(n):
        order = list(range(n))
        rng.shuffle(order)
        for j in order:
            if j == i:
                continue
            dot = torch.dot(pc_grads[i], task_grads[j])
            if dot < 0:
                pc_grads[i] -= dot * task_grads[j] / (task_grads[j].norm() ** 2 + 1e-12)
    return torch.stack(pc_grads).sum(dim=0)


def soft_cap_loss(loss: torch.Tensor, threshold: float) -> torch.Tensor:
    """Huber-style soft cap: identity below `threshold` (every normal batch is
    completely unaffected, full gradient, nothing discarded), sqrt-growth
    above it (an anomalous batch still pulls the model away from whatever
    produced it -- the signal isn't thrown away -- it just can't dominate the
    shared trunk's gradient the way a raw 10^11-scale loss would). No hard
    skip, no fixed clamp: this only reduces marginal influence past the
    threshold, continuously (value-continuous at the boundary; derivative
    has a kink there, same as ordinary Huber loss).

    Root cause (see EXPERIMENT_LOG.md): a rare, non-physical anomaly in the
    raw simulation data (a real solver artifact, confirmed by direct
    inspection -- not something this pipeline introduces) drove one batch's
    loss to ~10^11. Gradient-norm clipping alone caps the resulting step
    *size* but not its *direction* -- this caps the loss itself, before that
    direction is ever computed.

    A literal inf/nan raw loss is handled separately, before the threshold
    check: sqrt(excess) is itself inf/nan when `loss` is, so the sqrt-cap
    above only protects against large-but-FINITE anomalies -- an actual
    inf/nan means the forward pass broke down entirely (not just "a very bad
    batch"), and backpropagating through a real inf/nan produces inf/nan
    gradients with no meaningful direction, unlike the finite case where
    sqrt-capping still yields a well-defined finite gradient. Returns a fresh
    zero (no graph connection, so .backward() is a true no-op) rather than
    let it poison the shared trunk. Confirmed necessary in practice, not
    theoretical: an all14-task run saw planetswe's raw loss hit literal inf
    around step 10, and every other task's train_loss visibly degraded in
    lockstep the same steps -- shared-trunk gradient poisoning from a single
    task's exploded loss, exactly what this whole mechanism exists to prevent.
    """
    if not torch.isfinite(loss):
        return torch.zeros_like(loss)
    if loss <= threshold:
        return loss
    excess = loss - threshold
    return threshold + torch.sqrt(excess + 1.0) - 1.0


def union_channel_specs(tasks: dict) -> dict:
    specs = {}
    for cfg in tasks.values():
        for spec in cfg["field_spec"]:
            name = canonical_field_name(spec["key"])
            n = spec["n_components"]
            if name in specs and specs[name] != n:
                raise ValueError(f"Field '{name}' has conflicting n_components across tasks: {specs[name]} vs {n}")
            specs[name] = n
    return specs


def build_foundation_model(model_config: dict, device) -> FoundationModel:
    """Single source of truth for constructing a FoundationModel -- used both when
    starting training (model_config built from CLI args, see main()) and when
    loading a checkpoint for eval/resume (model_config loaded from the checkpoint's
    own saved "model_config" key). The two call sites read the exact same function,
    so they cannot drift apart from each other the way hand-duplicated construction
    logic in separate eval scripts previously could (and did -- see the 2026-08-20
    EXPERIMENT_LOG entry on checkpoint drift). The only way to change how a
    FoundationModel gets built is to change it here.

    model_config keys: canonical_dim, latent_dim, hidden_dim, cond_dim,
    decoder_hidden_dim, context_cond_hidden_dim, complex_term, amplitude_scale,
    boundary_geometry (bool), context_frames, and tasks (dict of task_name ->
    {"field_spec": ..., "decoder_kwargs": ...} -- just the two fields needed to
    rebuild the architecture, not the full build_tasks() config with data-loading-
    specific keys like batch/traj_limit/data_dir that eval doesn't need).
    """
    mc = model_config
    T = mc["context_frames"]
    # .get with the pre-existing default: old checkpoints/model_configs saved
    # before this key existed still load unchanged (every prior checkpoint was
    # built with kernel_size=7 everywhere, ConvNeXtBlock's own default).
    block_kernel_size = mc.get("block_kernel_size", 7)

    field_embedder = FieldEmbedder(mc["tasks"], canonical_dim=mc["canonical_dim"]).to(device)
    boundary_geometry = (
        BoundaryGeometryHead(mc["canonical_dim"], list(mc["tasks"].keys())).to(device)
        if mc["boundary_geometry"] else None
    )
    context_cond_encoder = PooledContextCondEncoder(
        in_channels=mc["canonical_dim"], context_frames=T, cond_dim=mc["cond_dim"],
        hidden_dim=mc["context_cond_hidden_dim"],
    ).to(device)
    encoder = SequenceConvEncoder(
        in_channels=mc["canonical_dim"], context_frames=T, latent_dim=mc["latent_dim"], hidden_dim=mc["hidden_dim"],
        block_kernel_size=block_kernel_size,
    ).to(device)
    operator = TransportOperator(
        latent_dim=mc["latent_dim"], hidden_dim=mc["hidden_dim"], cond_dim=mc["cond_dim"],
        terms=tuple(mc["operator_terms"]), film=True,
        complex_term=mc["complex_term"],
        block_kernel_size=block_kernel_size,
        # amplitude_scale is no longer a TransportOperator param (2026-08-22: removed,
        # it was fixed at 1.0 everywhere it was ever actually used -- see
        # ComplexAmplitudeTerm's docstring/forward). model_config may still carry an
        # "amplitude_scale" key from an older checkpoint; deliberately ignored here
        # rather than passed through, so old checkpoints still load.
    ).to(device)
    decoders = {
        t: SharedTrunkFieldHeadsDecoder(
            latent_dim=mc["latent_dim"], hidden_dim=mc["decoder_hidden_dim"], upsample=2,
            block_kernel_size=block_kernel_size,
            **mc["tasks"][t]["decoder_kwargs"],
        ).to(device)
        for t in mc["tasks"]
    }
    return FoundationModel(
        field_embedder, context_cond_encoder, encoder, operator, decoders,
        boundary_geometry=boundary_geometry,
    ).to(device)


def load_foundation_checkpoint(checkpoint_path: str, device) -> tuple:
    """The single, drift-proof way to load a saved FoundationModel checkpoint: builds
    the model via build_foundation_model() from the checkpoint's own model_config,
    then loads its full model_state_dict -- one call, no separate reconstruct-every-
    submodule-by-hand logic to keep in sync (that duplication across eval scripts is
    exactly what caused this checkpoint format to drift in the first place -- see the
    2026-08-20 EXPERIMENT_LOG entry). Returns (model, checkpoint_dict) -- the raw
    dict is still returned for callers that want epoch/val_loss_by_task/etc.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_foundation_model(ckpt["model_config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-frames", type=int, required=True)
    p.add_argument("--rollout-steps", type=int, required=True)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    # 2026-08-16: walked the blanket 2x-every-width expansion above back down --
    # it measurably blew up the all14 run's GPU footprint (114GB/183GB on a B200
    # for a model that's still only a few M params) without the five dims being
    # equally responsible. Traced actual memory flow instead of scaling
    # everything uniformly again:
    #   - latent_dim: real+complex spectral grid, held live across all
    #     rollout_steps for backprop (complex64 = 2x a real tensor at the same
    #     shape) -- the single biggest lever. Full revert to core3's 16.
    #   - decoder_hidden_dim: spatial conv run over the whole rollout flattened
    #     into the batch dim (B*rollout_steps) -- second-biggest lever. Full
    #     revert to core3's 48.
    #   - hidden_dim: spatial conv at H/2 and H/4 (shared by encoder + cond
    #     trunk) -- real but downsampled, so a moderate cost. Full revert to
    #     core3's 64.
    #   - canonical_dim: only 1x1 convs (no spatial mixing) but at *full* H x W,
    #     the largest resolution anywhere in the model -- linear cost, no
    #     rollout multiplier. Kept above the core3 baseline (16 vs 12) since the
    #     original per-task-capacity motivation still applies here specifically.
    #   - 2026-08-19: cond_dim reverted 128 -> 32. The 128 value (set 2026-08-16)
    #     was reasoned as "cheap so why not bigger" -- cheap in FLOPs/memory, but
    #     that's not the same as harmless to convergence, and it moved away from
    #     the one config actually known to reach a good number:
    #     configs/context_cond_pooled_helmholtz.yaml (the literal source of
    #     EXPERIMENT_LOG's 0.229 rayleigh_benard/broad8 result) uses cond_dim=32.
    #     No evidence 128 ever helped here; reverting to the known-good value
    #     rather than re-guessing.
    p.add_argument("--canonical-dim", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--cond-dim", type=int, default=32)
    p.add_argument("--context-cond-hidden-dim", type=int, default=128,
                    help="PooledContextCondEncoder's own internal trunk width, independent of the "
                         "shared --hidden-dim (used by the main encoder/operator/decoder). Matches "
                         "configs/context_cond_pooled_helmholtz.yaml, whose own comment explains why: "
                         "the pooled per-frame-then-pool architecture has headroom a shared 64 doesn't "
                         "give it. Previously this script reused --hidden-dim for both, silently losing "
                         "that distinction.")
    p.add_argument("--decoder-hidden-dim", type=int, default=48)
    p.add_argument("--boundary-geometry", action=argparse.BooleanOptionalAction, default=True,
                    help="2026-08-20: per-axis learned soft blend of zero/circular/replicate padding "
                         "(BoundaryGeometryHead + core/boundary.py), replacing the previous unconditional "
                         "torch.roll (always-circular) in the operator's advection/diffusion terms and the "
                         "streamfunction-derived velocity decode, plus the encoder's ConvNeXtBlock padding. "
                         "This dataset spans at least 6 distinct per-axis BC regimes (periodic/wall-Dirichlet/"
                         "wall-no-slip/open/open-Neumann/mixed-asymmetric -- see EXPERIMENT_LOG.md) that were "
                         "previously all treated identically. Zero-inited so training starts behaviorally "
                         "identical to the old code (100%% circular) and only drifts as useful -- on by "
                         "default; --no-boundary-geometry disables it entirely (every forward call falls back "
                         "to the pre-2026-08-20 unconditional behavior) for ablation.")
    p.add_argument("--operator-terms", default="advection,diffusion,skew,forcing",
                    help="comma-separated subset of TransportOperator.REAL_TERM_BUILDERS to include "
                         "(advection, diffusion, skew, forcing, local_attention) -- for term-ablation runs. "
                         "Default matches this script's original hard-coded terms tuple, so existing "
                         "invocations are unaffected. Validated against the operator's own registry, not "
                         "silently ignored on a typo.")
    # 2026-08-22 (ported from a parallel local branch): --block-kernel-size was
    # independently added alongside a blanket 2x-every-width rescale (24/32/128/64/96)
    # that predates -- and is now superseded by -- the more careful per-dim tracing
    # above (16/16/64/32/48), so only the width dims that trace above are kept; this
    # flag is orthogonal to that rescale (a different lever, not another width dim)
    # and is kept on its own merits.
    p.add_argument("--block-kernel-size", type=int, default=7,
                    help="depthwise kernel size inside every ConvNeXtBlock/FiLMConvNeXtBlock in the shared "
                         "trunk + decoders (context_cond_encoder, encoder, all real/complex operator terms' "
                         "conv bodies, decoder trunks). Depthwise conv cost scales as dim*K^2 (linear in "
                         "channels), while the pointwise convs that dominate each block's FLOPs scale as "
                         "dim^2 -- widening this is a cheap way to add real capacity/params without the "
                         "memory/runtime hit of widening canonical-dim/hidden-dim/decoder-hidden-dim "
                         "themselves (see EXPERIMENT_LOG.md's fused_spectral_encoder_big for the same lever "
                         "applied to the old split_encoder family). Default 7 matches every prior checkpoint.")
    p.add_argument("--batch-scale", type=float, default=1.0,
                    help="multiplies every task's own cfg['batch'] (rounded down, floor 1) -- a pure "
                         "runtime-memory knob for fitting a given GPU, doesn't touch model capacity/quality "
                         "the way shrinking canonical-dim/hidden-dim/decoder-hidden-dim would.")
    p.add_argument("--n-substeps", type=int, default=1,
                    help="DISCO-inspired sub-step integration: call the operator N times per output frame "
                         "at dt=1/N instead of once at dt=1 (see TransportOperator.forward's dt handling and "
                         "EXPERIMENT_LOG.md §20/§23). N=1 (default) is exactly the original single-full-step "
                         "behavior, unchanged for every existing config/checkpoint.")
    p.add_argument("--complex-term", action=argparse.BooleanOptionalAction, default=True,
                    help="add the FFT-fed complex-rotation branch (complex_proj + ComplexAmplitude/RotationTerm) "
                         "on top of the real Helmholtz terms -- see EXPERIMENT_LOG.md for the single-task history. "
                         "On by default: the complex branch beat real-only on core3 (0.2847 vs 0.3237 best mean "
                         "val loss) -- pass --no-complex-term to disable.")
    p.add_argument("--amplitude-scale", type=float, default=1.0,
                    help="tanh ceiling on the complex branch's per-step log-amplitude factor, exp(+-this)")
    p.add_argument("--loss-cap-threshold", type=float, default=25.0,
                    help="soft_cap_loss threshold -- comfortably above any observed normal-batch loss across "
                         "all three tasks (including active_matter's noisier variance-normalized spikes), well "
                         "below where a genuinely anomalous batch's raw loss lands (thousands+)")
    p.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True,
                    help="PCGrad (see pcgrad_combine): deconflict each task's gradient against every other "
                         "task's before summing, instead of just summing them directly. On by default for any "
                         "task_set with more than one task; pass --no-pcgrad to fall back to plain summing.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for PCGrad's per-step task-pairing order")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-dir", default="/app/checkpoints/foundation_model")
    p.add_argument("--data-root", default="/app/data/datasets",
                    help="parent dir containing <task>/data/{train,valid} for each task in TASKS")
    p.add_argument("--config-root", default="/app/configs",
                    help="parent dir containing param_subsets/*.json")
    p.add_argument("--task-set", default="core3", choices=["core3", "all14"],
                    help="core3 = original rayleigh_benard/shear_flow/active_matter recipe (default, "
                         "unchanged behavior). all14 = every 2D Well dataset, see build_tasks() for "
                         "per-task sampling choices.")
    p.add_argument("--task-subset", default=None,
                    help="comma-separated task names to keep from whichever --task-set was built, "
                         "e.g. 'rayleigh_benard,shear_flow' to drop active_matter from core3 for a "
                         "quick regression check against a smaller shared trunk. Every entry's own "
                         "config (batch, traj_cache_capacity, param_subset, decoder_kwargs, ...) is "
                         "unchanged -- this only shrinks which tasks train_loaders/task_names include, "
                         "same mechanism as leaving them out of build_tasks() entirely, just without "
                         "editing that function per one-off experiment. Unknown names raise, not silently "
                         "no-op, so a typo doesn't quietly train on more tasks than intended. Restricting "
                         "to one task also disables PCGrad (nothing to deconflict against) regardless of "
                         "--pcgrad.")
    p.add_argument("--train-file-limit-override", type=int, default=None,
                    help="overrides every selected task's own train_file_limit (a quick/fast-iteration "
                         "knob, analogous to rayleigh_benard's quick2.json subset but general to any "
                         "task -- e.g. --train-file-limit-override 2).")
    p.add_argument("--val-file-limit-override", type=int, default=None,
                    help="same as --train-file-limit-override, for val_file_limit.")
    p.add_argument("--param-subset-override", default=None,
                    help="overrides every selected task's own param_subset path -- needed instead of/on "
                         "top of --train-file-limit-override for tasks (e.g. shear_flow) whose file "
                         "selection is driven by an explicit param_subset JSON, which takes precedence "
                         "over train_file_limit in create_param_dataloaders. Points at a "
                         "configs/param_subsets/*.json path, same format as rayleigh_benard's quick2.json.")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                    help="if set, every 'epoch' is exactly this many steps regardless of any task's "
                         "natural loader length (every task -- including the largest -- draws via "
                         "repeat_forever, so none of them is privileged as 'defines the epoch'). If "
                         "unset, falls back to the original behavior: the longest task's own loader "
                         "length defines the epoch. Recommended when --task-set=all14, since letting "
                         "the largest of 14 tasks define the epoch forces the smallest ones to repeat "
                         "hundreds of times per epoch (core3's active_matter already does this ~47x "
                         "with 3 tasks; at 14 tasks the smallest natural loaders are ~10x smaller still).")
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    T, K = args.context_frames, args.rollout_steps
    tasks = build_tasks(args.data_root, args.config_root, task_set=args.task_set)
    if args.task_subset is not None:
        wanted = [t.strip() for t in args.task_subset.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in tasks]
        if unknown:
            raise ValueError(f"--task-subset names not in task_set='{args.task_set}': {unknown} "
                              f"(available: {sorted(tasks.keys())})")
        tasks = {t: tasks[t] for t in wanted}
        if len(tasks) == 1 and args.pcgrad:
            print(f"--task-subset restricted to a single task ({wanted[0]}) -- PCGrad has nothing to "
                  f"deconflict against, disabling it regardless of --pcgrad.")
            args.pcgrad = False

    if args.train_file_limit_override is not None or args.val_file_limit_override is not None:
        for cfg in tasks.values():
            if args.train_file_limit_override is not None:
                cfg["train_file_limit"] = args.train_file_limit_override
            if args.val_file_limit_override is not None:
                cfg["val_file_limit"] = args.val_file_limit_override

    if args.param_subset_override is not None:
        for cfg in tasks.values():
            cfg["param_subset"] = args.param_subset_override

    channel_specs = union_channel_specs(tasks)
    print(f"Canonical field registry: {channel_specs}")

    # model_config is the one dict that fully determines FoundationModel's
    # architecture -- saved verbatim into every checkpoint (see the save block
    # below) and fed to build_foundation_model() here and at load time, so
    # construction can never drift out of sync with what actually gets loaded.
    model_config = {
        "canonical_dim": args.canonical_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "cond_dim": args.cond_dim,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "context_cond_hidden_dim": args.context_cond_hidden_dim,
        "block_kernel_size": args.block_kernel_size,
        "complex_term": args.complex_term,
        "amplitude_scale": args.amplitude_scale,
        "boundary_geometry": args.boundary_geometry,
        "operator_terms": [t.strip() for t in args.operator_terms.split(",") if t.strip()],
        "context_frames": T,
        "tasks": {t: {"field_spec": cfg["field_spec"], "decoder_kwargs": cfg["decoder_kwargs"]}
                  for t, cfg in tasks.items()},
    }
    _unknown_terms = [t for t in model_config["operator_terms"] if t not in TransportOperator.REAL_TERM_BUILDERS]
    if _unknown_terms:
        raise ValueError(f"--operator-terms has unknown term(s) {_unknown_terms}; "
                          f"available: {sorted(TransportOperator.REAL_TERM_BUILDERS.keys())}")
    model = build_foundation_model(model_config, dev)

    train_loaders, val_loaders = {}, {}
    for task, cfg in tasks.items():
        param_choices = None
        if cfg.get("param_subset"):
            with open(cfg["param_subset"]) as f:
                param_choices = [tuple(c) for c in json.load(f)["combos"]]

        task_batch = max(1, int(cfg["batch"] * args.batch_scale))
        train_loader, val_loader, _ = dl.create_param_dataloaders(
            cfg["data_dir"], batch_size=task_batch, context_frames=T, predict_frames=K,
            num_workers=args.num_workers, param_choices=param_choices,
            train_file_limit=cfg.get("train_file_limit"), val_file_limit=cfg.get("val_file_limit"),
            traj_limit=cfg.get("traj_limit"), pair_stride=cfg.get("pair_stride", 1),
            field_spec=cfg["field_spec"], traj_cache_capacity=cfg["traj_cache_capacity"],
            file_select_fn=cfg.get("file_select_fn"),
        )
        train_loaders[task] = train_loader
        val_loaders[task] = val_loader
        print(f"[{task}] {len(train_loader)} train batches/epoch, {len(val_loader)} val batches, batch={task_batch}")

    opt = SOAP(model.parameters(), lr=args.lr)
    params = [p for p in model.parameters() if p.requires_grad]
    pcgrad_rng = random.Random(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(tasks.keys())
    if args.steps_per_epoch is not None:
        # Fixed budget: every task (including the largest) draws via repeat_forever,
        # none is privileged as "defines the epoch" -- see --steps-per-epoch help.
        steps_per_epoch = args.steps_per_epoch
    else:
        # Original behavior: the longest task's own loader length defines the epoch
        # (seen exactly once, no repeats), shorter tasks restart mid-epoch as needed
        # to keep supplying a batch every step.
        steps_per_epoch = max(len(train_loaders[t]) for t in task_names)
    task_iters = {t: repeat_forever(train_loaders[t]) for t in task_names}

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        task_loss_sum = {t: 0.0 for t in task_names}
        interval_start = time.perf_counter()

        for step in range(1, steps_per_epoch + 1):
            step_loss = 0.0
            task_grad_vecs = []
            for task in task_names:
                xb, yb, bparams = next(task_iters[task])
                xb, yb = xb.to(dev), yb.to(dev)
                initial, pred = model(xb, field_spec=tasks[task]["field_spec"], task=task, steps=K,
                                       return_initial_encode=True, n_substeps=args.n_substeps)
                initial_target = xb[:, -1, ...]
                initial_loss = well_style_vrmse(initial.unsqueeze(1), initial_target.unsqueeze(1)).mean()
                rollout_loss = well_style_vrmse(pred, yb).mean()
                raw_task_loss = (1.0 / K) * initial_loss + rollout_loss

                task_loss = soft_cap_loss(raw_task_loss, args.loss_cap_threshold)
                if not torch.isfinite(raw_task_loss):
                    print(f"  [non-finite loss SKIPPED, zero grad contributed] epoch={ep} step={step} "
                          f"task={task} raw_loss={raw_task_loss.item():.6g} bparams[0]={bparams[0].tolist()}")
                elif task_loss.item() > args.loss_cap_threshold:
                    print(f"  [soft-cap engaged] epoch={ep} step={step} task={task} "
                          f"raw_loss={raw_task_loss.item():.6g} capped_loss={task_loss.item():.6g} "
                          f"bparams[0]={bparams[0].tolist()}")

                # Each task's gradient computed in isolation (zero_grad before, capture
                # after) rather than accumulated in place -- PCGrad needs every task's
                # own gradient as a separate vector to pairwise-deconflict; see
                # pcgrad_combine. With --no-pcgrad this reduces to exactly the same math
                # as the old accumulate-in-place scheme (summed below), just computed
                # via separate backward passes instead of one shared .grad buffer.
                opt.zero_grad()
                task_loss.backward()
                task_grad_vecs.append(_flatten_grads(params))

                loss_item = float(task_loss.item())
                step_loss += loss_item
                task_loss_sum[task] += loss_item

            combined = (
                pcgrad_combine(task_grad_vecs, pcgrad_rng) if args.pcgrad
                else torch.stack(task_grad_vecs).sum(dim=0)
            )
            _set_grads_from_flat(params, combined)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += step_loss

            if step % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - interval_start) * 1000.0
                print(f"Epoch {ep}  step {step}/{steps_per_epoch}  "
                      f"combined_loss {step_loss:.6f}  time/step {elapsed_ms / args.log_interval:.2f}ms")
                interval_start = time.perf_counter()

        print(f"Epoch {ep}   train_loss (avg combined-step loss): {loss_sum / steps_per_epoch:.6f}  ({steps_per_epoch} steps)")
        for t in task_names:
            print(f"Epoch {ep}   train_loss[{t}]: {task_loss_sum[t] / steps_per_epoch:.6f}  ({steps_per_epoch} batches)")

        # Per-task validation, kept separate -- not comparable across tasks
        # (different channel composition / normalization distribution each
        # time, per EXPERIMENT_LOG.md).
        model.eval()
        val_results = {}
        with torch.no_grad():
            for t in task_names:
                vloss_sum, vbatches = 0.0, 0
                for xb, yb, _bparams in val_loaders[t]:
                    xb, yb = xb.to(dev), yb.to(dev)
                    pred = model(xb, field_spec=tasks[t]["field_spec"], task=t, steps=K,
                                 return_initial_encode=False, n_substeps=args.n_substeps)
                    vloss_sum += float(well_style_vrmse(pred, yb).mean().item())
                    vbatches += 1
                vloss_avg = vloss_sum / max(1, vbatches)
                val_results[t] = vloss_avg
                print(f"Epoch {ep}   valid_loss[{t}]: {vloss_avg:.6f}")

        # Per-task tag in the filename works for core3 (3 short-ish names) but not at
        # scale: 14 task names + losses blew past the filesystem's ~255-byte filename
        # limit and crashed the save entirely (RuntimeError: File name too long),
        # losing that epoch's weights. The full per-task breakdown is already saved
        # inside the checkpoint itself (val_loss_by_task below) -- the filename only
        # ever needed to be a quick-glance summary, so use the mean instead of every
        # task's own number. Safe regardless of how many tasks are in play.
        avg_val = sum(val_results.values()) / max(1, len(val_results))
        # Same _sub{N} tag convention as train.py's single-task checkpoints -- absent
        # (empty string) at the N=1 default so every existing filename is unaffected.
        substep_tag = f"_sub{args.n_substeps}" if args.n_substeps != 1 else ""
        ckpt_out = save_dir / f"foundation_ep{ep}_ctx{T}_roll{K}{substep_tag}_avg{avg_val:.4f}.pt"
        torch.save({
            # The whole registered module tree in one call -- field_embedder (every
            # per-task copy), context_cond_encoder, encoder, operator, complex_proj
            # (present iff --complex-term; simply absent from state_dict when
            # model.complex_proj is None, same "no key means no complex branch"
            # contract the old format had -- nothing special needed to preserve it),
            # every task's decoder, boundary_geometry (present iff
            # --boundary-geometry). Adding a new submodule to FoundationModel in the
            # future needs zero changes here to be included -- this is the fix for
            # the 2026-08-20 checkpoint-drift incident (boundary_geometry's weights
            # were silently never saved until this).
            "model_state_dict": model.state_dict(),
            # Everything needed to reconstruct this exact architecture via
            # build_foundation_model() before loading model_state_dict into it --
            # see load_foundation_checkpoint().
            "model_config": model_config,
            "rollout_steps": K,
            "n_substeps": args.n_substeps,
            "epoch": ep,
            "val_loss_by_task": val_results,
        }, ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")


if __name__ == "__main__":
    main()
