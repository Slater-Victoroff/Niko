import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict, defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random

from core.states import FieldState


RB_CHANNEL_ORDER = ("pressure", "buoyancy", "velocity_x", "velocity_y")
STACK_FRAMES_REMOVED_ERROR = (
    "stack_frames has been removed for Rayleigh-Benard training because it created invalid "
    "velocity-pair channels. Use physical channels [pressure, buoyancy, velocity_x, velocity_y]."
)

# A field_spec is an ordered list of {"key": <hdf5 dataset path>, "n_components": <int>}
# describing which datasets to load and stack into channels. n_components=1 for a scalar
# (t0_fields/*), 2 for a vector (t1_fields/*), 4 for a flattened 2x2 tensor (t2_fields/*,
# row-major: T_00,T_01,T_10,T_11) -- generalizes the same channel-stacking pipeline across
# every field rank The Well's format uses, not just RB's fixed 4-channel case.
RB_FIELD_SPEC = [
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t0_fields/buoyancy", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
]

ACTIVE_MATTER_FIELD_SPEC = [
    {"key": "t0_fields/concentration", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
    {"key": "t2_fields/D", "n_components": 4},
    {"key": "t2_fields/E", "n_components": 4},
]

SHEAR_FLOW_FIELD_SPEC = [
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t0_fields/tracer", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
]

# --- Remaining 2D Well datasets (added when scaling the foundation model beyond
# rayleigh_benard/shear_flow/active_matter to all 14 2D tasks). Confirmed against
# actual on-disk HDF5 structure (h5py direct inspection on AICR), not assumed from
# dataset names -- see inline notes for anything non-obvious.

# acoustic_scattering_{discontinuous,inclusions,maze} all share this exact layout.
# t0_fields/density and t0_fields/speed_of_sound are real fields but have NO time
# axis (shape (N,X,Y), not (N,T,X,Y) -- they describe the static acoustic medium
# per trajectory, not a time-varying quantity) so they don't fit field_spec's
# (N,T,...) contract at all and are dropped here. This is a real information loss
# (the three acoustic_scattering variants differ precisely in their medium
# structure, and simulation_parameters is empty here too, so nothing else carries
# that signal to the model) -- fine for a first pass that only predicts
# pressure/velocity propagation, but worth flagging rather than silently omitting.
ACOUSTIC_SCATTERING_FIELD_SPEC = [
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
]

# euler_multi_quadrants_{openBC,periodicBC} share this layout: compressible Euler
# equations (density/energy/pressure all vary independently -- not incompressible,
# so use_streamfunction is wrong here), momentum (not velocity) is the vector field.
EULER_MULTI_QUADRANTS_FIELD_SPEC = [
    {"key": "t0_fields/density", "n_components": 1},
    {"key": "t0_fields/energy", "n_components": 1},
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t1_fields/momentum", "n_components": 2},
]

# Pure reaction-diffusion (Gray-Scott): two concentration fields, no flow/velocity
# field in the data at all. Decoder must be built with predict_velocity=False.
GRAY_SCOTT_FIELD_SPEC = [
    {"key": "t0_fields/A", "n_components": 1},
    {"key": "t0_fields/B", "n_components": 1},
]

# Frequency-domain Helmholtz acoustics: real/imaginary parts of a complex pressure
# field, no velocity field. t0_fields/mask (the staircase geometry) has neither an
# N nor a T axis (shape (X,Y), literally one static array shared by the whole
# dataset) so it can't fit field_spec either -- dropped, same caveat as above.
# Decoder must be built with predict_velocity=False.
HELMHOLTZ_STAIRCASE_FIELD_SPEC = [
    {"key": "t0_fields/pressure_re", "n_components": 1},
    {"key": "t0_fields/pressure_im", "n_components": 1},
]

# Shallow-water equations on a sphere (dimensions are theta/phi, not x/y, but
# H5RayleighBenardFields' axis inference is position-based (always axes 1,2)
# so this is a mechanical non-issue). Height directly couples to velocity
# divergence via mass conservation, so this is NOT divergence-free --
# use_streamfunction=False, same reasoning as the compressible tasks above.
PLANETSWE_FIELD_SPEC = [
    {"key": "t0_fields/height", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
]

# Same physical system and field layout as rayleigh_benard (RB_FIELD_SPEC) --
# rayleigh_benard_uniform is just a differently-sampled parameter grid, not a
# different simulation. No separate field_spec needed; reuse RB_FIELD_SPEC
# directly at the call site.

# Compressible radiative-cooling instability: density and pressure both vary
# independently (not incompressible) -- use_streamfunction=False.
TURBULENT_RADIATIVE_LAYER_2D_FIELD_SPEC = [
    {"key": "t0_fields/density", "n_components": 1},
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
]

# Viscoelastic (Oldroyd-B/FENE-P-style, per the Re/Wi/beta/epsilon/Lmax params)
# flow: c_zz is the out-of-plane component of the polymer conformation tensor
# (present even in a 2D flow for this class of constitutive model), C is the
# in-plane 2x2 conformation tensor. Classic viscoelastic formulations carry an
# incompressible Newtonian-plus-polymer-stress momentum equation, so
# use_streamfunction=True is used here, same justification as shear_flow.
VISCOELASTIC_INSTABILITY_FIELD_SPEC = [
    {"key": "t0_fields/c_zz", "n_components": 1},
    {"key": "t0_fields/pressure", "n_components": 1},
    {"key": "t1_fields/velocity", "n_components": 2},
    {"key": "t2_fields/C", "n_components": 4},
]


def field_spec_channels(field_spec: List[Dict[str, Any]]) -> int:
    return sum(spec["n_components"] for spec in field_spec)


@dataclass(frozen=True)
class PairIndex:
    file_idx: int
    traj_idx: int
    t: int


def _format_shape(shape: Tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def _infer_xy_axes_from_shape(
    traj_shape: Tuple[int, ...],
    x_len: Optional[int],
    y_len: Optional[int],
    context_label: str,
) -> Tuple[int, int]:
    """X and Y are always the two axes immediately after time (positions 1, 2) in every
    Well-format dataset's trajectory-level layout (time axis already dropped here) -- any
    extra trailing axes (vector component, tensor row/col) come after. Confirmed by
    inspecting the actual on-disk shape of all three datasets in this project
    (rayleigh_benard, shear_flow, active_matter): position, not length, is what's fixed.

    Length-matching is used only as a sanity check when x_len != y_len (unambiguous);
    skipped when they're equal (e.g. active_matter's square 256x256 grid), where length
    alone genuinely can't distinguish which is which -- matching by length there is not
    just unnecessary, it would raise (multiple axes tie), which is what motivated this
    switch from length-based inference to position-based in the first place.
    """
    if len(traj_shape) < 3:
        raise ValueError(f"{context_label} must have at least 3 dims (T + 2 spatial). Got shape={traj_shape}")

    x_axis, y_axis = 1, 2

    if x_len is not None and y_len is not None and x_len != y_len:
        if traj_shape[x_axis] != x_len or traj_shape[y_axis] != y_len:
            raise ValueError(
                f"Axis layout mismatch for {context_label}: expected positions (1, 2) to be "
                f"(x_len={x_len}, y_len={y_len}) per the fixed Well-format convention, "
                f"got shape={traj_shape}."
            )

    return x_axis, y_axis


def inspect_rayleigh_benard_h5(
    filepath: str,
    pressure_key: str = "t0_fields/pressure",
    buoyancy_key: str = "t0_fields/buoyancy",
    velocity_key: str = "t1_fields/velocity",
    include_stats: bool = True,
) -> Dict[str, Any]:
    """Inspect RB HDF5 structure and inferred axis order needed for [T, 4, H, W]."""
    out: Dict[str, Any] = {
        "file": filepath,
        "keys": [],
        "datasets": {},
        "inferred": {},
    }

    print(f"Inspecting HDF5 file: {filepath}")
    with h5py.File(filepath, "r") as f:
        def walk(g, prefix=""):
            for k, v in g.items():
                p = f"{prefix}/{k}" if prefix else k
                if isinstance(v, h5py.Dataset):
                    print(f"DATASET {p} shape={v.shape} dtype={v.dtype}")
                    out["keys"].append(p)
                    out["datasets"][p] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}
                else:
                    print(f"GROUP   {p}")
                    out["keys"].append(p)
                    walk(v, p)

        walk(f)

        required = [pressure_key, buoyancy_key, velocity_key, "dimensions/x", "dimensions/y"]
        for key in required:
            if key not in f:
                raise KeyError(f"Required key '{key}' not found in {filepath}")

        x_len = int(f["dimensions/x"].shape[0])
        y_len = int(f["dimensions/y"].shape[0])

        p_shape = tuple(f[pressure_key].shape)
        b_shape = tuple(f[buoyancy_key].shape)
        v_shape = tuple(f[velocity_key].shape)

        if len(p_shape) != 4 or len(b_shape) != 4 or len(v_shape) != 5:
            raise ValueError(
                "Unexpected RB field shapes. Expected pressure/buoyancy=(N,T,*,*) and velocity=(N,T,*,*,2). "
                f"Got pressure={p_shape}, buoyancy={b_shape}, velocity={v_shape}."
            )

        p_traj_shape = p_shape[1:]
        v_traj_shape = v_shape[1:]

        p_x_axis, p_y_axis = _infer_xy_axes_from_shape(p_traj_shape, x_len, y_len, "pressure trajectory")
        v_comp_axes = [ax for ax in range(1, len(v_traj_shape)) if v_traj_shape[ax] == 2]
        if len(v_comp_axes) != 1:
            raise ValueError(
                "Could not infer velocity component axis (size 2) uniquely from shape "
                f"{v_traj_shape}; candidate axes={v_comp_axes}."
            )
        v_comp_axis = v_comp_axes[0]
        v_x_axis, v_y_axis = _infer_xy_axes_from_shape(v_traj_shape, x_len, y_len, "velocity trajectory")

        inferred = {
            "x_len": x_len,
            "y_len": y_len,
            "pressure_traj_shape": p_traj_shape,
            "velocity_traj_shape": v_traj_shape,
            "pressure_axes": {"x": p_x_axis, "y": p_y_axis},
            "velocity_axes": {"comp": v_comp_axis, "x": v_x_axis, "y": v_y_axis},
            "target_channel_order": RB_CHANNEL_ORDER,
        }
        out["inferred"] = inferred

        print("Inferred axis mapping for [T, 4, H, W]:")
        print(f"  pressure trajectory shape={p_traj_shape} -> (T, Y, X) using y_axis={p_y_axis}, x_axis={p_x_axis}")
        print(
            f"  velocity trajectory shape={v_traj_shape} -> (T, 2, Y, X) "
            f"using comp_axis={v_comp_axis}, y_axis={v_y_axis}, x_axis={v_x_axis}"
        )
        print(f"  channel order={RB_CHANNEL_ORDER}")

        if include_stats:
            sample_idx = 0
            p_arr = np.asarray(f[pressure_key][sample_idx, ...], dtype=np.float32)
            b_arr = np.asarray(f[buoyancy_key][sample_idx, ...], dtype=np.float32)
            v_arr = np.asarray(f[velocity_key][sample_idx, ...], dtype=np.float32)

            print(
                f"STATS {pressure_key} sample traj shape={p_arr.shape} "
                f"min={np.nanmin(p_arr):.6g} max={np.nanmax(p_arr):.6g} mean={np.nanmean(p_arr):.6g} std={np.nanstd(p_arr):.6g}"
            )
            print(
                f"STATS {buoyancy_key} sample traj shape={b_arr.shape} "
                f"min={np.nanmin(b_arr):.6g} max={np.nanmax(b_arr):.6g} mean={np.nanmean(b_arr):.6g} std={np.nanstd(b_arr):.6g}"
            )
            print(
                f"STATS {velocity_key} sample traj shape={v_arr.shape} "
                f"min={np.nanmin(v_arr):.6g} max={np.nanmax(v_arr):.6g} mean={np.nanmean(v_arr):.6g} std={np.nanstd(v_arr):.6g}"
            )
            if v_arr.ndim >= 2 and v_arr.shape[-1] == 2:
                vx = v_arr[..., 0]
                vy = v_arr[..., 1]
                print(
                    f"STATS velocity_x sample traj min={np.nanmin(vx):.6g} max={np.nanmax(vx):.6g} "
                    f"mean={np.nanmean(vx):.6g} std={np.nanstd(vx):.6g}"
                )
                print(
                    f"STATS velocity_y sample traj min={np.nanmin(vy):.6g} max={np.nanmax(vy):.6g} "
                    f"mean={np.nanmean(vy):.6g} std={np.nanstd(vy):.6g}"
                )

    return out


class H5RayleighBenardFields(Dataset):
    """Despite the name (kept for backward compatibility with existing call
    sites), this class is dataset-agnostic: which HDF5 datasets to load and
    how to stack them into channels is entirely driven by `field_spec` (see
    RB_FIELD_SPEC/ACTIVE_MATTER_FIELD_SPEC/SHEAR_FLOW_FIELD_SPEC above).
    Defaults to RB_FIELD_SPEC, so every existing call site that doesn't pass
    field_spec explicitly keeps its exact current behavior.
    """

    def __init__(
        self,
        filepaths: List[str],
        field_spec: Optional[List[Dict[str, Any]]] = None,
        file_limit: Optional[int] = None,
        traj_limit: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        cache_mode: str = "file",  # "none" | "file" | "traj"
        transform=None,
        context_frames: int = 1,
        predict_frames: int = 1,
        device: Optional[torch.device] = None,
        file_params: Optional[List[Optional[tuple]]] = None,
        return_params: bool = False,
        traj_cache_capacity: int = 32,
        pair_stride: int = 1,
        traj_seed: int = 42,
        **legacy_kwargs,
    ):
        if "stack_frames" in legacy_kwargs:
            raise ValueError(STACK_FRAMES_REMOVED_ERROR)

        self.field_spec = field_spec if field_spec is not None else RB_FIELD_SPEC
        self.n_channels = field_spec_channels(self.field_spec)
        self.dtype = dtype
        self.cache_mode = cache_mode
        self.transform = transform
        self.files = filepaths
        self.context_frames = int(context_frames)
        self.predict_frames = int(predict_frames)
        self.device = torch.device(device) if device is not None else None
        self.traj_cache_capacity = max(1, int(traj_cache_capacity))
        # sample every `pair_stride`-th start position within a trajectory instead of
        # every consecutive one -- needed for datasets whose trajectories are very long
        # (gray_scott T=1001, planetswe T=1008: even a single trajectory yields ~990-1000
        # windows, which file_limit/traj_limit alone can't tame since neither controls
        # samples *within* one trajectory). Default 1 preserves exact existing behavior.
        self.pair_stride = max(1, int(pair_stride))
        # 2026-08-19: traj_limit's selection -- see below -- needs its own seed, kept
        # independent of create_param_dataloaders' file-selection `seed` param even
        # though both currently default to 42 and are usually passed the same value by
        # the caller, so this class stays correct/reproducible on its own if ever
        # constructed directly (as several call sites/tests already do).
        self.traj_seed = int(traj_seed)
        # optional per-file params (e.g. rayleigh/prandtl, or reynolds/schmidt,
        # or L/zeta/alpha -- whatever field_spec's dataset uses) parallel to `filepaths`
        self.file_params = file_params
        self.return_params = bool(return_params)

        self._pairs: List[PairIndex] = []
        self._layouts: List[Dict[str, Any]] = []
        if file_limit is not None:
            self.files = self.files[:file_limit]
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                layout = self._inspect_layout(f, path)
                self._layouts.append(layout)
                n_traj = layout["n_traj"]
                T = layout["time_steps"]
                if traj_limit is not None and traj_limit < n_traj:
                    # Seeded random sample, not a positional head-slice -- see the
                    # 2026-08-19 comment above self.traj_seed for why. Sorted purely
                    # for deterministic/readable _pairs ordering; the *set* of indices
                    # is what matters, not their order.
                    traj_indices = sorted(random.Random(self.traj_seed + fi).sample(range(n_traj), traj_limit))
                else:
                    traj_indices = list(range(n_traj))

            needed = self.context_frames + self.predict_frames

            max_start = -1
            if T >= needed:
                max_start = T - needed

            for tj in traj_indices:
                for s in range(0, max_start + 1, self.pair_stride):
                    self._pairs.append(PairIndex(fi, tj, s))

        self._open_file_idx: Optional[int] = None
        self._h5: Optional[h5py.File] = None
        self._dsets: Optional[List[Any]] = None
        self._traj_cache: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
        self._traj_cache_hits = 0
        self._traj_cache_misses = 0

    def _inspect_layout(self, f: h5py.File, path: str) -> Dict[str, Any]:
        for spec in self.field_spec:
            if spec["key"] not in f:
                raise KeyError(f"Missing required key '{spec['key']}' in {path}")

        # dimensions/x + dimensions/y are OPTIONAL: only used as a sanity check in
        # _infer_xy_axes_from_shape (skipped entirely when x_len/y_len are None -- axis
        # position, not length, is what's actually load-bearing there). Not every Well
        # dataset uses Cartesian axis names -- planetswe is on a lat-lon sphere grid
        # (dimensions/theta, dimensions/phi), no dimensions/x or dimensions/y at all.
        if "dimensions/x" in f and "dimensions/y" in f:
            x_len = int(f["dimensions/x"].shape[0])
            y_len = int(f["dimensions/y"].shape[0])
        else:
            x_len = y_len = None

        n_traj = None
        time_steps = None
        field_layouts = []
        for spec in self.field_spec:
            shape = tuple(f[spec["key"]].shape)
            # (N, T, *spatial*, *components*) -- components: none for scalar (n_components=1),
            # one size-2 axis for vector (n_components=2), two size-2 axes for a tensor (n_components=4).
            n_comp_axes = 0 if spec["n_components"] == 1 else (1 if spec["n_components"] == 2 else 2)
            expected_ndim = 2 + 2 + n_comp_axes  # N, T, x, y, [components...]
            if len(shape) != expected_ndim:
                raise ValueError(
                    f"Expected dataset '{spec['key']}' (n_components={spec['n_components']}) to be "
                    f"{expected_ndim}D (N,T,*,*{',2'*n_comp_axes}). Got shape={shape} in {path}"
                )

            if n_traj is None:
                n_traj, time_steps = int(shape[0]), int(shape[1])
            elif shape[0] != n_traj or shape[1] != time_steps:
                raise ValueError(
                    f"Trajectory/time dimensions mismatch for '{spec['key']}': "
                    f"expected (N={n_traj}, T={time_steps}), got shape={shape} in {path}"
                )

            traj_shape = shape[1:]
            x_axis, y_axis = _infer_xy_axes_from_shape(traj_shape, x_len, y_len, spec["key"])
            field_layouts.append({"key": spec["key"], "shape": shape, "x_axis": x_axis, "y_axis": y_axis,
                                   "n_components": spec["n_components"]})

        return {
            "path": path,
            "n_traj": n_traj,
            "time_steps": time_steps,
            "x_len": x_len,
            "y_len": y_len,
            "fields": field_layouts,
        }

    def __len__(self) -> int:
        return len(self._pairs)

    def _ensure_open(self, file_idx: int):
        if self._h5 is not None and self._open_file_idx == file_idx:
            return

        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

        path = self.files[file_idx]
        self._h5 = h5py.File(path, "r")
        self._dsets = [self._h5[spec["key"]] for spec in self.field_spec]
        self._open_file_idx = file_idx

    def _field_to_t_c_yx(self, arr: np.ndarray, x_axis: int, y_axis: int, n_components: int) -> np.ndarray:
        """arr: raw per-trajectory field, shape (T, ..., x_axis, y_axis, ...) with 0 or more
        trailing 'component' axes beyond x/y (vector: one axis of size 2; tensor: two axes of
        size 2 each). Returns (T, n_components, Y, X); any component axes are flattened in
        their original relative order (tensor case: row-major, e.g. D_00,D_01,D_10,D_11) --
        generalizes the old separate scalar/vector-only logic to any field rank.
        """
        remaining_axes = [ax for ax in range(arr.ndim) if ax not in (0, x_axis, y_axis)]
        arr = np.moveaxis(arr, [y_axis, x_axis] + remaining_axes, [1, 2] + list(range(3, 3 + len(remaining_axes))))
        T, Y, X = arr.shape[0], arr.shape[1], arr.shape[2]
        arr = arr.reshape(T, Y, X, n_components)
        return np.moveaxis(arr, 3, 1)  # -> (T, n_components, Y, X)

    def _get_traj_tensor(self, file_idx: int, traj_idx: int) -> torch.Tensor:
        """Load one trajectory and convert to (T, n_channels, H, W), channels in field_spec order."""
        assert self._dsets is not None

        layout = self._layouts[file_idx]
        arrays = []
        for dset, field_layout in zip(self._dsets, layout["fields"]):
            raw = np.asarray(dset[traj_idx, ...], dtype=np.float32)
            arrays.append(self._field_to_t_c_yx(
                raw, x_axis=field_layout["x_axis"], y_axis=field_layout["y_axis"],
                n_components=field_layout["n_components"],
            ))

        fields = np.concatenate(arrays, axis=1)  # [T, n_channels, H, W]
        x = torch.from_numpy(np.ascontiguousarray(fields)).to(self.dtype)
        return x

    def _get_cached_traj_tensor(self, file_idx: int, traj_idx: int) -> torch.Tensor:
        key = (file_idx, traj_idx)
        cached = self._traj_cache.get(key)
        if cached is not None:
            self._traj_cache.move_to_end(key)
            self._traj_cache_hits += 1
            return cached

        # Miss: ensure source file is open, load tensor, then cache it.
        self._ensure_open(file_idx)
        traj = self._get_traj_tensor(file_idx, traj_idx)
        self._traj_cache[key] = traj
        self._traj_cache.move_to_end(key)
        self._traj_cache_misses += 1

        while len(self._traj_cache) > self.traj_cache_capacity:
            self._traj_cache.popitem(last=False)

        return traj

    def cache_stats(self) -> Dict[str, float]:
        total = self._traj_cache_hits + self._traj_cache_misses
        hit_rate = (self._traj_cache_hits / total) if total > 0 else 0.0
        return {
            "hits": float(self._traj_cache_hits),
            "misses": float(self._traj_cache_misses),
            "hit_rate": float(hit_rate),
            "size": float(len(self._traj_cache)),
            "capacity": float(self.traj_cache_capacity),
        }

    def __getitem__(self, idx: int):
        p = self._pairs[idx]

        # multi-trajectory LRU cache path
        if self.cache_mode == "traj":
            traj = self._get_cached_traj_tensor(p.file_idx, p.traj_idx)
        else:
            # lazy load single trajectory into memory for this access
            self._ensure_open(p.file_idx)
            traj = self._get_traj_tensor(p.file_idx, p.traj_idx)

        s = p.t
        C = self.context_frames
        P = self.predict_frames

        ctx = traj[s : s + C]      # (C, n_channels, H, W)
        tgt = traj[s + C : s + C + P]  # (P, n_channels, H, W)

        if self.transform is not None:
            ctx = torch.stack([self.transform(f) for f in ctx], dim=0)
            tgt = torch.stack([self.transform(f) for f in tgt], dim=0)

        # Loud shape/channel checks to prevent channel-semantic regressions.
        assert ctx.ndim == 4, f"ctx must be 4D [T, C, H, W]; got shape={tuple(ctx.shape)}"
        assert tgt.ndim == 4, f"tgt must be 4D [T, C, H, W]; got shape={tuple(tgt.shape)}"
        assert ctx.shape[1] == self.n_channels, (
            f"ctx channel dimension must be {self.n_channels} channels per field_spec. "
            f"Got shape={tuple(ctx.shape)}"
        )
        assert tgt.shape[1] == self.n_channels, (
            f"tgt channel dimension must be {self.n_channels} channels per field_spec. "
            f"Got shape={tuple(tgt.shape)}"
        )

        if self.return_params:
            param = None
            if self.file_params is not None:
                parsed = self.file_params[p.file_idx]
                if parsed is not None:
                    param = torch.tensor(parsed, dtype=torch.float32)
                else:
                    param = None
            return ctx, tgt, param

        return ctx, tgt

    def __del__(self):
        # best-effort cleanup
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class H5VelocityFramePairs(Dataset):
    """Removed legacy dataset kept only as a loud failure path."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "H5VelocityFramePairs has been removed for Rayleigh-Benard training. "
            + STACK_FRAMES_REMOVED_ERROR
        )


def read_params_from_h5(path: str) -> Optional[tuple]:
    """Read simulation parameter values directly from the HDF5 file itself --
    every Well-format file self-describes its own varying parameters via
    attrs['simulation_parameters'] (an ordered list of names) and a matching
    scalars/<name> dataset per name. Dataset-agnostic by construction: this
    reads rayleigh_benard's (Rayleigh, Prandtl), shear_flow's (Reynolds,
    Schmidt), and active_matter's (L, zeta, alpha) identically, without a
    per-dataset filename regex -- replaces the old Rayleigh/Prandtl-specific
    filename parser, which had no way to generalize to a 3-parameter dataset
    or different parameter names. Verified to return numerically identical
    values to the old filename-regex approach for rayleigh_benard.
    """
    try:
        with h5py.File(path, "r") as f:
            names = f.attrs.get("simulation_parameters")
            if names is None:
                return None
            values = []
            for name in names:
                key = f"scalars/{name}"
                if key not in f:
                    return None
                values.append(float(f[key][()]))
            return tuple(values)
    except Exception:
        return None


def select_representative_files(files: List[str], k: int, key_fn) -> List[str]:
    """Deterministically pick up to k files spanning the range of key_fn(file), instead
    of clustering wherever directory/alphabetical order happens to put the first k.

    Concretely: sorted(os.listdir(...))[:k] (what a plain file_limit does) picked all 7
    of rayleigh_benard_uniform's Rayleigh=1e10 combos plus one Rayleigh=1e6 file for a
    file_limit=8 -- Rayleigh spans 1e6-1e10 in the actual data, so that's a 3-decade gap
    in the one parameter that matters, not a representative sample of it. Never random
    either -- same input always picks the same files.

    Groups files by key first (so e.g. planetswe's 3 seeds per initial condition collapse
    to one representative before spacing, rather than burning 3 of k slots on one IC),
    then evenly spaces k picks across the sorted unique keys.
    """
    groups: Dict[Any, List[str]] = defaultdict(list)
    for f in files:
        groups[key_fn(f)].append(f)
    keys_sorted = sorted(groups.keys())
    if k >= len(keys_sorted):
        picked_keys = keys_sorted
    else:
        idxs = sorted({round(i) for i in np.linspace(0, len(keys_sorted) - 1, k)})
        picked_keys = [keys_sorted[i] for i in idxs]
    return [sorted(groups[key])[0] for key in picked_keys]


def _round_sig(x: float, sig: int = 6) -> float:
    """Round to `sig` significant figures (not fixed decimal places -- params
    here span both huge (Rayleigh ~1e10) and small (Prandtl 0.1) magnitudes,
    so a fixed decimal-place round wouldn't work uniformly)."""
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _params_match_key(params: tuple, sig: int = 6) -> tuple:
    """Matching key for comparing a requested --param-subset combo against a
    file's actual params. Values here are stored as float32 in the HDF5 file
    but a human-written JSON combo (e.g. 0.1) parses as an exact float64 --
    float32(0.1) upcast to float64 is 0.10000000149011612, which does NOT
    bit-match Python's 0.1 literal, so exact tuple equality silently drops
    real matches for any decimal that isn't exactly binary-representable
    (0.1, 0.2, 0.3, ... -- confirmed concretely: this dropped 2 of 8 requested
    shear_flow combos before this fix). Rounding both sides to 6 significant
    figures before comparing absorbs that float32-precision noise (~1e-7
    relative) while still easily distinguishing any two genuinely different
    parameter values in this project's datasets (which differ by orders of
    magnitude or at least whole small-integer multiples, never anywhere near
    1e-6 relative). Only used for matching -- the actual params fed to the
    model as conditioning input stay at full stored precision.
    """
    return tuple(_round_sig(v, sig) for v in params)


def _group_files_by_params(filepaths: List[str]) -> Dict[tuple, List[str]]:
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for fp in filepaths:
        parsed = read_params_from_h5(fp)
        if parsed is None:
            continue
        groups[parsed].append(fp)
    return groups


class StreamingBlockBatchSampler(Sampler[List[int]]):
    """Streaming lane-based temporal-block sampler for RB training.

    Each active lane corresponds to one temporal block from one (file_idx, traj_idx).
    Batches are formed by taking one sample from each active lane; lanes persist
    across consecutive batches until exhausted, then are replaced by new blocks.
    """

    def __init__(
        self,
        pairs: List[PairIndex],
        batch_size: int,
        block_size: int = 16,
        seed: int = 42,
        drop_last: bool = False,
        prefer_file_diversity: bool = True,
        rotate_blocks: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.pairs = pairs
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.prefer_file_diversity = bool(prefer_file_diversity)
        self.rotate_blocks = bool(rotate_blocks)
        self.epoch = 0

        self.blocks: List[Dict[str, Any]] = []
        current_indices: List[int] = []
        current_key: Optional[Tuple[int, int]] = None

        for idx, p in enumerate(self.pairs):
            key = (p.file_idx, p.traj_idx)
            if current_key is None:
                current_key = key

            if key != current_key or len(current_indices) >= self.block_size:
                self.blocks.append(
                    {
                        "indices": current_indices,
                        "file_idx": current_key[0],
                        "traj_idx": current_key[1],
                    }
                )
                current_indices = []
                current_key = key

            current_indices.append(idx)

        if current_indices:
            assert current_key is not None
            self.blocks.append(
                {
                    "indices": current_indices,
                    "file_idx": current_key[0],
                    "traj_idx": current_key[1],
                }
            )

        self.total_samples = sum(len(b["indices"]) for b in self.blocks)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _pop_preferred_block(
        self,
        pool_blocks: List[Dict[str, Any]],
        active_lanes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not pool_blocks:
            return None

        active_files = {b["file_idx"] for b in active_lanes}
        active_trajs = {(b["file_idx"], b["traj_idx"]) for b in active_lanes}

        # First pass: new trajectory and (optionally) new file.
        for i, b in enumerate(pool_blocks):
            traj_key = (b["file_idx"], b["traj_idx"])
            if traj_key in active_trajs:
                continue
            if self.prefer_file_diversity and b["file_idx"] in active_files:
                continue
            return pool_blocks.pop(i)

        # Second pass: new trajectory regardless of file.
        for i, b in enumerate(pool_blocks):
            traj_key = (b["file_idx"], b["traj_idx"])
            if traj_key not in active_trajs:
                return pool_blocks.pop(i)

        # Fallback.
        return pool_blocks.pop(0)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Build per-iteration mutable block states.
        pool_blocks: List[Dict[str, Any]] = []
        for b in self.blocks:
            idxs = list(b["indices"])
            if self.rotate_blocks and len(idxs) > 1:
                offset = rng.randrange(len(idxs))
                idxs = idxs[offset:] + idxs[:offset]
            pool_blocks.append(
                {
                    "indices": idxs,
                    "pos": 0,
                    "file_idx": b["file_idx"],
                    "traj_idx": b["traj_idx"],
                }
            )

        rng.shuffle(pool_blocks)

        active_lanes: List[Dict[str, Any]] = []
        while len(active_lanes) < self.batch_size and pool_blocks:
            nxt = self._pop_preferred_block(pool_blocks, active_lanes)
            if nxt is None:
                break
            active_lanes.append(nxt)

        while active_lanes:
            batch: List[int] = []
            for lane in active_lanes:
                if lane["pos"] < len(lane["indices"]):
                    batch.append(lane["indices"][lane["pos"]])
                    lane["pos"] += 1

            if len(batch) == self.batch_size:
                yield batch
            elif batch and not self.drop_last:
                yield batch

            # Drop exhausted lanes.
            active_lanes = [lane for lane in active_lanes if lane["pos"] < len(lane["indices"])]

            # Refill lanes with preferred diversity.
            while len(active_lanes) < self.batch_size and pool_blocks:
                nxt = self._pop_preferred_block(pool_blocks, active_lanes)
                if nxt is None:
                    break
                active_lanes.append(nxt)

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def debug_sampler_diversity(
    dataset: H5RayleighBenardFields,
    batch_sampler: StreamingBlockBatchSampler,
    num_batches: int = 10,
):
    """Inspect batch diversity from sampler indices without loading HDF5 tensors."""
    print(f"Sampler diversity debug: inspecting up to {num_batches} batches")
    prev_trajs: Optional[set] = None
    for bi, batch_indices in enumerate(batch_sampler):
        if bi >= num_batches:
            break
        batch_pairs = [dataset._pairs[i] for i in batch_indices]
        files = [p.file_idx for p in batch_pairs]
        trajs = [(p.file_idx, p.traj_idx) for p in batch_pairs]
        ts = [p.t for p in batch_pairs]
        traj_set = set(trajs)
        persisted = len(traj_set & prev_trajs) if prev_trajs is not None else 0

        file_hist: Dict[int, int] = defaultdict(int)
        for fidx in files:
            file_hist[fidx] += 1

        print(
            f"  batch[{bi}] size={len(batch_indices)} "
            f"unique_files={len(set(files))} "
            f"unique_trajs={len(set(trajs))} "
            f"persist_from_prev={persisted} "
            f"t_range=({min(ts) if ts else None},{max(ts) if ts else None}) "
            f"file_hist={dict(sorted(file_hist.items()))}"
        )
        prev_trajs = traj_set


def debug_cache_effectiveness(dataset: H5RayleighBenardFields):
    stats = dataset.cache_stats()
    print(
        "Trajectory cache stats: "
        f"hits={int(stats['hits'])} misses={int(stats['misses'])} "
        f"hit_rate={stats['hit_rate']:.3f} "
        f"size={int(stats['size'])}/{int(stats['capacity'])}"
    )


def create_param_dataloaders(
    base_dir: str,
    param_choice: Optional[tuple] = None,
    param_choices: Optional[List[tuple]] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    train_file_limit: Optional[int] = None,
    val_file_limit: Optional[int] = None,
    traj_limit: Optional[int] = None,
    pair_stride: int = 1,
    context_frames: int = 1,
    predict_frames: int = 1,
    train_subdir: str = "train",
    valid_subdir: str = "valid",
    device: Optional[str] = None,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    use_all_params: bool = True,
    return_params: bool = True,
    debug_sanity: bool = False,
    debug_sampler_batches: int = 0,
    cache_mode: str = "traj",
    traj_cache_capacity: Optional[int] = None,
    block_size: int = 32,
    field_spec: Optional[List[Dict[str, Any]]] = None,
    file_select_fn=None,
    **legacy_kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[tuple]]:
    if "stack_frames" in legacy_kwargs:
        raise ValueError(STACK_FRAMES_REMOVED_ERROR)
    if "block_shuffle_train" in legacy_kwargs:
        raise ValueError(
            "block_shuffle_train is no longer supported. "
            "Training now always uses StreamingBlockBatchSampler."
        )
    if shuffle_train is not True:
        print("Note: shuffle_train is ignored; training now always uses StreamingBlockBatchSampler.")

    if traj_cache_capacity is None:
        traj_cache_capacity = max(2 * int(batch_size), 64)

    rng = random.Random(seed)
    train_dir = os.path.join(base_dir, train_subdir)
    valid_dir = os.path.join(base_dir, valid_subdir)

    def _list_h5(d):
        if not os.path.isdir(d):
            return []
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.hdf5')]

    train_files_all = _list_h5(train_dir)
    valid_files_all = _list_h5(valid_dir)

    # See select_representative_files: pick k files spanning the varying parameter's
    # range instead of train_file_limit/val_file_limit's plain head-slice of whatever
    # order the directory listing happens to be in.
    if file_select_fn is not None:
        if train_file_limit is not None and train_file_limit < len(train_files_all):
            train_files_all = file_select_fn(train_files_all, train_file_limit)
        if val_file_limit is not None and val_file_limit < len(valid_files_all):
            valid_files_all = file_select_fn(valid_files_all, val_file_limit)

    train_groups = _group_files_by_params(train_files_all)
    valid_groups = _group_files_by_params(valid_files_all)

    # Restrict to an explicit list of (rayleigh, prandtl) combos -- e.g. for a quick
    # speed/timing check without paying for the full ~300GB dataset. Takes priority
    # over use_all_params/param_choice; per-file params are still tracked since more
    # than one distinct combo may be present.
    chosen = None
    if param_choices:
        wanted_keys = {_params_match_key(tuple(c)) for c in param_choices}
        file_params_cache = {fp: read_params_from_h5(fp) for fp in train_files_all + valid_files_all}

        def _matches(fp):
            p = file_params_cache[fp]
            return p is not None and _params_match_key(p) in wanted_keys

        train_files = [fp for fp in train_files_all if _matches(fp)]
        val_files = [fp for fp in valid_files_all if _matches(fp)]
        found_keys = {_params_match_key(file_params_cache[fp]) for fp in train_files + val_files}
        missing = wanted_keys - found_keys
        if missing:
            print(f"[create_param_dataloaders] warning: no files found for combos {missing}")
        print(f"[create_param_dataloaders] param_choices={sorted(wanted_keys)} -> "
              f"{len(train_files)} train file(s), {len(val_files)} val file(s)")

        train_file_params = [file_params_cache[fp] for fp in train_files]
        val_file_params = [file_params_cache[fp] for fp in val_files]
    # prefer parameter pairs that exist in both train and valid when not using the full split
    elif not use_all_params:
        common_keys = set(train_groups.keys()) & set(valid_groups.keys())
        print(common_keys)
        chosen = param_choice or rng.choice(list(common_keys))
        train_files = train_groups[chosen]
        val_files = valid_groups[chosen]
        train_file_params = None
        val_file_params = None
    else:
        # use all train/valid files and compute per-file params
        train_files = train_files_all
        val_files = valid_files_all
        def _params_for_list(lst):
            out = []
            for fp in lst:
                parsed = read_params_from_h5(fp)
                out.append(parsed)
            return out

        train_file_params = _params_for_list(train_files)
        val_file_params = _params_for_list(val_files)

    trainset = H5RayleighBenardFields(
        train_files,
        field_spec=field_spec,
        file_limit=train_file_limit,
        traj_limit=traj_limit,
        pair_stride=pair_stride,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=train_file_params,
        return_params=return_params or use_all_params,
        cache_mode=cache_mode,
        traj_cache_capacity=traj_cache_capacity,
        traj_seed=seed,
    )

    valset = H5RayleighBenardFields(
        val_files,
        field_spec=field_spec,
        file_limit=val_file_limit,
        traj_limit=traj_limit,
        pair_stride=pair_stride,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=val_file_params,
        return_params=return_params or use_all_params,
        cache_mode=cache_mode,
        traj_cache_capacity=traj_cache_capacity,
        traj_seed=seed,
    )

    train_prefetch = 2 if num_workers and num_workers > 0 else None
    train_batch_sampler = StreamingBlockBatchSampler(
        trainset._pairs,
        batch_size=batch_size,
        block_size=block_size,
        seed=seed,
        drop_last=False,
        prefer_file_diversity=True,
        rotate_blocks=True,
    )
    train_loader = DataLoader(
        trainset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=train_prefetch,
        persistent_workers=(num_workers > 0),
    )

    # For validation we usually don't shuffle; allow opt-in and separate generator
    val_gen = torch.Generator()
    val_gen.manual_seed(seed + 1)
    val_prefetch = 2 if num_workers and num_workers > 0 else None
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        generator=val_gen if shuffle_val else None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=val_prefetch,
    )

    if debug_sanity and len(train_files) > 0:
        inspect_rayleigh_benard_h5(train_files[0])
        debug_rb_batch_sanity(train_loader)
        debug_cache_effectiveness(trainset)
    if debug_sampler_batches > 0:
        debug_sampler_diversity(trainset, train_batch_sampler, num_batches=debug_sampler_batches)

    return train_loader, val_loader, chosen


def debug_rb_batch_sanity(loader: DataLoader):
    """Print one-batch RB sanity diagnostics for shapes and channel stats."""
    batch = next(iter(loader))
    if len(batch) == 3:
        xb, yb, _ = batch
    else:
        xb, yb = batch

    print(f"Sanity batch xb.shape={tuple(xb.shape)} yb.shape={tuple(yb.shape)}")

    assert xb.ndim == 5, f"xb must be [B, T, C, H, W], got {tuple(xb.shape)}"
    assert yb.ndim == 5, f"yb must be [B, T, C, H, W], got {tuple(yb.shape)}"
    assert xb.shape[2] == 4, (
        "xb channel dimension must be 4 physical channels [pressure, buoyancy, velocity_x, velocity_y], "
        f"got {tuple(xb.shape)}"
    )
    assert yb.shape[2] == 4, (
        "yb channel dimension must be 4 physical channels [pressure, buoyancy, velocity_x, velocity_y], "
        f"got {tuple(yb.shape)}"
    )

    x_fields = FieldState.from_tensor(xb)
    y_fields = FieldState.from_tensor(yb)

    print("Input per-channel mean/std:")
    print(
        f"  0:pressure mean={float(x_fields.pressure.mean()):.6g} std={float(x_fields.pressure.std()):.6g}\n"
        f"  1:buoyancy mean={float(x_fields.buoyancy.mean()):.6g} std={float(x_fields.buoyancy.std()):.6g}\n"
        f"  2:velocity_x mean={float(x_fields.velocity_x.mean()):.6g} std={float(x_fields.velocity_x.std()):.6g}\n"
        f"  3:velocity_y mean={float(x_fields.velocity_y.mean()):.6g} std={float(x_fields.velocity_y.std()):.6g}"
    )
    print("Target per-channel mean/std:")
    print(
        f"  0:pressure mean={float(y_fields.pressure.mean()):.6g} std={float(y_fields.pressure.std()):.6g}\n"
        f"  1:buoyancy mean={float(y_fields.buoyancy.mean()):.6g} std={float(y_fields.buoyancy.std()):.6g}\n"
        f"  2:velocity_x mean={float(y_fields.velocity_x.mean()):.6g} std={float(y_fields.velocity_x.std()):.6g}\n"
        f"  3:velocity_y mean={float(y_fields.velocity_y.mean()):.6g} std={float(y_fields.velocity_y.std()):.6g}"
    )


def move_batch_to_device(batch, device: Optional[str]):
    """Move a (ctx, tgt) or (ctx, tgt, params) batch to `device` from the main process.

    Returns the moved batch. If `device` is None or not available,
    returns the input batch unchanged.
    """
    dev = torch.device(device) if device is not None else None
    if dev is None:
        return batch

    if len(batch) == 3:
        ctx, tgt, params = batch
        params_out = None if params is None else params.to(dev, non_blocking=True)
        return ctx.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True), params_out

    ctx, tgt = batch
    return ctx.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)


if __name__ == "__main__":
    # Simple, no-CLI smoke test for running inside the container
    BASE_DIR = "/app/data/datasets/rayleigh_benard/data"
    BATCH_SIZE = 2
    # context : predict (e.g., 3:1)
    CONTEXT_FRAMES = 3
    PREDICT_FRAMES = 1

    print("Creating dataloaders for a random chosen parameter pair...")
    train_loader, val_loader, chosen = create_param_dataloaders(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        context_frames=CONTEXT_FRAMES,
        predict_frames=PREDICT_FRAMES,
        debug_sanity=True,
    )
    print(f"Chosen parameter pair: {chosen}")

    # Move the batch to the chosen device in the main process to avoid
    # initializing CUDA inside forked DataLoader worker subprocesses.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch = next(iter(train_loader))
    moved = move_batch_to_device(batch, str(device))
    if len(moved) == 3:
        ctx_batch, tgt_batch, p_batch = moved
        print(f"ctx shape: {ctx_batch.shape}, tgt shape: {tgt_batch.shape}, params shape: {None if p_batch is None else p_batch.shape}")
    else:
        ctx_batch, tgt_batch = moved
        print(f"ctx shape: {ctx_batch.shape}, tgt shape: {tgt_batch.shape}")
