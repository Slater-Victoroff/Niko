import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
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
    """Infer x/y axes in a trajectory tensor based on dimensions/x and dimensions/y lengths.

    The time axis must be axis 0. This helper only infers spatial axes for axis positions >= 1.
    """
    if len(traj_shape) < 3:
        raise ValueError(f"{context_label} must have at least 3 dims (T + 2 spatial). Got shape={traj_shape}")

    if x_len is None or y_len is None:
        raise ValueError(
            f"Cannot infer axis order for {context_label}: missing dimensions/x or dimensions/y in HDF5 file."
        )

    x_axes = [ax for ax in range(1, len(traj_shape)) if traj_shape[ax] == x_len]
    y_axes = [ax for ax in range(1, len(traj_shape)) if traj_shape[ax] == y_len]

    if len(x_axes) != 1 or len(y_axes) != 1:
        raise ValueError(
            f"Could not uniquely infer x/y axes for {context_label} with shape={traj_shape}, "
            f"x_len={x_len}, y_len={y_len}, x_axes={x_axes}, y_axes={y_axes}."
        )

    x_axis = x_axes[0]
    y_axis = y_axes[0]
    if x_axis == y_axis:
        raise ValueError(
            f"Inferred same axis for x and y in {context_label}: axis={x_axis}, shape={traj_shape}."
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
    def __init__(
        self,
        filepaths: List[str],
        pressure_key: str = "t0_fields/pressure",
        buoyancy_key: str = "t0_fields/buoyancy",
        velocity_key: str = "t1_fields/velocity",
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
        **legacy_kwargs,
    ):
        if "stack_frames" in legacy_kwargs:
            raise ValueError(STACK_FRAMES_REMOVED_ERROR)

        self.pressure_key = pressure_key
        self.buoyancy_key = buoyancy_key
        self.velocity_key = velocity_key
        self.dtype = dtype
        self.cache_mode = cache_mode
        self.transform = transform
        self.files = filepaths
        self.context_frames = int(context_frames)
        self.predict_frames = int(predict_frames)
        self.device = torch.device(device) if device is not None else None
        self.traj_cache_capacity = max(1, int(traj_cache_capacity))
        # optional per-file params (rayleigh, prandtl) parallel to `filepaths`
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
                if traj_limit is not None:
                    n_traj = min(n_traj, traj_limit)

            needed = self.context_frames + self.predict_frames

            max_start = -1
            if T >= needed:
                max_start = T - needed

            for tj in range(n_traj):
                for s in range(max_start + 1):
                    self._pairs.append(PairIndex(fi, tj, s))

        self._open_file_idx: Optional[int] = None
        self._h5: Optional[h5py.File] = None
        self._pressure_dset = None
        self._buoyancy_dset = None
        self._velocity_dset = None
        self._traj_cache: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
        self._traj_cache_hits = 0
        self._traj_cache_misses = 0

    def _inspect_layout(self, f: h5py.File, path: str) -> Dict[str, Any]:
        for key in (self.pressure_key, self.buoyancy_key, self.velocity_key, "dimensions/x", "dimensions/y"):
            if key not in f:
                raise KeyError(f"Missing required key '{key}' in {path}")

        pressure_shape = tuple(f[self.pressure_key].shape)
        buoyancy_shape = tuple(f[self.buoyancy_key].shape)
        velocity_shape = tuple(f[self.velocity_key].shape)

        if len(pressure_shape) != 4:
            raise ValueError(
                f"Expected pressure dataset '{self.pressure_key}' to be 4D (N,T,*,*). "
                f"Got shape={pressure_shape} in {path}"
            )
        if len(buoyancy_shape) != 4:
            raise ValueError(
                f"Expected buoyancy dataset '{self.buoyancy_key}' to be 4D (N,T,*,*). "
                f"Got shape={buoyancy_shape} in {path}"
            )
        if len(velocity_shape) != 5:
            raise ValueError(
                f"Expected velocity dataset '{self.velocity_key}' to be 5D (N,T,*,*,2). "
                f"Got shape={velocity_shape} in {path}"
            )

        if pressure_shape[:2] != buoyancy_shape[:2] or pressure_shape[:2] != velocity_shape[:2]:
            raise ValueError(
                "Trajectory/time dimensions mismatch among RB fields: "
                f"pressure={pressure_shape}, buoyancy={buoyancy_shape}, velocity={velocity_shape} in {path}"
            )

        n_traj = int(pressure_shape[0])
        time_steps = int(pressure_shape[1])
        x_len = int(f["dimensions/x"].shape[0])
        y_len = int(f["dimensions/y"].shape[0])

        p_traj_shape = pressure_shape[1:]
        b_traj_shape = buoyancy_shape[1:]
        v_traj_shape = velocity_shape[1:]

        p_x_axis, p_y_axis = _infer_xy_axes_from_shape(p_traj_shape, x_len, y_len, "pressure trajectory")
        b_x_axis, b_y_axis = _infer_xy_axes_from_shape(b_traj_shape, x_len, y_len, "buoyancy trajectory")
        v_x_axis, v_y_axis = _infer_xy_axes_from_shape(v_traj_shape, x_len, y_len, "velocity trajectory")

        v_comp_axes = [ax for ax in range(1, len(v_traj_shape)) if v_traj_shape[ax] == 2]
        if len(v_comp_axes) != 1:
            raise ValueError(
                f"Could not uniquely infer velocity component axis (size 2) from shape {v_traj_shape} in {path}."
            )
        v_comp_axis = v_comp_axes[0]

        return {
            "path": path,
            "n_traj": n_traj,
            "time_steps": time_steps,
            "x_len": x_len,
            "y_len": y_len,
            "pressure_shape": pressure_shape,
            "buoyancy_shape": buoyancy_shape,
            "velocity_shape": velocity_shape,
            "pressure_axes": {"x": p_x_axis, "y": p_y_axis},
            "buoyancy_axes": {"x": b_x_axis, "y": b_y_axis},
            "velocity_axes": {"comp": v_comp_axis, "x": v_x_axis, "y": v_y_axis},
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
        self._pressure_dset = self._h5[self.pressure_key]
        self._buoyancy_dset = self._h5[self.buoyancy_key]
        self._velocity_dset = self._h5[self.velocity_key]
        self._open_file_idx = file_idx

    def _scalar_to_tyx(self, arr: np.ndarray, x_axis: int, y_axis: int) -> np.ndarray:
        # Input is trajectory-level scalar with time axis=0.
        # Output must be (T, Y, X).
        return np.moveaxis(arr, [y_axis, x_axis], [1, 2])

    def _velocity_to_t2yx(self, arr: np.ndarray, comp_axis: int, x_axis: int, y_axis: int) -> np.ndarray:
        # Input is trajectory-level velocity with time axis=0.
        # Output must be (T, 2, Y, X).
        return np.transpose(arr, (0, comp_axis, y_axis, x_axis))

    def _get_traj_tensor(self, file_idx: int, traj_idx: int) -> torch.Tensor:
        """Load one trajectory and convert to (T, 4, H, W) channel order [p,b,u,v]."""
        assert self._pressure_dset is not None
        assert self._buoyancy_dset is not None
        assert self._velocity_dset is not None

        layout = self._layouts[file_idx]

        p = np.asarray(self._pressure_dset[traj_idx, ...], dtype=np.float32)
        b = np.asarray(self._buoyancy_dset[traj_idx, ...], dtype=np.float32)
        v = np.asarray(self._velocity_dset[traj_idx, ...], dtype=np.float32)

        p_tyx = self._scalar_to_tyx(
            p,
            x_axis=layout["pressure_axes"]["x"],
            y_axis=layout["pressure_axes"]["y"],
        )
        b_tyx = self._scalar_to_tyx(
            b,
            x_axis=layout["buoyancy_axes"]["x"],
            y_axis=layout["buoyancy_axes"]["y"],
        )
        v_t2yx = self._velocity_to_t2yx(
            v,
            comp_axis=layout["velocity_axes"]["comp"],
            x_axis=layout["velocity_axes"]["x"],
            y_axis=layout["velocity_axes"]["y"],
        )

        # [T, 4, H, W] with channel order [pressure, buoyancy, velocity_x, velocity_y]
        fields = np.concatenate([p_tyx[:, None, :, :], b_tyx[:, None, :, :], v_t2yx], axis=1)
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

        ctx = traj[s : s + C]      # (C, 4, H, W)
        tgt = traj[s + C : s + C + P]  # (P, 4, H, W)

        if self.transform is not None:
            ctx = torch.stack([self.transform(f) for f in ctx], dim=0)
            tgt = torch.stack([self.transform(f) for f in tgt], dim=0)

        # Loud shape/channel checks to prevent channel-semantic regressions.
        assert ctx.ndim == 4, f"ctx must be 4D [T, C, H, W]; got shape={tuple(ctx.shape)}"
        assert tgt.ndim == 4, f"tgt must be 4D [T, C, H, W]; got shape={tuple(tgt.shape)}"
        assert ctx.shape[1] == 4, (
            "ctx channel dimension must be 4 physical channels in order "
            "[pressure, buoyancy, velocity_x, velocity_y]. "
            f"Got shape={tuple(ctx.shape)}"
        )
        assert tgt.shape[1] == 4, (
            "tgt channel dimension must be 4 physical channels in order "
            "[pressure, buoyancy, velocity_x, velocity_y]. "
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


def _parse_params_from_filename(filename: str):
    """Parse Rayleigh and Prandtl values from a filename.

    Returns (rayleigh: float, prandtl: float) or None if not found.
    """
    # match numbers like: 1e10, 1e-1, 5e-1, 10, 2.5, etc., and ensure we stop before the .hdf5 extension
    num = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?"
    pattern = rf"Rayleigh_({num})_Prandtl_({num})\.hdf5$"
    m = re.search(pattern, filename, flags=re.I)
    if not m:
        return None
    try:
        r = float(m.group(1))
        p = float(m.group(2))
        return (r, p)
    except Exception:
        return None


def _group_files_by_params(filepaths: List[str]) -> Dict[tuple, List[str]]:
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for fp in filepaths:
        fn = os.path.basename(fp)
        parsed = _parse_params_from_filename(fn)
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

    train_groups = _group_files_by_params(train_files_all)
    valid_groups = _group_files_by_params(valid_files_all)

    # Restrict to an explicit list of (rayleigh, prandtl) combos -- e.g. for a quick
    # speed/timing check without paying for the full ~300GB dataset. Takes priority
    # over use_all_params/param_choice; per-file params are still tracked since more
    # than one distinct combo may be present.
    chosen = None
    if param_choices:
        wanted = {tuple(c) for c in param_choices}
        train_files = [fp for fp in train_files_all
                       if _parse_params_from_filename(os.path.basename(fp)) in wanted]
        val_files = [fp for fp in valid_files_all
                     if _parse_params_from_filename(os.path.basename(fp)) in wanted]
        found = {_parse_params_from_filename(os.path.basename(fp)) for fp in train_files + val_files}
        missing = wanted - found
        if missing:
            print(f"[create_param_dataloaders] warning: no files found for combos {missing}")
        print(f"[create_param_dataloaders] param_choices={sorted(wanted)} -> "
              f"{len(train_files)} train file(s), {len(val_files)} val file(s)")

        def _params_for_list(lst):
            return [_parse_params_from_filename(os.path.basename(fp)) for fp in lst]

        train_file_params = _params_for_list(train_files)
        val_file_params = _params_for_list(val_files)
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
                parsed = _parse_params_from_filename(os.path.basename(fp))
                out.append(parsed)
            return out

        train_file_params = _params_for_list(train_files)
        val_file_params = _params_for_list(val_files)

    trainset = H5RayleighBenardFields(
        train_files,
        file_limit=train_file_limit,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=train_file_params,
        return_params=return_params or use_all_params,
        cache_mode=cache_mode,
        traj_cache_capacity=traj_cache_capacity,
    )

    valset = H5RayleighBenardFields(
        val_files,
        file_limit=val_file_limit,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=val_file_params,
        return_params=return_params or use_all_params,
        cache_mode=cache_mode,
        traj_cache_capacity=traj_cache_capacity,
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
