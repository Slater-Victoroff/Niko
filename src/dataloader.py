import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


@dataclass(frozen=True)
class PairIndex:
    file_idx: int
    traj_idx: int
    t: int


class H5VelocityFramePairs(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        dataset_key: str = "t1_fields/velocity",
        file_limit: Optional[int] = None,
        traj_limit: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        cache_mode: str = "file",  # "none" | "file" | "traj"
        transform=None,
        stack_frames: bool = False,
        context_frames: int = 1,
        predict_frames: int = 1,
        device: Optional[torch.device] = None,
        file_params: Optional[List[Optional[tuple]]] = None,
        return_params: bool = False,
    ):
        self.dataset_key = dataset_key
        self.dtype = dtype
        self.cache_mode = cache_mode
        self.transform = transform
        self.files = filepaths
        self.stack_frames = stack_frames
        self.context_frames = int(context_frames)
        self.predict_frames = int(predict_frames)
        self.device = torch.device(device) if device is not None else None
        # optional per-file params (rayleigh, prandtl) parallel to `filepaths`
        self.file_params = file_params
        self.return_params = bool(return_params)

        self._pairs: List[PairIndex] = []
        if file_limit is not None:
            self.files = self.files[:file_limit]
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                dset = f[self.dataset_key]
                n_traj = dset.shape[0]
                T = dset.shape[1]
                if traj_limit is not None:
                    n_traj = min(n_traj, traj_limit)

            # valid start indices s depend on whether we form stacked pairs or single frames:
            # - if stack_frames is False: need C + P frames -> s + (C + P) <= T
            # - if stack_frames is True (we form overlapping pairs (f_i,f_{i+1})),
            #   need C + P + 1 frames -> s + (C + P + 1) <= T
            needed = self.context_frames + self.predict_frames
            if self.stack_frames:
                needed += 1

            max_start = -1
            if T >= needed:
                max_start = T - needed

            for tj in range(n_traj):
                for s in range(max_start + 1):
                    self._pairs.append(PairIndex(fi, tj, s))

        self._open_file_idx: Optional[int] = None
        self._h5: Optional[h5py.File] = None
        self._dset = None

        self._cached_traj_idx: Optional[int] = None
        self._cached_traj_tensor: Optional[torch.Tensor] = None

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
        self._dset = self._h5[self.dataset_key]
        self._open_file_idx = file_idx

        self._cached_traj_idx = None
        self._cached_traj_tensor = None

    def _get_traj_tensor(self, traj_idx: int) -> torch.Tensor:
        """
        Load one trajectory (T, X, Y, 2) into memory and convert to (T, 2, Y, X).
        Used only when cache_mode == "traj".
        """
        assert self._dset is not None
        arr = self._dset[traj_idx, ...]  # numpy / h5py array view -> materialize on slicing
        x = torch.from_numpy(np.array(arr, copy=False)).to(self.dtype)  # (T, X, Y, 2)
        x = x.permute(0, 3, 2, 1).contiguous()  # (T, 2, Y, X)
        return x

    def __getitem__(self, idx: int):
        p = self._pairs[idx]
        self._ensure_open(p.file_idx)

        # Ensure cached traj loaded when using traj cache
        if self.cache_mode == "traj":
            if self._cached_traj_tensor is None or self._cached_traj_idx != p.traj_idx:
                self._cached_traj_tensor = self._get_traj_tensor(p.traj_idx)
                self._cached_traj_idx = p.traj_idx

            traj = self._cached_traj_tensor
        else:
            # lazy load single trajectory into memory for this access
            arr0 = self._dset[p.traj_idx, ...]
            traj = torch.from_numpy(np.array(arr0, copy=False)).to(self.dtype).permute(0, 3, 2, 1).contiguous()

        s = p.t
        C = self.context_frames
        P = self.predict_frames

        if self.stack_frames:
            # Form overlapping pairs: for i in [0..C-1], pair = (frame_{s+i}, frame_{s+i+1}) -> 4 channels
            ctx_pairs = []
            for i in range(C):
                a = traj[s + i]
                b = traj[s + i + 1]
                pair = torch.cat([a, b], dim=0)  # (4, H, W)
                ctx_pairs.append(pair)
            ctx_out = torch.stack(ctx_pairs, dim=0)  # (C,4,H,W)

            tgt_pairs = []
            for j in range(P):
                a = traj[s + C + j]
                b = traj[s + C + j + 1]
                pair = torch.cat([a, b], dim=0)
                tgt_pairs.append(pair)
            tgt_out = torch.stack(tgt_pairs, dim=0)  # (P,4,H,W)

            if self.transform is not None:
                # apply transform per-pair (apply to each 4-channel tensor)
                ctx_out = torch.stack([self.transform(fp) for fp in ctx_out], dim=0)
                tgt_out = torch.stack([self.transform(fp) for fp in tgt_out], dim=0)

            if self.return_params:
                param = None
                if self.file_params is not None:
                    parsed = self.file_params[p.file_idx]
                    if parsed is not None:
                        param = torch.tensor(parsed, dtype=torch.float32)
                    else:
                        param = None
                return ctx_out, tgt_out, param

            return ctx_out, tgt_out

        # default: no stacking across time; keep time dimension then channels=2
        ctx = traj[s : s + C]      # (C, 2, H, W)
        tgt = traj[s + C : s + C + P]  # (P, 2, H, W)

        if self.transform is not None:
            ctx = torch.stack([self.transform(f) for f in ctx], dim=0)
            tgt = torch.stack([self.transform(f) for f in tgt], dim=0)

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


def create_param_dataloaders(
    base_dir: str,
    param_choice: Optional[tuple] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    train_file_limit: Optional[int] = None,
    val_file_limit: Optional[int] = None,
    stack_frames: bool = True,
    context_frames: int = 1,
    predict_frames: int = 1,
    train_subdir: str = "train",
    valid_subdir: str = "valid",
    device: Optional[str] = None,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    use_all_params: bool = True,
    return_params: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[tuple]]:
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

    # prefer parameter pairs that exist in both train and valid when not using the full split
    chosen = None
    if not use_all_params:
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

    trainset = H5VelocityFramePairs(
        train_files,
        file_limit=train_file_limit,
        stack_frames=stack_frames,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=train_file_params,
        return_params=return_params or use_all_params,
    )

    valset = H5VelocityFramePairs(
        val_files,
        file_limit=val_file_limit,
        stack_frames=stack_frames,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
        file_params=val_file_params,
        return_params=return_params or use_all_params,
    )

    # Use a seeded generator for reproducible shuffling when requested.
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=gen if shuffle_train else None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # For validation we usually don't shuffle; allow opt-in and separate generator
    val_gen = torch.Generator()
    val_gen.manual_seed(seed + 1)
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        generator=val_gen if shuffle_val else None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, chosen


def move_batch_to_device(batch, device: Optional[str]):
    """Move a (ctx, tgt) batch to `device` from the main process.

    Returns the moved batch (ctx, tgt). If `device` is None or not available,
    returns the input batch unchanged.
    """
    dev = torch.device(device) if device is not None else None
    if dev is None:
        return batch
    ctx, tgt = batch
    return ctx.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)


if __name__ == "__main__":
    # Simple, no-CLI smoke test for running inside the container
    BASE_DIR = "/app/data/datasets/rayleigh_benard/data"
    BATCH_SIZE = 2
    STACK_FRAMES = True
    # context : predict (e.g., 3:1)
    CONTEXT_FRAMES = 3
    PREDICT_FRAMES = 1

    print("Creating dataloaders for a random chosen parameter pair...")
    train_loader, val_loader, chosen = create_param_dataloaders(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        stack_frames=STACK_FRAMES,
        context_frames=CONTEXT_FRAMES,
        predict_frames=PREDICT_FRAMES,
    )
    print(f"Chosen parameter pair: {chosen}")

    # Move the batch to the chosen device in the main process to avoid
    # initializing CUDA inside forked DataLoader worker subprocesses.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch = next(iter(train_loader))
    ctx_batch, tgt_batch = move_batch_to_device(batch, device)
    print(f"ctx shape: {ctx_batch.shape}, tgt shape: {tgt_batch.shape}")
