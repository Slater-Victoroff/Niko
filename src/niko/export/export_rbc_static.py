import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


CHANNELS = ["pressure", "buoyancy", "velocity_x", "velocity_y"]


def print_h5_tree(path: Path):
    print(f"\n=== HDF5 tree: {path} ===")
    with h5py.File(path, "r") as f:
        def walk(g, prefix=""):
            for k, v in g.items():
                p = f"{prefix}/{k}" if prefix else k
                if isinstance(v, h5py.Dataset):
                    print(f"DATASET {p} shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"GROUP   {p}")
                    walk(v, p)

        walk(f)
    print("=== End tree ===\n")


def _all_dataset_paths(h5: h5py.File) -> list[str]:
    out: list[str] = []

    def walk(g, prefix=""):
        for k, v in g.items():
            p = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                out.append(p)
            else:
                walk(v, p)

    walk(h5)
    return sorted(out)


def _find_unique_dataset(h5: h5py.File, exact_path: str, basename: str) -> str:
    all_paths = _all_dataset_paths(h5)
    if exact_path in h5:
        return exact_path

    matches = [p for p in all_paths if p.split("/")[-1].lower() == basename.lower()]
    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise RuntimeError(
            f"Could not find dataset for '{basename}'. Expected '{exact_path}' or a unique dataset named '{basename}'."
        )

    raise RuntimeError(
        f"Ambiguous dataset name for '{basename}'. Matches: {matches}. "
        "Please standardize the HDF5 layout."
    )


def _infer_xy_axes_from_shape(traj_shape: tuple[int, ...], x_len: int, y_len: int, label: str) -> tuple[int, int]:
    if len(traj_shape) < 3:
        raise RuntimeError(f"{label} trajectory must have >=3 dims (T + 2 spatial), got {traj_shape}")

    x_axes = [ax for ax in range(1, len(traj_shape)) if traj_shape[ax] == x_len]
    y_axes = [ax for ax in range(1, len(traj_shape)) if traj_shape[ax] == y_len]

    if len(x_axes) != 1 or len(y_axes) != 1:
        raise RuntimeError(
            f"Ambiguous x/y axes for {label}. shape={traj_shape}, x_len={x_len}, y_len={y_len}, x_axes={x_axes}, y_axes={y_axes}"
        )

    return x_axes[0], y_axes[0]


def _parse_params_from_filename(path: Path) -> tuple[float | None, float | None]:
    num = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?"
    pattern = rf"Rayleigh_({num})_Prandtl_({num})\.hdf5$"
    m = re.search(pattern, path.name, flags=re.I)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _stats_per_channel(x_tchw: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ci, ch in enumerate(CHANNELS):
        arr = x_tchw[:, ci, :, :]
        out[ch] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    return out


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _file_slug(input_path: Path) -> str:
    """Derive a filesystem-safe slug from an HDF5 filename."""
    stem = input_path.stem  # e.g. rayleigh_benard_Rayleigh_1e10_Prandtl_1
    num = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?"
    m = re.search(rf"Rayleigh_({num})_Prandtl_({num})$", stem, flags=re.I)
    if m:
        ra = m.group(1).replace("+", "").lower()
        pr = m.group(2).replace("+", "").lower()
        return f"rayleigh_{ra}_prandtl_{pr}"
    # Fallback: sanitize the whole stem
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).lower()


def export_rbc_static(input_path: Path, out_dir: Path, verify: bool = False):
    print_h5_tree(input_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rayleigh, prandtl = _parse_params_from_filename(input_path)

    with h5py.File(input_path, "r") as f:
        pressure_key = _find_unique_dataset(f, "t0_fields/pressure", "pressure")
        buoyancy_key = _find_unique_dataset(f, "t0_fields/buoyancy", "buoyancy")
        velocity_key = _find_unique_dataset(f, "t1_fields/velocity", "velocity")

        if "dimensions/x" not in f or "dimensions/y" not in f:
            raise RuntimeError("Expected dimensions/x and dimensions/y datasets in HDF5 file.")

        p_ds = f[pressure_key]
        b_ds = f[buoyancy_key]
        v_ds = f[velocity_key]

        if p_ds.ndim != 4 or b_ds.ndim != 4 or v_ds.ndim != 5:
            raise RuntimeError(
                "Expected pressure/buoyancy to be 4D (N,T,*,*) and velocity to be 5D (N,T,*,*,2). "
                f"Got pressure={p_ds.shape}, buoyancy={b_ds.shape}, velocity={v_ds.shape}."
            )

        n_seq = int(p_ds.shape[0])
        if n_seq != 5:
            raise RuntimeError(f"Expected exactly 5 sequences, found {n_seq}.")

        if p_ds.shape[:2] != b_ds.shape[:2] or p_ds.shape[:2] != v_ds.shape[:2]:
            raise RuntimeError(
                f"Mismatched (N,T): pressure={p_ds.shape[:2]} buoyancy={b_ds.shape[:2]} velocity={v_ds.shape[:2]}"
            )

        x_len = int(f["dimensions/x"].shape[0])
        y_len = int(f["dimensions/y"].shape[0])

        p_traj_shape = tuple(p_ds.shape[1:])
        b_traj_shape = tuple(b_ds.shape[1:])
        v_traj_shape = tuple(v_ds.shape[1:])

        p_x_ax, p_y_ax = _infer_xy_axes_from_shape(p_traj_shape, x_len, y_len, "pressure")
        b_x_ax, b_y_ax = _infer_xy_axes_from_shape(b_traj_shape, x_len, y_len, "buoyancy")
        v_x_ax, v_y_ax = _infer_xy_axes_from_shape(v_traj_shape, x_len, y_len, "velocity")

        comp_axes = [ax for ax in range(1, len(v_traj_shape)) if v_traj_shape[ax] == 2]
        if len(comp_axes) != 1:
            raise RuntimeError(
                f"Ambiguous velocity component axis (size=2). velocity trajectory shape={v_traj_shape}, candidates={comp_axes}"
            )
        v_comp_ax = comp_axes[0]

        sequence_entries = []

        for seq_idx in range(n_seq):
            p_seq = np.asarray(p_ds[seq_idx], dtype=np.float32)
            b_seq = np.asarray(b_ds[seq_idx], dtype=np.float32)
            v_seq = np.asarray(v_ds[seq_idx], dtype=np.float32)

            p_thw = np.moveaxis(p_seq, [0, p_y_ax, p_x_ax], [0, 1, 2])
            b_thw = np.moveaxis(b_seq, [0, b_y_ax, b_x_ax], [0, 1, 2])
            v_thwc = np.moveaxis(v_seq, [0, v_y_ax, v_x_ax, v_comp_ax], [0, 1, 2, 3])

            if v_thwc.shape[-1] != 2:
                raise RuntimeError(f"Velocity component axis reorder failed, got shape={v_thwc.shape}")

            vx = v_thwc[..., 0]
            vy = v_thwc[..., 1]

            x_tchw = np.stack([p_thw, b_thw, vx, vy], axis=1)
            x_tchw = np.ascontiguousarray(x_tchw, dtype=np.float32)

            t, c, h, w = x_tchw.shape
            if c != 4:
                raise RuntimeError(f"Expected 4 channels, got shape={x_tchw.shape}")

            seq_dir = out_dir / f"sequence_{seq_idx:03d}"
            seq_dir.mkdir(parents=True, exist_ok=True)

            bin_path = seq_dir / "sequence_f32.bin"
            data_le = x_tchw.astype("<f4", copy=False)
            data_le.tofile(bin_path)

            byte_length = int(bin_path.stat().st_size)
            expected_bytes = int(t * c * h * w * 4)
            if byte_length != expected_bytes:
                raise RuntimeError(
                    f"byteLength mismatch for sequence_{seq_idx:03d}: got={byte_length}, expected={expected_bytes}"
                )

            seq_manifest = {
                "shape": [int(t), int(c), int(h), int(w)],
                "dtype": "float32",
                "layout": "TCHW",
                "channels": CHANNELS,
                "source_file": input_path.name,
                "sequence_index": int(seq_idx),
                "rayleigh": rayleigh,
                "prandtl": prandtl,
                "stats": _stats_per_channel(x_tchw),
                "file": "sequence_f32.bin",
                "byteLength": byte_length,
            }
            _write_json(seq_dir / "manifest.json", seq_manifest)

            sequence_entries.append(
                {
                    "sequence_index": int(seq_idx),
                    "dir": seq_dir.name,
                    "manifest": f"{seq_dir.name}/manifest.json",
                    "file": f"{seq_dir.name}/sequence_f32.bin",
                    "shape": [int(t), int(c), int(h), int(w)],
                    "byteLength": byte_length,
                }
            )

    top_manifest = {
        "dataset": "rayleigh-benard-static",
        "source_file": input_path.name,
        "channels": CHANNELS,
        "num_sequences": len(sequence_entries),
        "rayleigh": rayleigh,
        "prandtl": prandtl,
        "sequences": sequence_entries,
    }
    _write_json(out_dir / "manifest.json", top_manifest)

    print(f"Export complete: {out_dir}")
    print(f"Sequences exported: {len(sequence_entries)}")

    if verify:
        verify_export(out_dir)

    return sequence_entries, rayleigh, prandtl


def export_rbc_directory(
    input_dir: Path,
    out_dir: Path,
    verify: bool = False,
    glob: str = "*.hdf5",
):
    files = sorted(input_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No files matching '{glob}' found in {input_dir}")

    print(f"Found {len(files)} HDF5 files in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_files_entries = []

    for i, hdf5_path in enumerate(files):
        slug = _file_slug(hdf5_path)
        file_out_dir = out_dir / slug
        print(f"\n[{i+1}/{len(files)}] {hdf5_path.name} → {file_out_dir.name}/")
        seq_entries, rayleigh, prandtl = export_rbc_static(
            input_path=hdf5_path,
            out_dir=file_out_dir,
            verify=verify,
        )
        all_files_entries.append(
            {
                "source_file": hdf5_path.name,
                "slug": slug,
                "dir": slug,
                "manifest": f"{slug}/manifest.json",
                "rayleigh": rayleigh,
                "prandtl": prandtl,
                "num_sequences": len(seq_entries),
                "sequences": [
                    {**s, "file": f"{slug}/{s['file']}", "manifest": f"{slug}/{s['manifest']}"}
                    for s in seq_entries
                ],
            }
        )

    top_manifest = {
        "dataset": "rayleigh-benard-static",
        "channels": CHANNELS,
        "num_files": len(all_files_entries),
        "total_sequences": sum(e["num_sequences"] for e in all_files_entries),
        "files": all_files_entries,
    }
    _write_json(out_dir / "manifest.json", top_manifest)

    print(f"\n=== Directory export complete ===")
    print(f"Files exported: {len(all_files_entries)}")
    print(f"Total sequences: {top_manifest['total_sequences']}")
    print(f"Top-level manifest: {out_dir / 'manifest.json'}")


def verify_export(out_dir: Path):
    print("\n=== Verify export ===")
    top = json.loads((out_dir / "manifest.json").read_text())

    for s in top["sequences"]:
        seq_dir = out_dir / s["dir"]
        seq_manifest = json.loads((seq_dir / "manifest.json").read_text())
        shape = seq_manifest["shape"]
        t, c, h, w = shape
        bin_path = seq_dir / seq_manifest["file"]

        raw = np.fromfile(bin_path, dtype="<f4")
        expected = t * c * h * w
        if raw.size != expected:
            raise RuntimeError(
                f"Verify failed for {seq_dir.name}: value count mismatch got={raw.size} expected={expected}"
            )
        x = raw.reshape(t, c, h, w)

        print(f"[{seq_dir.name}]")
        print(f"shape={list(x.shape)}")
        print(f"min={float(x.min()):.6g}")
        print(f"max={float(x.max()):.6g}")
        print(f"mean={float(x.mean()):.6g}")
        print(f"std={float(x.std()):.6g}")
        vals = " ".join(f"{float(v):.6g}" for v in x.reshape(-1)[:16])
        print("first_values:")
        print(vals)

    print("=== Verify done ===")


def main():
    p = argparse.ArgumentParser(description="Export RB HDF5 file(s) to static browser-loadable binaries.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to a single rayleigh_benard_*.hdf5 file")
    src.add_argument("--input-dir", help="Directory containing multiple *.hdf5 files to export all at once")
    p.add_argument("--out", required=True, help="Output directory root")
    p.add_argument("--verify", action="store_true", help="Reload exported files and print shape/stats/first values")
    p.add_argument("--glob", default="*.hdf5", help="Glob pattern for --input-dir (default: *.hdf5)")

    args = p.parse_args()
    out_dir = Path(args.out)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        export_rbc_static(input_path=input_path, out_dir=out_dir, verify=args.verify)
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
        export_rbc_directory(input_dir=input_dir, out_dir=out_dir, verify=args.verify, glob=args.glob)


if __name__ == "__main__":
    main()
