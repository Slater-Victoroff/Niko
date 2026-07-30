import argparse
import copy
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure `src/niko` is on sys.path so this works from project root:
# python src/niko/export/export_webgpu_bundle.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    ckpt = Path(checkpoint)
    if ckpt.exists():
        return ckpt.resolve()

    # Convenience: allow passing only filename for /app/checkpoints in container.
    container_ckpt = Path("/app/checkpoints") / checkpoint
    if container_ckpt.exists():
        return container_ckpt.resolve()

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}. Also checked {container_ckpt}."
    )


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    explicit_config_path: Path | None = None,
):
    ckpt_obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
        ckpt_cfg = ckpt_obj.get("config")
        metadata = {k: v for k, v in ckpt_obj.items() if k != "model_state_dict"}
    else:
        state_dict = ckpt_obj
        ckpt_cfg = None
        metadata = {}

    # Explicit --config always wins.
    if explicit_config_path is not None:
        print(f"Loading config from: {explicit_config_path}")
        with open(explicit_config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt_cfg
        if cfg is not None and "name" not in cfg.get("param_encoder", {}):
            # Config was saved after build_model mutated it — reload from disk.
            cfg = None

        if cfg is None:
            config_path_str = metadata.get("config_path")
            if config_path_str is not None:
                config_path = Path(config_path_str)
                search_roots = [
                    Path("/app"),
                    checkpoint_path.parents[2],
                    checkpoint_path.parents[3],
                ]
                resolved = None
                if config_path.exists():
                    resolved = config_path
                else:
                    for root in search_roots:
                        for candidate in (root / config_path.name, root / "configs" / config_path.name):
                            if candidate.exists():
                                resolved = candidate
                                break
                        if resolved:
                            break
                if resolved:
                    print(f"Reloading config from disk: {resolved}")
                    with open(resolved) as f:
                        cfg = yaml.safe_load(f)

        if cfg is None:
            raise ValueError(
                "No usable model config found. Pass --config /path/to/your.yaml to specify it explicitly."
            )

    try:
        model = build_model(copy.deepcopy(cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to build model from config: {e}") from e

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg, metadata


def _serialize_weights_and_manifest(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    include_state_dict_names: bool,
) -> tuple[dict[str, Any], bytes, int]:
    state_items = sorted(model.state_dict().items())

    tensors_manifest: dict[str, Any] = {}
    weights_blob = bytearray()
    total_params = 0

    for idx, (name, tensor) in enumerate(state_items):
        # 4-byte alignment
        pad = (-len(weights_blob)) % 4
        if pad:
            weights_blob.extend(b"\x00" * pad)

        t = tensor.detach().to("cpu").contiguous()
        arr = t.numpy().astype("<f4", copy=False).reshape(-1)
        raw = arr.tobytes(order="C")

        offset = len(weights_blob)
        n_values = int(arr.size)
        n_bytes = int(len(raw))

        key = name if include_state_dict_names else f"tensor_{idx:05d}"
        tensors_manifest[key] = {
            "shape": list(t.shape),
            "dtype": "float32",
            "offset_bytes": offset,
            "n_values": n_values,
            "n_bytes": n_bytes,
        }

        weights_blob.extend(raw)
        total_params += n_values

    manifest = {
        "format": "iph-webgpu-bundle",
        "format_version": 1,
        "dtype": "float32",
        "endianness": "little",
        "weight_file": "weights.f32.bin",
        "layout_convention": "torch",
        "tensors": tensors_manifest,
        "model": {
            "class": model.__class__.__name__,
            "config": cfg,
            "metadata": metadata,
        },
    }

    return manifest, bytes(weights_blob), total_params


def main():
    p = argparse.ArgumentParser(description="Export a Niko checkpoint to a .iph WebGPU bundle.")
    p.add_argument("checkpoint", help="Checkpoint path, or checkpoint filename under /app/checkpoints")
    p.add_argument("--out", required=True, help="Output .iph path")
    p.add_argument("--config", default=None, help="Optional YAML config override")
    p.add_argument(
        "--device",
        default=None,
        help="Device like cuda:0 or cpu. Default is cuda:0 if available else cpu.",
    )
    p.add_argument("--dtype", default="float32", help="Export dtype (currently only float32)")
    p.add_argument(
        "--include-state-dict-names",
        type=_str2bool,
        default=True,
        help="Whether to keep state_dict names in manifest tensor keys (true/false).",
    )

    args = p.parse_args()

    if args.dtype.lower() != "float32":
        raise ValueError("Only --dtype float32 is currently supported.")

    if args.device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    explicit_config = Path(args.config) if args.config else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg, metadata = _load_model_from_checkpoint(ckpt_path, dev, explicit_config)

    manifest, weights_bytes, total_params = _serialize_weights_and_manifest(
        model=model,
        cfg=cfg,
        metadata=metadata,
        include_state_dict_names=bool(args.include_state_dict_names),
    )

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", manifest_json)
        zf.writestr("weights.f32.bin", weights_bytes)

    n_tensors = len(manifest["tensors"])
    size_mb = len(weights_bytes) / (1024.0 * 1024.0)

    print("=== Export Summary ===")
    print(f"output: {out_path}")
    print(f"num_tensors: {n_tensors}")
    print(f"total_parameters: {total_params}")
    print(f"weights_size_mb: {size_mb:.3f}")


if __name__ == "__main__":
    main()
