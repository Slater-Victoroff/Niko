import argparse
import copy
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Ensure `src/niko` is on sys.path so this works from project root:
# python src/niko/eval/probe_rb_checkpoint.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model
from core.states import Params
import dataloader as dl


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


def _resolve_test_subdir(base_dir: Path, requested_subdir: str) -> str:
    requested_path = base_dir / requested_subdir
    if requested_path.is_dir():
        return requested_subdir

    # Friendly fallback for setups that use valid/ instead of test/
    if requested_subdir == "test" and (base_dir / "valid").is_dir():
        print("Requested test split not found; falling back to valid split.")
        return "valid"

    raise FileNotFoundError(
        f"Split folder not found: {requested_path}. "
        f"Available under {base_dir}: {[p.name for p in base_dir.iterdir() if p.is_dir()]}"
    )


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    explicit_config_path: Path | None = None,
):
    # Support WebGPU bundle checkpoints exported as .iph.
    if checkpoint_path.suffix.lower() == ".iph":
        return _load_model_from_iph_bundle(checkpoint_path, device, explicit_config_path)

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
            # Try to find the original YAML from the path stored in the checkpoint.
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

    # build_model mutates nested dicts via pop("name"), so use a copy
    try:
        model = build_model(copy.deepcopy(cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to build model from config: {e}") from e

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg, metadata


def _load_model_from_iph_bundle(
    checkpoint_path: Path,
    device: torch.device,
    explicit_config_path: Path | None = None,
):
    with zipfile.ZipFile(checkpoint_path, mode="r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names or "weights.f32.bin" not in names:
            raise ValueError(
                f"{checkpoint_path} is not a supported .iph bundle (missing manifest.json or weights.f32.bin)."
            )

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        weights_raw = zf.read("weights.f32.bin")

    meta_model = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    cfg_from_manifest = meta_model.get("config") if isinstance(meta_model, dict) else None
    metadata = meta_model.get("metadata", {}) if isinstance(meta_model, dict) else {}

    if explicit_config_path is not None:
        print(f"Loading config from: {explicit_config_path}")
        with open(explicit_config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = cfg_from_manifest

    if cfg is None:
        raise ValueError(
            "No usable model config found in .iph bundle. Pass --config /path/to/your.yaml."
        )

    try:
        model = build_model(copy.deepcopy(cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to build model from config: {e}") from e

    tensors = manifest.get("tensors", {})
    if not isinstance(tensors, dict) or len(tensors) == 0:
        raise ValueError("Invalid .iph bundle: manifest.tensors is empty or missing.")

    model_keys = set(model.state_dict().keys())
    tensor_keys = set(tensors.keys())

    # Exporter can optionally use generic keys (tensor_00000), which are not mappable.
    if not tensor_keys.issubset(model_keys):
        extra = sorted(tensor_keys - model_keys)
        preview = ", ".join(extra[:10])
        raise ValueError(
            "This .iph bundle does not contain state_dict tensor names compatible with the model. "
            "Re-export with --include-state-dict-names true. "
            f"Example non-matching keys: {preview}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    total_bytes = len(weights_raw)

    for name, spec in tensors.items():
        offset = int(spec["offset_bytes"])
        n_values = int(spec["n_values"])
        shape = tuple(int(v) for v in spec["shape"])
        n_bytes = int(spec["n_bytes"])

        if offset < 0 or n_bytes < 0 or (offset + n_bytes) > total_bytes:
            raise ValueError(
                f"Corrupt tensor slice for '{name}': offset={offset} n_bytes={n_bytes} total={total_bytes}"
            )

        arr = np.frombuffer(weights_raw, dtype="<f4", count=n_values, offset=offset)
        t = torch.from_numpy(arr.copy()).reshape(shape)
        state_dict[name] = t

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch when loading .iph bundle. missing={missing} unexpected={unexpected}"
        )

    model.to(device)
    model.eval()
    return model, cfg, metadata


def print_tensor_probe(name: str, x: torch.Tensor, n_values: int = 16):
    t = x.detach().to("cpu")
    flat = t.reshape(-1)

    print(f"[{name}]")
    print(f"shape={list(t.shape)}")
    print(f"min={float(t.min()):.6g}")
    print(f"max={float(t.max()):.6g}")
    print(f"mean={float(t.mean()):.6g}")
    print(f"std={float(t.std(unbiased=False)):.6g}")
    print("first_values:")
    n = min(int(n_values), int(flat.numel()))
    vals = " ".join(f"{float(v):.6g}" for v in flat[:n])
    print(vals)
    print()


def _to_tensor_output(output):
    if hasattr(output, "grid"):
        return output.grid
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


def _to_grid_tensor(x):
    if hasattr(x, "grid"):
        return x.grid
    if torch.is_tensor(x):
        return x
    return None


def attach_activation_hooks(model, layer_names: list[str]):
    modules = dict(model.named_modules())
    activations: dict[str, list[torch.Tensor]] = {}
    handles = []

    for name in layer_names:
        if name not in modules:
            available = sorted(modules.keys())
            preview = ", ".join(available[:40])
            raise ValueError(
                f"Layer '{name}' not found. First available module names: {preview}"
            )

        def _hook(_module, _inputs, output, layer_name=name):
            t = _to_tensor_output(output)
            if t is None:
                return
            activations.setdefault(layer_name, []).append(t.detach().to("cpu"))

        handles.append(modules[name].register_forward_hook(_hook))

    return activations, handles


def _matching_file_indices(dataset, file_filter: str):
    target = os.path.basename(file_filter)
    indices = []
    for i, p in enumerate(dataset._pairs):
        fp = dataset.files[p.file_idx]
        if os.path.basename(fp) == target:
            indices.append(i)
    return indices


def get_one_sample(loader, sample_index: int, file_filter: str | None = None):
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")

    dataset = loader.dataset

    if file_filter is not None:
        candidate_indices = _matching_file_indices(dataset, file_filter)
        if not candidate_indices:
            available = sorted({os.path.basename(p) for p in dataset.files})
            preview = ", ".join(available[:30])
            raise ValueError(
                f"No windows found for file '{file_filter}'. Available files include: {preview}"
            )
        if sample_index >= len(candidate_indices):
            raise IndexError(
                f"sample_index={sample_index} out of range for file '{file_filter}'. "
                f"num_windows_in_file={len(candidate_indices)}"
            )
        global_index = candidate_indices[sample_index]
        local_window_index = sample_index
    else:
        if sample_index >= len(dataset):
            raise IndexError(
                f"sample_index={sample_index} out of range. Total samples={len(dataset)}"
            )
        global_index = sample_index
        pair0 = dataset._pairs[global_index]
        file_indices = [i for i, p in enumerate(dataset._pairs) if p.file_idx == pair0.file_idx]
        local_window_index = file_indices.index(global_index)

    pair = dataset._pairs[global_index]
    sample = dataset[global_index]
    if len(sample) == 3:
        xb, yb, bparams = sample
    else:
        xb, yb = sample
        bparams = None

    xb = xb.unsqueeze(0)
    yb = yb.unsqueeze(0)
    if bparams is not None:
        bparams = bparams.unsqueeze(0)

    context_frames = int(dataset.context_frames)
    rollout_steps = int(dataset.predict_frames)
    t0 = int(pair.t)
    metadata = {
        "file": os.path.basename(dataset.files[pair.file_idx]),
        "global_sample_index": int(global_index),
        "local_window_index": int(local_window_index),
        "context_frames": context_frames,
        "rollout_steps": rollout_steps,
        "context_frame_range": [t0, t0 + context_frames - 1],
        "target_frame_range": [t0 + context_frames, t0 + context_frames + rollout_steps - 1],
    }
    return xb, yb, bparams, metadata


def print_weight_tensor(model, path: str, n_values: int = 16):
    """Extract and print a weight/bias tensor from the model by dot notation path."""
    parts = path.replace(".", " ").split()
    obj = model
    
    for part in parts:
        # Handle indexing like "transport_net.0.weight" -> transport_net[0]
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    
    if not torch.is_tensor(obj):
        raise ValueError(f"Path '{path}' does not resolve to a tensor")
    
    t = obj.detach().to("cpu")
    flat = t.reshape(-1)
    
    print(f"[{path}]")
    print(f"shape={list(t.shape)}")
    print(f"min={float(t.min()):.6g}")
    print(f"max={float(t.max()):.6g}")
    print(f"mean={float(t.mean()):.6g}")
    print(f"std={float(t.std(unbiased=False)):.6g}")
    print("first_values:")
    n = min(int(n_values), int(flat.numel()))
    vals = " ".join(f"{float(v):.6g}" for v in flat[:n])
    print(vals)
    print()


def main():
    p = argparse.ArgumentParser(
        description="Probe one RB checkpoint sample and print activation summaries.",
    )
    p.add_argument("checkpoint", help="Checkpoint path, or checkpoint filename under /app/checkpoints")
    p.add_argument(
        "--config",
        default=None,
        help="Path to training YAML. Required for checkpoints that don't embed config_path.",
    )
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--file",
        default=None,
        help="Optional HDF5 filename to probe (basename match), e.g. rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5",
    )
    p.add_argument(
        "--layers",
        default="",
        help="Comma-separated module names from model.named_modules(), e.g. encoder.conv1,encoder.gelu1",
    )
    p.add_argument("--print-values", type=int, default=16)
    p.add_argument(
        "--device",
        default=None,
        help="Device like cuda:0 or cpu. Default is cuda:0 if available else cpu.",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Comma-separated weight paths to print, e.g. operator.transport_net.0.weight,operator.transport_net.0.bias",
    )

    args = p.parse_args()

    if args.device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    explicit_config = Path(args.config) if args.config else None

    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg, meta = _load_model_from_checkpoint(ckpt_path, dev, explicit_config)

    context_frames = int(meta.get("context_frames", cfg["encoder"].get("context_frames", 1)))
    rollout_steps = int(meta.get("rollout_steps", cfg.get("rollout_steps", 1)))

    # Print model weights if requested
    weight_paths = [s.strip() for s in args.weights.split(",") if s.strip()]
    if weight_paths:
        print("=" * 60)
        print("MODEL WEIGHTS")
        print("=" * 60)
        for path in weight_paths:
            try:
                print_weight_tensor(model, path, args.print_values)
            except Exception as e:
                print(f"[{path}]")
                print(f"Error: {e}")
                print()
        print("=" * 60)

    data_dir = Path("/app/data/datasets/rayleigh_benard/data")
    split = _resolve_test_subdir(data_dir, args.split)
    print(f"Probing split: {split} (base: {data_dir})")

    _, probe_loader, _ = dl.create_param_dataloaders(
        str(data_dir),
        batch_size=8,
        num_workers=0,
        context_frames=context_frames,
        predict_frames=rollout_steps,
        train_file_limit=None,
        val_file_limit=None,
        train_subdir=split,
        valid_subdir=split,
        shuffle_train=False,
        shuffle_val=False,
        use_all_params=True,
        return_params=True,
    )

    xb, yb, bparams, sample_meta = get_one_sample(
        probe_loader,
        args.sample_index,
        file_filter=args.file,
    )
    xb = xb.to(dev)
    yb = yb.to(dev)

    if bparams is None:
        bparams = torch.zeros((1, 2), dtype=torch.float32, device=dev)
    else:
        bparams = bparams.to(device=dev, dtype=torch.float32)

    params = Params(values=bparams)

    params_list = [f"{float(v):.6g}" for v in bparams.detach().to("cpu").reshape(-1)]
    print(f"file: {sample_meta['file']}")
    print(f"global_sample_index: {sample_meta['global_sample_index']}")
    print(f"local_window_index: {sample_meta['local_window_index']}")
    print(f"context_frames: {sample_meta['context_frames']}")
    print(f"rollout_steps: {sample_meta['rollout_steps']}")
    print(f"context_frame_range: {sample_meta['context_frame_range']}")
    print(f"target_frame_range: {sample_meta['target_frame_range']}")
    print(f"params: [{', '.join(params_list)}]")
    print()

    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()]
    activations = {}
    handles = []
    if layer_names:
        activations, handles = attach_activation_hooks(model, layer_names)

    # If operator is requested in --layers, also print conditioned input slices
    # without requiring any command changes.
    operator_probe: dict[str, torch.Tensor] = {}
    if "operator" in layer_names and hasattr(model, "operator"):
        def _operator_pre_hook(_module, inputs, kwargs=None):
            x_in = _to_grid_tensor(inputs[0]) if len(inputs) > 0 else None
            cond = None
            if kwargs is not None:
                cond = kwargs.get("cond")
            if x_in is not None and "x" not in operator_probe:
                operator_probe["x"] = x_in.detach()
            if torch.is_tensor(cond) and "cond" not in operator_probe:
                operator_probe["cond"] = cond.detach()

        try:
            handles.append(model.operator.register_forward_pre_hook(_operator_pre_hook, with_kwargs=True))
        except TypeError:
            # Fallback for older torch versions without with_kwargs support.
            def _operator_pre_hook_no_kwargs(_module, inputs):
                x_in = _to_grid_tensor(inputs[0]) if len(inputs) > 0 else None
                if x_in is not None and "x" not in operator_probe:
                    operator_probe["x"] = x_in.detach()

            handles.append(model.operator.register_forward_pre_hook(_operator_pre_hook_no_kwargs))

    try:
        with torch.no_grad():
            pred = model(
                xb,
                steps=rollout_steps,
                params=params,
                return_initial_encode=False,
            )
    finally:
        for h in handles:
            h.remove()

    print_tensor_probe("xb", xb, args.print_values)
    print_tensor_probe("params", bparams, args.print_values)
    print_tensor_probe("pred", pred, args.print_values)
    print_tensor_probe("yb", yb, args.print_values)

    if "operator" in layer_names:
        x_for_xc = operator_probe.get("x")
        cond_for_xc = operator_probe.get("cond")
        if x_for_xc is None:
            print("[xc]")
            print("operator input not captured")
            print()
        elif cond_for_xc is None:
            print("[xc]")
            print("operator cond not captured")
            print()
        elif hasattr(model.operator, "_conditioned_input"):
            try:
                with torch.no_grad():
                    xc = model.operator._conditioned_input(x_for_xc, cond_for_xc)
                if xc.ndim == 4 and xc.shape[0] > 0 and xc.shape[1] > 16 and xc.shape[2] > 0 and xc.shape[3] >= 8:
                    print("xc[0,0,0,:8]:", xc[0, 0, 0, :8].detach().to("cpu").tolist())
                    print("xc[0,16,0,:8]:", xc[0, 16, 0, :8].detach().to("cpu").tolist())
                    print()
                else:
                    print("[xc]")
                    print(f"xc shape is {list(xc.shape)}; expected at least [1, 17, 1, 8] for requested slices")
                    print()
            except Exception as e:
                print("[xc]")
                print(f"failed to compute conditioned input: {e}")
                print()
        else:
            print("[xc]")
            print("operator has no _conditioned_input method")
            print()

    for name in layer_names:
        if name not in activations:
            print(f"[{name}]")
            print("no tensor output captured")
            print()
            continue

        captured = activations[name]
        if len(captured) == 1:
            t = captured[0]
            # Decoder runs on flattened [B*T, ...] in one call. Unflatten for per-step logging.
            if name.startswith("decoder") and t.ndim >= 1 and xb.shape[0] > 0 and rollout_steps > 1 and t.shape[0] == xb.shape[0] * rollout_steps:
                t_bt = t.reshape(xb.shape[0], rollout_steps, *t.shape[1:])
                for i in range(rollout_steps):
                    print_tensor_probe(f"{name}[step={i}]", t_bt[:, i], args.print_values)
            else:
                print_tensor_probe(name, t, args.print_values)
        else:
            for i, t in enumerate(captured):
                print_tensor_probe(f"{name}[step={i}]", t, args.print_values)


if __name__ == "__main__":
    main()
