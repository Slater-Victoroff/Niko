import time
import torch
import argparse

from baseline import LastFrameBaseline, MeanFrameBaseline, ConvNeXTDecoder


def vrmse(pred, target, eps=1e-7):
    # mse per (B,C)
    mse = ((pred - target) ** 2).mean(dim=(-2, -1))
    var = target.std(dim=(-2, -1)) ** 2
    return torch.sqrt(mse / (var + eps)).mean()


def run_benchmark(batch_size=4, time_steps=3, channels=2, H=64, W=64, iters=50, warmup=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # synthetic batch: (B, T, C, H, W)
    ctx = torch.randn(batch_size, time_steps, channels, H, W, dtype=torch.float32, device=device)
    tgt = torch.randn(batch_size, channels, H, W, dtype=torch.float32, device=device)

    models = {
        "last": LastFrameBaseline().to(device),
        "mean": MeanFrameBaseline(spatial_mean=False).to(device),
        "mean_const": MeanFrameBaseline(spatial_mean=True).to(device),
    }

    # Optionally add a small ConvNeXT decoder if GPU/CPU available. Use hdim=channels and small k=2.
    try:
        decoder = ConvNeXTDecoder(k=2, hdim=channels)
        models["convnext_decoder"] = decoder.to(device)
        # prepare decoder input: (B, hdim, H//k, W//k)
        dec_in = torch.randn(batch_size, channels, H // 2, W // 2, device=device)
    except Exception as e:
        print(f"Skipping ConvNeXTDecoder (constructor failed): {e}")

    results = {}
    for name, m in models.items():
        # warmup
        for _ in range(warmup):
            if name == "convnext_decoder":
                _ = m(dec_in)
            else:
                _ = m(ctx)
        # timed runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sum_vr = 0.0
        vr_count = 0
        for _ in range(iters):
            if name == "convnext_decoder":
                out = m(dec_in)
            else:
                out = m(ctx)
            if isinstance(out, torch.Tensor) and out.shape == tgt.shape:
                try:
                    v = vrmse(out, tgt).item()
                    sum_vr += v
                    vr_count += 1
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / iters * 1000.0
        avg_vr = (sum_vr / vr_count) if vr_count > 0 else None
        results[name] = (avg_ms, avg_vr)

    print("Benchmark results:")
    for k, v in results.items():
        if isinstance(v, tuple):
            ms, vr = v
            vr_str = f"vrmse: {vr:.6f}" if vr is not None else "vrmse: n/a"
            print(f" - {k}: {ms:.3f} ms, {vr_str}")
        else:
            print(f" - {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight baseline benchmark (docker-friendly CLI)")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--iters", type=int, default=50, help="number of measured iterations")
    args = parser.parse_args()

    # keep a small, fixed synthetic configuration that's friendly for containers
    run_benchmark(batch_size=args.batch, time_steps=3, H=64, W=64, iters=args.iters)
