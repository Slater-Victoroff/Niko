import os
import numpy as np
import h5py
from PIL import Image


def vel_uv_to_rg_png(vel_xy2: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    vel_xy2: (X, Y, 2) float32
    Returns: uint8 RGB image (Y, X, 3) where R=u, G=v, B=0
    """
    u = vel_xy2[..., 0]
    v = vel_xy2[..., 1]

    # Map [-scale, +scale] -> [0, 1]
    r = np.clip((u / scale + 1.0) * 0.5, 0.0, 1.0)
    g = np.clip((v / scale + 1.0) * 0.5, 0.0, 1.0)
    b = np.zeros_like(r)

    rgb_xy = np.stack([r, g, b], axis=-1)          # (X, Y, 3)
    rgb_yx = np.transpose(rgb_xy, (1, 0, 2))       # (Y, X, 3) for PIL
    return (rgb_yx * 255.0).astype(np.uint8)


def save_frames_simple(data_dir: str, out_dir: str, file_index: int = 0, traj_index: int = 0, t_start: int = 0, t_end: int = 50):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    path = os.path.join(data_dir, files[file_index])
    print("Using file:", path)

    with h5py.File(path, "r") as f:
        vel = f[VEL_KEY]  # (40, 200, 512, 128, 2)
        print("Velocity shape:", vel.shape)
        T = vel.shape[1]
        t_end = min(t_end, T)

        for t in range(t_start, t_end):
            v = vel[traj_index, t]   # (X, Y, 2)
            img = vel_uv_to_rg_png(v, scale=SCALE)
            out_path = os.path.join(out_dir, f"vel_{traj_index:02d}_{t:04d}.png")
            Image.fromarray(img).save(out_path)

    print("Saved frames to:", out_dir)

if __name__ == "__main__":
    DATA_DIR = "app/data/datasets/rayleigh_benard/data/train/"
    OUT_DIR  = "app/data/vel_frames_simple"
    VEL_KEY  = "t1_fields/velocity"

    SCALE = 1.0  # maps velocity values in [-SCALE, +SCALE] to [0, 255]
    save_frames_simple(DATA_DIR, OUT_DIR, file_index=0, traj_index=0, t_start=0, t_end=200)