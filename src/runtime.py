import os
from typing import Optional

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from dataloader import create_param_dataloaders
from niko import NikoBlock, vrmse


def load_model(checkpoint_path: str, device: str, context_frames: int, predict_frames: int, num_params: int):
	dev = torch.device(device)
	model = NikoBlock(
		target_shape=(4, 128, 512),
		k=2,
		core_ranks=(4, 50, 100, 2),
		h_dim=64,
		context_len=context_frames,
		pred_len=predict_frames,
		num_params=num_params,
		device=device,
	)

	state = torch.load(checkpoint_path, map_location=dev)
	model.load_state_dict(state)
	model.to(dev)
	model.eval()
	return model


def evaluate(
	model: torch.nn.Module,
	val_loader,
	device: str,
	predict_frames: int,
	num_params: int,
	save_examples: Optional[str] = None,
	visualize_examples: Optional[str] = None,
	log_interval: int = 20,
	verbose: bool = True,
):
	dev = torch.device(device)
	total = 0.0
	batches = 0
	examples = []

	total_batches = None
	try:
		total_batches = len(val_loader)
	except Exception:
		total_batches = None

	start_time = time.time()

	with torch.no_grad():
		for batch in val_loader:
			batch_start = time.time()

			# batch may be (ctx, tgt) or (ctx, tgt, params)
			if len(batch) == 3:
				ctx, tgt, params = batch
			else:
				ctx, tgt = batch
				params = None

			ctx = ctx.to(dev)
			if predict_frames == 1:
				tgt = tgt.to(dev).squeeze(1)
			else:
				tgt = tgt.to(dev)

			batch_size = ctx.shape[0]
			if params is None:
				params = torch.zeros((batch_size, num_params), dtype=torch.float32, device=dev)
			else:
				params = params.to(device=dev, dtype=torch.float32)

			pred = model(ctx, params)
			loss = vrmse(pred, tgt)
			total += float(loss.item())
			batches += 1

			if (save_examples is not None or visualize_examples is not None) and len(examples) < 5:
				# capture base frame (last context frame), prediction and target on CPU
				base = ctx[:, -1, ...].detach().cpu()
				examples.append((base, pred.cpu(), tgt.cpu()))

			batch_end = time.time()
			if verbose and (batches % log_interval == 0 or (total_batches is not None and batches == total_batches)):
				elapsed = batch_end - start_time
				avg_loss = total / batches if batches > 0 else float('inf')
				avg_batch_time = elapsed / batches if batches > 0 else 0.0
				eta = None
				if total_batches is not None:
					remaining = max(total_batches - batches, 0)
					eta = remaining * avg_batch_time

				mem_info = ""
				if dev.type == 'cuda' and torch.cuda.is_available():
					try:
						allocated = torch.cuda.memory_allocated(dev) / 1024 ** 2
						reserved = torch.cuda.memory_reserved(dev) / 1024 ** 2
						mem_info = f" | GPU mem alloc: {allocated:.1f}MB reserved: {reserved:.1f}MB"
					except Exception:
						mem_info = ""

				if total_batches is not None:
					eta_str = f", ETA: {eta:.1f}s" if eta is not None else ""
					print(
						f"Eval batch {batches}/{total_batches} | avg_loss: {avg_loss:.6f} | last_loss: {loss.item():.6f} | elapsed: {elapsed:.1f}s{eta_str}{mem_info}"
					)
				else:
					print(
						f"Eval batch {batches} | avg_loss: {avg_loss:.6f} | last_loss: {loss.item():.6f} | elapsed: {elapsed:.1f}s{mem_info}"
					)

	avg = total / batches if batches > 0 else float('inf')

	if save_examples is not None:
		os.makedirs(save_examples, exist_ok=True)
		for i, (_base, p, t) in enumerate(examples):
			torch.save({'pred': p, 'target': t, 'base': _base}, os.path.join(save_examples, f'example_{i}.pt'))

	if visualize_examples is not None:
		os.makedirs(visualize_examples, exist_ok=True)
		for i, (base, p, t) in enumerate(examples):
			# base, p, t shapes: (B, C, H, W) or (B, CH, H, W) where B==batch size saved
			# we'll visualize the first item in the batch
			b_base = base[0]
			b_pred = p[0]
			b_tgt = t[0]

			def to_magnitude(x):
				# x: (C, H, W) - take first two channels as vector components
				arr = x.numpy()
				c = arr.shape[0]
				if c >= 2:
					u = arr[0]
					v = arr[1]
					mag = np.sqrt(u ** 2 + v ** 2)
				else:
					mag = arr[0]
				return mag

			mag_base = to_magnitude(b_base)
			mag_tgt = to_magnitude(b_tgt)
			mag_pred = to_magnitude(b_pred)
			mag_res = np.abs(mag_tgt - mag_pred)

			fig, axes = plt.subplots(2, 2, figsize=(10, 8))
			im0 = axes[0, 0].imshow(mag_base, cmap='viridis')
			axes[0, 0].set_title('Base (last context) magnitude')
			fig.colorbar(im0, ax=axes[0, 0])

			im1 = axes[0, 1].imshow(mag_tgt, cmap='viridis')
			axes[0, 1].set_title('Ground Truth magnitude')
			fig.colorbar(im1, ax=axes[0, 1])

			im2 = axes[1, 0].imshow(mag_pred, cmap='viridis')
			axes[1, 0].set_title('Prediction magnitude')
			fig.colorbar(im2, ax=axes[1, 0])

			im3 = axes[1, 1].imshow(mag_res, cmap='magma')
			axes[1, 1].set_title('Residual magnitude |GT - Pred|')
			fig.colorbar(im3, ax=axes[1, 1])

			plt.tight_layout()
			out_path = os.path.join(visualize_examples, f'viz_{i}.png')
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

	return avg


def main():
	# Configuration - edit these variables before running the file directly
	DEVICE = "cuda:0"
	DATA_DIR = "/app/data/datasets/rayleigh_benard/data"
	CHECKPOINT = "/app/src/model_epoch_1.pt"
	BATCH_SIZE = 4
	CONTEXT_FRAMES = 4
	PREDICT_FRAMES = 1
	NUM_PARAMS = 2
	NUM_WORKERS = 2
	VAL_FILE_LIMIT = None
	SAVE_EXAMPLES = None  # set to a path to save examples, e.g. '/tmp/niko_examples'

	# evaluate on the test split (set TEST_SUBDIR to your test folder name)
	TEST_SUBDIR = "valid"
	print(f"Loading test dataloader from {os.path.join(DATA_DIR, TEST_SUBDIR)}")
	train_loader, val_loader, chosen = create_param_dataloaders(
		DATA_DIR,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		stack_frames=True,
		context_frames=CONTEXT_FRAMES,
		predict_frames=PREDICT_FRAMES,
		train_file_limit=None,
		val_file_limit=VAL_FILE_LIMIT,
		device=DEVICE,
		train_subdir=TEST_SUBDIR,
		valid_subdir=TEST_SUBDIR,
		shuffle_train=False,
		shuffle_val=True,
	)

	print(f"Constructing model and loading checkpoint {CHECKPOINT}")
	model = load_model(CHECKPOINT, DEVICE, CONTEXT_FRAMES, PREDICT_FRAMES, NUM_PARAMS)

	# Instead of running full validation here, produce a few sample outputs
	# and write them to a durable folder. The full `evaluate` function above
	# is left intact for manual/CI use, but we don't call it by default.
	SAMPLE_OUTPUT_DIR = os.environ.get('SAMPLE_OUTPUT_DIR', '/app/niko_samples')
	NUM_SAMPLES = 4

	print("Generating a small set of sample outputs from the validation loader...")

	def save_sample_outputs(model, val_loader, device, num_params, out_dir, num_samples=1, predict_frames=1):
		dev = torch.device(device)
		os.makedirs(out_dir, exist_ok=True)
		saved = 0
		with torch.no_grad():
			for batch in val_loader:
				if len(batch) == 3:
					ctx, tgt, params = batch
				else:
					ctx, tgt = batch
					params = None

				ctx = ctx.to(dev)
				batch_size = ctx.shape[0]
				if params is None:
					params = torch.zeros((batch_size, num_params), dtype=torch.float32, device=dev)
				else:
					params = params.to(device=dev, dtype=torch.float32)

				pred = model(ctx, params)

				# Prepare CPU tensors and visualize the first item in the batch
				base = ctx[:, -1, ...].detach().cpu()
				tgt_cpu = tgt.detach().cpu()
				pred_cpu = pred.detach().cpu()

				# If predict_frames==1 the dataloader returns a time dim for tgt (B, P, C, H, W)
				# squeeze that axis so we consistently work with (B, C, H, W)
				if predict_frames == 1:
					if tgt_cpu.ndim == 5 and tgt_cpu.shape[1] == 1:
						tgt_cpu = tgt_cpu.squeeze(1)
					if pred_cpu.ndim == 5 and pred_cpu.shape[1] == 1:
						pred_cpu = pred_cpu.squeeze(1)

				b_base = base[0]
				b_pred = pred_cpu[0]
				b_tgt = tgt_cpu[0]

				# Save per-channel grayscale 2x2 plots: Base | Ground Truth / Prediction | Residual
				base_arr = np.asarray(b_base)
				pred_arr = np.asarray(b_pred)
				tgt_arr = np.asarray(b_tgt)

				def extract_channel(arr, c):
					arr = np.asarray(arr)
					# If array has explicit channel dim (C, H, W), return channel c if present
					if arr.ndim == 3:
						if c < arr.shape[0]:
							return arr[c]
						# missing channel -> zeros of spatial shape
						return np.zeros_like(arr[0])
					# If array is 2D (H, W) treat it as single-channel: return data for c==0, zeros otherwise
					if arr.ndim == 2:
						if c == 0:
							return arr
						return np.zeros_like(arr)
					# fallback: try to squeeze to 2D and behave similarly
					s = np.squeeze(arr)
					if s.ndim == 2:
						if c == 0:
							return s
						return np.zeros_like(s)
					# last resort: return zeros
					return np.zeros((1, 1))

				c_dim = max(
					base_arr.shape[0] if base_arr.ndim == 3 else 1,
					pred_arr.shape[0] if pred_arr.ndim == 3 else 1,
					tgt_arr.shape[0] if tgt_arr.ndim == 3 else 1,
				)
				def ensure_2d(img):
					arr = np.asarray(img)
					# If already 2D, return as-is
					if arr.ndim == 2:
						return arr
					# If 3D, it should be (C, H, W) already reduced by extract_channel, but handle gracefully
					if arr.ndim == 3:
						# prefer first spatial slice if some extra dim exists
						return np.squeeze(arr)
					# fallback: squeeze to 2D
					return np.squeeze(arr)

				for c in range(c_dim):
					base_ch = ensure_2d(extract_channel(base_arr, c)).astype(np.float32)
					tgt_ch = ensure_2d(extract_channel(tgt_arr, c)).astype(np.float32)
					pred_ch = ensure_2d(extract_channel(pred_arr, c)).astype(np.float32)
					res_ch = np.abs(tgt_ch - pred_ch).astype(np.float32)

					# Normalize grayscale channels together for fair comparison
					eps = 1e-8
					vmin = float(min(base_ch.min(), tgt_ch.min(), pred_ch.min()))
					vmax = float(max(base_ch.max(), tgt_ch.max(), pred_ch.max()))
					if vmax - vmin < eps:
						vmax = vmin + 1.0
					base_n = (base_ch - vmin) / (vmax - vmin)
					gt_n = (tgt_ch - vmin) / (vmax - vmin)
					pred_n = (pred_ch - vmin) / (vmax - vmin)

					# Normalize residual to its own max for visibility
					res_max = float(res_ch.max())
					if res_max < eps:
						res_max = 1.0
					res_n = res_ch / res_max

					# Convert grayscale to RGB
					rgb_base = np.stack([base_n, base_n, base_n], axis=-1)
					rgb_pred = np.stack([pred_n, pred_n, pred_n], axis=-1)
					rgb_gt = np.stack([gt_n, gt_n, gt_n], axis=-1)
					# Apply magma colormap to residual (returns RGBA)
					cmap = plt.get_cmap('magma')
					rgb_res = cmap(res_n)[..., :3]

					# Create mosaic: top row [base | pred], bottom row [gt | res]
					h, w = base_n.shape
					row_top = np.concatenate([rgb_base, rgb_pred], axis=1)
					row_bot = np.concatenate([rgb_gt, rgb_res], axis=1)
					mosaic = np.concatenate([row_top, row_bot], axis=0)

					# Save mosaic to disk with labels drawn via matplotlib (no axes, no padding)
					fig = plt.figure(frameon=False, figsize=(mosaic.shape[1] / 150.0, mosaic.shape[0] / 150.0), dpi=150)
					ax = fig.add_axes([0, 0, 1, 1])
					ax.imshow(mosaic, interpolation='nearest')
					# Add small white labels with black stroke for readability
					label_kw = dict(color='white', fontsize=10, weight='bold', va='top')
					ax.text(5, 5, 'Base', **label_kw)
					ax.text(w + 5, 5, 'Prediction', **label_kw)
					ax.text(5, h + 5, 'Ground Truth', **label_kw)
					ax.text(w + 5, h + 5, 'Residual', **label_kw)
					ax.axis('off')
					out_path = os.path.join(out_dir, f'sample_{saved}_ch{c}.png')
					fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
					plt.close(fig)

				saved += 1
				if saved >= num_samples:
					break
		return saved

	saved_count = save_sample_outputs(model, val_loader, DEVICE, NUM_PARAMS, SAMPLE_OUTPUT_DIR, num_samples=NUM_SAMPLES, predict_frames=PREDICT_FRAMES)
	print(f"Saved {saved_count} sample(s) to {SAMPLE_OUTPUT_DIR}")


if __name__ == '__main__':
	main()

