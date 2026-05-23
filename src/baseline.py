import torch
import torch.nn as nn
import math

from dataloader import create_param_dataloaders


class ConvNeXtBlock(nn.Module):
	def __init__(self, dim, device='cuda'):
		super().__init__()
		self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim).to(device)
		self.norm = nn.LayerNorm(dim, eps=1e-6).to(device)
		self.pw_conv1 = nn.Linear(dim, 4 * dim).to(device)
		self.act = nn.GELU()
		self.pw_conv2 = nn.Linear(4 * dim, dim).to(device)

		nn.init.zeros_(self.pw_conv2.weight)
		nn.init.zeros_(self.pw_conv2.bias)

	def forward(self, x):
		identity = x
		x = self.dw_conv(x)
		x = x.permute(0, 2, 3, 1)
		x = self.norm(x)
		x = self.pw_conv1(x)
		x = self.act(x)
		x = self.pw_conv2(x)
		x = x.permute(0, 3, 1, 2)
		return identity + x


class BasicUpres(nn.Module):
	def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
		super().__init__()
		self.k = k

		self.project_in = nn.Conv2d(in_channels, hidden, kernel_size=1).to(device)
		self.convnext = ConvNeXtBlock(hidden, device=device)
		self.project_out = nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1).to(device)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=k)

		for m in [self.project_in, self.project_out]:
			nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x = self.project_in(x)
		x = self.convnext(x)
		x = self.project_out(x)
		return self.pixel_shuffle(x)


def ConvNeXTDecoder(k, hdim):
	"""
	Convenience builder that returns a BasicUpres decoder.
	Assumes input channels == `hdim` and output is 3-channel RGB.
	"""
	return BasicUpres(in_channels=hdim, out_channels=3, hidden=hdim, k=k)


class LastFrameBaseline(nn.Module):
	"""Baseline that predicts the last context frame unchanged.

	forward(context_frames, params=None) -> Tensor
	- expects `context_frames` shaped (B, T, C, H, W) and returns (B, C, H, W)
	"""
	def __init__(self):
		super().__init__()

	def forward(self, context_frames, params=None):
		if not isinstance(context_frames, torch.Tensor):
			raise TypeError("context_frames must be a torch.Tensor")
		if context_frames.dim() == 5:
			return context_frames[:, -1, ...]
		if context_frames.dim() == 4:
			return context_frames
		raise ValueError(f"Unexpected context_frames shape: {context_frames.shape}")


class MeanFrameBaseline(nn.Module):
	"""Baseline that predicts the mean across the temporal context.

	If `spatial_mean=True` the predictor computes the per-channel scalar mean
	across time and spatial dims and returns a constant image (B, C, H, W).
	Otherwise it returns the pixelwise temporal mean (B, C, H, W).
	"""
	def __init__(self, spatial_mean: bool = False):
		super().__init__()
		self.spatial_mean = bool(spatial_mean)

	def forward(self, context_frames, params=None):
		if not isinstance(context_frames, torch.Tensor):
			raise TypeError("context_frames must be a torch.Tensor")
		if context_frames.dim() == 5:
			# (B, T, C, H, W)
			tmean = context_frames.mean(dim=1)  # (B, C, H, W)
		elif context_frames.dim() == 4:
			# assume already (B, C, H, W)
			tmean = context_frames
		else:
			raise ValueError(f"Unexpected context_frames shape: {context_frames.shape}")

		if self.spatial_mean:
			# compute per-sample, per-channel scalar and expand to full frame
			# mean over H,W -> (B, C, 1, 1)
			m = tmean.mean(dim=(-2, -1), keepdim=True)
			return m.expand_as(tmean)
		return tmean


class NikoBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
        super().__init__()
        self.k = k

        self.project_in = nn.Conv2d(in_channels, hidden, kernel_size=1).to(device)
        self.convnext = ConvNeXtBlock(hidden, device=device)
        self.project_out = nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1).to(device)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=k)

        for m in [self.project_in, self.project_out]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.convnext(x)
        x = self.project_out(x)
        return self.pixel_shuffle(x)
