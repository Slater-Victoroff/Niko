from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from contextlib import contextmanager
import time

from core.modules import ParamEncoderBase, EncoderBase, OperatorBase, DecoderBase
from core.states import LatentState, Params
import logging


logger = logging.getLogger(__name__)


def _tensor_summary(name: str, t: Optional[Tensor]) -> str:
    if t is None:
        return f"{name}: None"
    try:
        # reduce stats to scalars
        nm = torch.isnan(t).any().item()
        inf = torch.isinf(t).any().item()
        finite = torch.isfinite(t).all().item()
        t_min = float(t.min())
        t_max = float(t.max())
        t_mean = float(t.mean())
        return (
            f"{name}: shape={tuple(t.shape)} device={t.device} dtype={t.dtype} "
            f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} "
            f"has_nan={nm} has_inf={inf} all_finite={finite}"
        )
    except Exception as e:
        return f"{name}: (could not summarize: {e})"


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        param_encoder: ParamEncoderBase,
        encoder: EncoderBase,
        operator: OperatorBase,
        decoder: DecoderBase,
    ):
        super().__init__()
        self.param_encoder = param_encoder
        self.encoder = encoder
        self.operator = operator
        self.decoder = decoder

        # log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print overall and per-component parameter counts
        component_names = ["param_encoder", "encoder", "operator", "decoder"]
        for name in component_names:
            m = getattr(self, name, None)
            if m is None:
                print(f"{name} params: total=0, trainable=0")
            else:
                total_c = sum(p.numel() for p in m.parameters())
                trainable_c = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"{name} params: total={total_c}, trainable={trainable_c}")

        print(f"LatentDynamicsModel params: total={total_params}, trainable={trainable_params}")

    def rollout_latent(
        self,
        z0: LatentState,
        steps: int,
        cond: Optional[Tensor] = None,
    ) -> List[LatentState]:
        preds = []
        z = z0
        for i in range(steps):
            z = self.operator(z, cond=cond)
            preds.append(z)
        return preds

    def forward(
        self,
        x_context: Tensor,
        steps: int,
        params: Optional[Params] = None,
        return_initial_encode: bool = True,
        debug_timing: bool = False,
    ) -> Tensor:
        @contextmanager
        def time_block(timings: dict, key: str):
            if debug_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if debug_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings[key] = time.perf_counter() - t0

        cond = self.param_encoder(params)
        timings: dict[str, float] = {}

        # encoder
        with time_block(timings, "encoder"):
            if hasattr(self.encoder, "cond_dim") and self.encoder.cond_dim is not None:
                assert cond.shape[1] == self.encoder.cond_dim, f"param_encoder output dim {cond.shape[1]} does not match encoder cond_dim {self.encoder.cond_dim}"
                z0 = self.encoder(x_context, cond=cond)
            else:
                z0 = self.encoder(x_context)

        # rollout
        with time_block(timings, "rollout"):
            zs = self.rollout_latent(z0, steps=steps, cond=cond)

        # decode each latent into a raw tensor [B, C, H, W]
        with time_block(timings, "decoder"):
            # Stack latent states and decode in one batch
            zs_stacked = torch.stack([z.grid for z in zs], dim=1)
            # Merge batch and steps dimensions for decoder: [B, T, C, H, W] -> [B*T, C, H, W]
            B, T = zs_stacked.shape[:2]
            zs_flat = zs_stacked.reshape(B * T, *zs_stacked.shape[2:])
            cond_flat = cond[:, None, :].expand(B, T, cond.shape[-1]).reshape(B * T, cond.shape[-1]) if cond is not None else None
            response_flat = self.decoder(zs_flat, cond=cond_flat)
            # Restore original batch and steps dimensions: [B*T, ...] -> [B, T, ...]
            response = response_flat.reshape(B, T, *response_flat.shape[1:])

        if debug_timing:
            total = sum(timings.values())
            print(
                f"Timing (s) encoder={timings.get('encoder', 0.0):0.4f} "
                f"rollout={timings.get('rollout', 0.0):0.4f} "
                f"decoder={timings.get('decoder', 0.0):0.4f} "
                f"total={total:0.4f}"
            )

        if return_initial_encode:
            decoded_last_frame = self.decoder(z0.grid, cond=cond)
            return decoded_last_frame, response
        return response