from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.states import Params
from core.modules import ParamEncoderBase
from operators.direct_field_operator import DirectFieldOperator


class DirectFieldModel(nn.Module):
    """Operator-only dynamics model with no encoder/decoder latent bottleneck.

    Interface-compatible with LatentDynamicsModel.forward() so train.py works unchanged.
    Takes the last context frame and rolls it forward `steps` times in physical space.
    """

    def __init__(
        self,
        operator: DirectFieldOperator,
        param_encoder: Optional[ParamEncoderBase] = None,
    ):
        super().__init__()
        self.operator = operator
        self.param_encoder = param_encoder

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if param_encoder is not None:
            pe_total = sum(p.numel() for p in param_encoder.parameters())
            pe_train = sum(p.numel() for p in param_encoder.parameters() if p.requires_grad)
            print(f"param_encoder params: total={pe_total}, trainable={pe_train}")
        op_total = sum(p.numel() for p in operator.parameters())
        op_train = sum(p.numel() for p in operator.parameters() if p.requires_grad)
        print(f"operator params: total={op_total}, trainable={op_train}")
        print(f"DirectFieldModel params: total={total}, trainable={trainable}")

    def forward(
        self,
        x_context: Tensor,
        steps: int,
        params: Optional[Params] = None,
        return_initial_encode: bool = False,
        debug_timing: bool = False,
    ) -> Tensor:
        cond = self.param_encoder(params) if self.param_encoder is not None else None

        x = x_context[:, -1]  # [B, C, H, W] — last context frame
        preds = []
        for _ in range(steps):
            x = self.operator(x, cond=cond)
            preds.append(x)

        out = torch.stack(preds, dim=1)  # [B, steps, C, H, W]

        if return_initial_encode:
            # No encode/decode round-trip — return the last context frame as anchor.
            # Anchor loss is trivially zero; set anchor_target: false in the config.
            anchor = x_context[:, -1]
            return anchor, out
        return out
