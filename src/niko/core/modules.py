from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .states import LatentState, Params
from functools import wraps


class ValidatedModule(nn.Module):
    """Mixin that auto-wraps subclass `forward` with `validate_call`.

    Subclasses (typically ABC base classes) should inherit from this mixin so
    concrete subclasses automatically get their `forward` validated.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        f = cls.__dict__.get("forward", None)
        if f is not None and not getattr(f, "_validation_wrapped", False):
            wrapped = validate_call(f)
            setattr(wrapped, "_validation_wrapped", True)
            setattr(cls, "forward", wrapped)


def validate_call(fn):
    """Decorator that calls `self.validate_input(...)` with the same
    arguments passed to the wrapped function before invoking it.

    This lets each concrete class implement a `validate_input(...)` method
    (taking the same positional arguments as `forward`) and ensures validation
    runs consistently for encoders, operators, decoders, and param encoders.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        validator = getattr(self, "validate_input", None)
        if callable(validator):
            # Pass through the same args/kwargs the forward will receive.
            validator(*args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper



class ParamEncoderBase(ValidatedModule, nn.Module, ABC):
    @abstractmethod
    def forward(self, params: Optional[Params] = None) -> Tensor:
        raise NotImplementedError


class EncoderBase(ValidatedModule, nn.Module, ABC):
    @abstractmethod
    def forward(self, x_context: Tensor, cond: Optional[Tensor] = None) -> LatentState:
        raise NotImplementedError

    def validate_input(
        self,
        x_context: Tensor,
        cond: Optional[Tensor] = None,
    ) -> None:
        context_frames = getattr(self, "context_frames", None)
        in_channels = getattr(self, "in_channels", None)
        cond_dim = getattr(self, "cond_dim", None)
        assert x_context.ndim == 5, f"Expected x_context to have 5 dimensions [B, T_context, C, H, W], got {x_context.ndim}"

        b, t, c, h, w = x_context.shape
        assert t == context_frames, f"Expected {context_frames} context frames, got {t}"
        assert c == in_channels, f"Expected {in_channels} channels, got {c}"

        if cond is not None:
            if cond_dim is None or cond_dim == 0:
                raise ValueError(f"Encoder expects no conditioning, but cond was provided with shape {cond.shape}")
            assert cond.shape[0] == b, f"Expected batch size {b}, got {cond_dim[0]}"
            assert cond.shape[1] == cond_dim, f"Expected cond_dim {cond_dim}, got {cond.shape[1]}"


class OperatorBase(ValidatedModule, nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        z: LatentState,
        cond: Optional[Tensor] = None,
        bc_weights: Optional[Tuple[Tensor, Tensor]] = None,
        dt: float = 1.0,
    ) -> LatentState:
        raise NotImplementedError

    def validate_input(
        self, z: LatentState, cond: Optional[Tensor] = None,
        bc_weights: Optional[Tuple[Tensor, Tensor]] = None, dt: float = 1.0,
    ) -> None:
        assert isinstance(z, LatentState), f"Expected z to be a LatentState, got {type(z)}"


class DecoderBase(ValidatedModule, nn.Module, ABC):
    @abstractmethod
    def forward(
        self, z: Tensor | LatentState, cond: Optional[Tensor] = None,
        bc_weights: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        raise NotImplementedError
