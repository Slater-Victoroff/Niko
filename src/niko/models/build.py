from encoders.param_encoder import RBParamEncoder, MultiParamEncoder
from encoders.context_cond import ContextCondEncoder, PooledContextCondEncoder, LatentContextCondEncoder

from encoders.sequence_conv import SequenceConvEncoder

from operators.linear_local import LinearLocalOperator
from operators.transport_operator import (
    TransportOperator,
    AdvectionDiffusionOperator,
    HelmholtzTransportOperator,
    LocalAttentionTransportOperator,
    FiLMAdvectionDiffusionOperator,
    FiLMHelmholtzTransportOperator,
    FiLMLocalAttentionTransportOperator,
)
from operators.direct_field_operator import DirectFieldOperator

from decoders.shared_conv import SharedConvDecoder
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder

from models.dynamics_model import LatentDynamicsModel
from models.direct_model import DirectFieldModel


def build_param_encoder(cfg):
    name = cfg.pop("name")
    kwargs = dict(cfg)

    param_encoder_map = {
        "rb_param": RBParamEncoder,
        "multi_param": MultiParamEncoder,
    }

    cls = param_encoder_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown param encoder: {name}")
    return cls(**kwargs)


def build_context_cond_encoder(cfg):
    kwargs = dict(cfg)
    name = kwargs.pop("name", "stacked")

    context_cond_encoder_map = {
        "stacked": ContextCondEncoder,
        "pooled": PooledContextCondEncoder,
        "latent_pooled": LatentContextCondEncoder,
    }

    cls = context_cond_encoder_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown context_cond_encoder: {name}")
    return cls(**kwargs)


def build_encoder(cfg):
    name = cfg.pop("name")
    kwargs = dict(cfg)

    encoder_map = {
        "sequence_conv": SequenceConvEncoder,
    }

    cls = encoder_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown encoder: {name}")
    return cls(**kwargs)


def build_operator(cfg):
    name = cfg.pop("name")
    kwargs = dict(cfg)

    operator_map = {
        "linear_local": LinearLocalOperator,
        "transport": TransportOperator,
        "advection_diffusion": AdvectionDiffusionOperator,
        "helmholtz_transport": HelmholtzTransportOperator,
        "local_attention_transport": LocalAttentionTransportOperator,
        "film_advection_diffusion": FiLMAdvectionDiffusionOperator,
        "film_helmholtz": FiLMHelmholtzTransportOperator,
        "film_local_attention": FiLMLocalAttentionTransportOperator,
    }

    cls = operator_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown operator: {name}")
    return cls(**kwargs)


def build_decoder(cfg):
    name = cfg.pop("name")
    kwargs = dict(cfg)

    decoder_map = {
        "shared_conv": SharedConvDecoder,
        "shared_heads": SharedTrunkFieldHeadsDecoder,
    }

    cls = decoder_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown decoder: {name}")
    return cls(**kwargs)


def build_direct_model(cfg):
    param_encoder = build_param_encoder(cfg["param_encoder"])

    op_cfg = dict(cfg["operator"])
    op_cfg.pop("name")
    operator = DirectFieldOperator(**op_cfg)

    return DirectFieldModel(operator=operator, param_encoder=param_encoder)


def build_model(cfg):
    if cfg.get("model_type") == "direct":
        return build_direct_model(cfg)

    # context_cond_encoder (cond inferred from x_context) and param_encoder (cond from
    # ground-truth Params) are mutually exclusive -- see LatentDynamicsModel's own check.
    param_encoder = None
    context_cond_encoder = None
    if "context_cond_encoder" in cfg:
        context_cond_encoder = build_context_cond_encoder(cfg["context_cond_encoder"])
    else:
        param_encoder = build_param_encoder(cfg["param_encoder"])

    encoder = build_encoder(cfg["encoder"])
    operator = build_operator(cfg["operator"])
    decoder = build_decoder(cfg["decoder"])

    model_cfg = cfg.get("model", {})

    return LatentDynamicsModel(
        param_encoder=param_encoder,
        context_cond_encoder=context_cond_encoder,
        encoder=encoder,
        operator=operator,
        decoder=decoder,
        **model_cfg,
    )
