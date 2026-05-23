from encoders.param_encoder import RBParamEncoder

from encoders.sequence_conv import SequenceConvEncoder

from operators.linear_local import LinearLocalOperator
from operators.advection_diffusion import AdvectionDiffusionOperator
from operators.helmholtz_transport import HelmholtzTransportOperator

from decoders.shared_conv import SharedConvDecoder
from decoders.shared_heads import SharedTrunkFieldHeadsDecoder

from models.dynamics_model import LatentDynamicsModel


def build_param_encoder(cfg):
    name = cfg.pop("name")
    kwargs = dict(cfg)

    param_encoder_map = {
        "rb_param": RBParamEncoder,
    }

    cls = param_encoder_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown param encoder: {name}")
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
        "advection_diffusion": AdvectionDiffusionOperator,
        "helmholtz_transport": HelmholtzTransportOperator,
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


def build_model(cfg):
    param_encoder = build_param_encoder(cfg["param_encoder"])
    encoder = build_encoder(cfg["encoder"])
    operator = build_operator(cfg["operator"])
    decoder = build_decoder(cfg["decoder"])

    model_cfg = cfg.get("model", {})

    return LatentDynamicsModel(
        param_encoder=param_encoder,
        encoder=encoder,
        operator=operator,
        decoder=decoder,
        **model_cfg,
    )
