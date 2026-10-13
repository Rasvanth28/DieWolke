import inspect

from .dsen2cr import DSen2CR
from .reconstructor import BridgeNet
from .unet import Masker, UNet

MODELS = {"bridge": BridgeNet, "masker": Masker, "dsen2cr": DSen2CR}


def build_model(cfg):
    """Instantiate cfg['model']['type'] with the keys its constructor accepts.

    Configs inherit the bridge's model keys from base.yaml; keys a model does not take
    (and pipeline-only keys such as mask_source) are ignored.
    """
    m = dict(cfg["model"])
    cls = MODELS[m.pop("type")]
    params = inspect.signature(cls.__init__).parameters
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in m.items() if k in params}
    return cls(**kwargs)


__all__ = ["BridgeNet", "DSen2CR", "Masker", "UNet", "build_model", "MODELS"]
