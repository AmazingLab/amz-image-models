from torch import nn

from aim.engine import recursive_build


def build_model(cfg: dict, registry: dict = None) -> nn.Module:
    return recursive_build(cfg, registry)
