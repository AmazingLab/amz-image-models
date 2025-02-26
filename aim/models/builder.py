from torch import nn

from aim.engine import recursive_build


def build_model(cfg: dict) -> nn.Module:
    return recursive_build(cfg)
