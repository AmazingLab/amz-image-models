from typing import TypeAlias, Union, Iterable, Dict, Any

import torch
from torch import optim

from aim.engine import recursive_build

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


def build_optimizer(cfg, registry: dict = None) -> optim.Optimizer:
    return recursive_build(cfg, registry)
