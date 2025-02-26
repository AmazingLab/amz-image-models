from torch import optim

from aim.engine import recursive_build


def build_optimizer(cfg, params) -> optim.Optimizer:
    cfg['params'] = params
    return recursive_build(cfg)
