from aim.engine import recursive_build


def build_optimizer(cfg, params):
    cfg['params'] = params
    return recursive_build(cfg)
