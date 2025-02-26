from aim.engine import recursive_build


def build_model(cfg: dict):
    return recursive_build(cfg)
