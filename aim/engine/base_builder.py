import importlib
from typing import Union, Callable


def resolve_dict(cfg: dict):
    assert 'type' in cfg
    _cfg = cfg.copy()
    type_name = _cfg.pop('type')
    args = _cfg.pop('args') if 'args' in _cfg else ()
    kwargs = _cfg
    return type_name, args, kwargs


def build(type_name: Union[str, Callable], *args, **kwargs):
    if callable(type_name):
        return type_name(*args, **kwargs)
    path_and_name = type_name.rsplit(".", 1)
    if len(path_and_name) == 1:
        module_path, class_name = '__main__', type_name
    else:
        module_path, class_name = path_and_name
    module = importlib.import_module(module_path)
    try:
        obj = getattr(module, class_name)(*args, **kwargs)
    except Exception as e:
        raise e
    return obj


def recursive_build(cfg):
    """
    # Demo 1: build pytorch model with module name
    >>> model_cfg = {
    >>>     'type': 'torch.nn.Sequential',
    >>>     'args': [
    >>>         {'type': 'torch.nn.Linear', 'in_features': 16, 'out_features': 32},
    >>>         {'type': 'torch.nn.Linear', 'in_features': 32, 'out_features': 32},
    >>>         {'type': 'torch.nn.Linear', 'in_features': 32, 'out_features': 32},
    >>>         {'type': 'torch.nn.Linear', 'in_features': 32, 'out_features': 16},
    >>>     ]
    >>> }
    >>> model = recursive_build(cfg)
    >>> print(model)
    # Demo 2: build pytorch model with module
    >>> from torch.nn import Linear, Sequential
    >>> model_cfg = {
    >>>     'type': Sequential,
    >>>     'args': [
    >>>         {'type': Linear, 'in_features': 16, 'out_features': 32},
    >>>         {
    >>>             'type': Sequential,
    >>>             'args': [
    >>>                 {'type': Linear, 'in_features': 32, 'out_features': 32},
    >>>                 {'type': Linear, 'in_features': 32, 'out_features': 32},
    >>>             ]
    >>>         },
    >>>         {'type': Linear, 'in_features': 32, 'out_features': 16},
    >>>     ]
    >>> }
    >>> model = recursive_build(cfg)
    >>> print(model)
    # Demo 2: build pytorch transforms with module name
    >>> transform_cfg = {
    >>>     'type': 'torchvision.transforms.Compose',
    >>>     'transforms': [
    >>>         {'type': 'torchvision.transforms.ToTensor'},
    >>>         {
    >>>             'type': 'torchvision.transforms.Normalize',
    >>>             'mean': [0.485, 0.456, 0.406],
    >>>             'std': [0.229, 0.224, 0.225]
    >>>         }
    >>>     ]
    >>> }
    >>> transform = recursive_build(transform_cfg)
    >>> print(transform)
    """
    if not isinstance(cfg, (dict, list)):
        return cfg
    for k, v in cfg.items():
        if isinstance(v, dict):
            if 'type' in v:
                cfg[k] = recursive_build(v)
        if isinstance(v, list):
            if isinstance(v[0], dict) and 'type' in v[0]:
                cfg[k] = [recursive_build(item) for item in v]

    # print(cfg)
    type_name, args, kwargs = resolve_dict(cfg)
    return build(type_name, *args, **kwargs)
