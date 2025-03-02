from importlib.util import spec_from_file_location, module_from_spec
import sys


def load_config(config_path, module_name="_default_config"):
    spec = spec_from_file_location(module_name, config_path)
    cfg = module_from_spec(spec)
    sys.modules[module_name] = cfg
    spec.loader.exec_module(cfg)
    return cfg
