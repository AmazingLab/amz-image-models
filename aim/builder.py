import importlib
import warnings


def build(name: str, *args, **kwargs):
    path_and_name = name.rsplit(".", 1)
    if len(path_and_name) == 1:
        module_path, class_name = '__main__', name
    else:
        module_path, class_name = path_and_name
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(*args, **kwargs)


def build_from_register(name: str, registry: dict, *args, restrict: bool = True, **kwargs):
    if name not in registry:
        if restrict:
            raise ValueError(f"Class `{name}` not found in registry!")
        else:
            warnings.warn(f"Warning: `{name}` is not registered; building from installed modules.")
            return build(name, *args, **kwargs)

    _class = registry[name]
    assert callable(_class), "Registered object is not callable, build failed."
    return _class(*args, **kwargs)
