import importlib


def build(name: str, *args, **kwargs):
    path_and_name = name.rsplit(".", 1)
    if len(path_and_name) == 1:
        module_path, class_name = '__main__', name
    else:
        module_path, class_name = path_and_name
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(*args, **kwargs)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Failed to import {name}: {str(e)}")


def build_from_register(name: str, registry: dict, *args, restrict: bool = True, **kwargs):
    if name not in registry:
        if restrict:
            raise ValueError(f"Model {name} not found in registry!")
        else:
            print(f"Warning {name} is not registered, use unsafe build.")
            return build(name, *args, **kwargs)

    _class = registry[name]
    assert callable(_class), "Class is not callable, build failed."
    return _class(*args, **kwargs)
