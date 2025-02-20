import importlib

_registry = {}


def get_class_reference(cls):
    module = cls.__module__
    if module == "__main__":
        return cls.__name__
    else:
        f"{module}.{cls.__name__}"
    return


def auto_register(name: str = None):
    def decorator(cls):
        _name = name if name else get_class_reference(cls)
        if _name in _registry:
            raise ValueError(f"`{_name}` is already registered!")
        _registry[_name] = cls
        return cls

    return decorator


def register(cls, name: str = None, override: bool = False):
    _name = name if name else get_class_reference(cls)
    if not override and _name in _registry:
        raise ValueError(f"`{_name}` is already registered!")
    _registry[_name] = cls
    return cls


def build(name: str, *args, restrict: bool = True, **kwargs):
    if name not in _registry:
        if restrict:
            raise ValueError(f"Model {name} not found in registry!")
        else:
            print(f"Warning {name} is not registered, use unsafe build.")
            return unsafe_build(name, *args, **kwargs)

    _class = _registry[name]
    assert callable(_class), "Class is not callable, build failed."
    return _class(*args, **kwargs)


def unsafe_build(name: str, *args, **kwargs):
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
