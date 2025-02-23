import importlib

_registry = {}


class Registry(object):
    def __init__(self, global_register: bool = False):
        self._registry = _registry if global_register else {}

    def auto_register(self, name: str = None):
        def decorator(cls):
            _name = name if name else get_class_path(cls)
            if _name in self._registry:
                raise ValueError(f"`{_name}` is already registered!")
            self._registry[_name] = cls
            return cls

        return decorator

    def register(self, name: str = None):
        def decorator(cls):
            _name = name if name else get_class_path(cls)
            if _name in self._registry:
                raise ValueError(f"`{_name}` is already registered!")
            self._registry[_name] = cls
            return cls

        return decorator

    def force_register(self, cls, name: str = None, override: bool = True):
        _name = name if name else get_class_path(cls)
        if not override and _name in self._registry:
            raise ValueError(f"`{_name}` is already registered!")
        self._registry[_name] = cls
        return cls


def get_class_path(cls):
    module = cls.__module__
    if module == "__main__":
        return cls.__name__
    else:
        f"{module}.{cls.__name__}"
    return
