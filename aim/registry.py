global_registry = {}


class Registry(object):
    _registry = global_registry

    def __init__(self, global_register: bool = False):
        if not global_register:
            self._registry = {}

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

    def __call__(self, cls, name: str = None, override: bool = True):
        return self.force_register(cls, name, override)

    def get_registry(self):
        return self._registry


def get_class_path(cls):
    module = cls.__module__
    if module == "__main__":
        return cls.__name__
    else:
        f"{module}.{cls.__name__}"
    return
