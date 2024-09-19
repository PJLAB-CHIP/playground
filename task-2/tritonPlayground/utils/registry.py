class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None):

        def decorator(obj):
            register_name = name if name else obj.__name__
            if register_name in self._registry:
                raise ValueError(f"{register_name} is already registered.")
            self._registry[register_name] = obj
            return obj

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} is not registered.")
        return self._registry[name]

    def create(self, name, *args, **kwargs):
        obj = self.get(name)
        if not callable(obj):
            raise ValueError(f"{name} is not a callable.")
        return obj(*args, **kwargs)
