def override(parent_cls):
    """Decorator for documenting method overrides.

    Args:
        parent_cls: The superclass that provides the overridden method. If
            `parent_class` does not actually have the method or the class, in which
            method is defined is not a subclass of `parent_class`, an error is raised.
    """

    class OverrideCheck:
        def __init__(self, func, expected_parent_cls):
            self.func = func
            self.expected_parent_cls = expected_parent_cls

        def __set_name__(self, owner, name):
            if not issubclass(owner, self.expected_parent_cls):
                raise TypeError(
                    f"When using the @override decorator, {owner.__name__} must be a "
                    f"subclass of {parent_cls.__name__}!"
                )
            setattr(owner, name, self.func)

    def decorator(method):
        if method.__name__ not in dir(parent_cls):
            raise NameError(
                f"When using the @override decorator, {method.__name__} must override "
                f"the respective method (with the same name) of {parent_cls.__name__}!"
            )
        OverrideCheck(method, parent_cls)
        return method

    return decorator
