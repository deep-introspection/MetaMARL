def override(parent_cls):
    """Decorator for documenting and enforcing method overrides.

    Validates at class-definition time that the decorated method:
    (a) actually exists on ``parent_cls``, and (b) is defined on a proper
    subclass of ``parent_cls``.  Raises ``NameError`` or ``TypeError``
    otherwise.

    Parameters
    ----------
    parent_cls : type
        The superclass that provides the method being overridden.

    Returns
    -------
    Callable
        The original method, unchanged, after override validation passes.

    Raises
    ------
    NameError
        If the decorated method name does not exist on ``parent_cls``.
    TypeError
        If the class that owns the decorated method is not a subclass of
        ``parent_cls``.

    Examples
    --------
    >>> class Base:
    ...     def step(self): ...
    >>> class Child(Base):
    ...     @override(Base)
    ...     def step(self): return 42
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
