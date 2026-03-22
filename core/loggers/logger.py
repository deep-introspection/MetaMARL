from typing import Self, Any, ClassVar
from typing import TypeAlias

from core.loggers.schemas import LoggerSchema

Path: TypeAlias = tuple[str, ...]


class ResultsLogger:
    __token__: ClassVar[object] = object()
    __root__: Any
    __tree__: Self = None
    __paths__: tuple[Path, ...]
    __refs__: dict[Path, Any]  # TODO replace any in refs

    def __new__(cls, *, _token: object | None = None) -> "ResultsLogger":
        if _token is not cls.__token__:
            raise TypeError(
                "ResultsLogger cannot be instantiated directly. "
                "Use ResultsLogger.from_schema(schema)"
            )
        return super().__new__(cls)

    def __init__(self, *, _token: object | None = None) -> None:
        if _token is not self.__token__:
            raise TypeError(
                "ResultsLogger cannot be instantiated directly. "
                "Use ResultsLogger.from_schema(schema)"
            )

    # TODO immutability
    @classmethod
    def from_schema(cls, schema: LoggerSchema) -> "ResultsLogger":
        tree, paths, refs = cls._build_tree_from_schema(schema)
        self = cls.__new__(cls, _token=cls.__token__)
        self.__tree__ = tree
        self.__paths__ = paths
        self.__refs__ = refs
        return self

    @classmethod
    def _build_tree_from_schema(cls, schema: LoggerSchema) -> dict: ...
