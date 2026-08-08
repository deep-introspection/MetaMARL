from typing import Self, Any, ClassVar, Union
from typing import TypeAlias
from collections import deque
import tree # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.base import Metric

Path: TypeAlias = tuple[str, ...]
Node = Metric | dict[str, Node] # validate recursive imports


class MetricLogger:
    __token__: ClassVar[object] = object()
    __schema__: MetricSchema
    __refs__: dict[Path, Metric]
    __root__: str
    __tree__: dict[str, Node]

    def __new__(cls, *, _token: object | None = None) -> "MetricLogger":
        if _token is not cls.__token__:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )
        return super().__new__(cls)

    def __init__(self, *, _token: object | None = None) -> None:
        if _token is not self.__token__:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )

    # TODO immutability
    @classmethod
    def from_schema(cls, schema: type[MetricSchema]) -> "MetricLogger":
        tree, refs = cls._build_refs_from_schema(schema)
        self = cls.__new__(cls, _token=cls.__token__)
        self.__schema__ = schema
        self.__root__ = schema.__name__
        self.__tree__ = tree
        self.__refs__ = refs
        return self

    @classmethod
    def _build_refs_from_schema(
        cls, 
        schema: type[MetricSchema],
        prefix: Path = (),
    ) -> tuple[dict[str, Node], dict[Path, Metric]]:
        # TODO guardrails when Metric isnt well formatted
        refs: dict[Path, Metric] = {}
        tree: dict[str, Node] = {}

        for field_name, field in schema.__pydantic_fields__.items():
            path = prefix + (field_name,)
            ann = field.annotation
            extra = field.json_schema_extra or {}
            
            if isinstance(ann, type) and issubclass(ann, MetricSchema):
                sub_tree, child_res = cls._build_refs_from_schema(ann, prefix=path)
                tree[field_name] = sub_tree
                refs.update(child_res)
                continue
            
            protocol = field.json_schema_extra.get("reduce", ReduceProtocol.MEAN)
            metric = MetricFactory.create(protocol)
            tree[field_name] = metric
            refs[path] = metric

        return tree, refs


    def push_data(self, data: MetricSchema) -> None:
        """Logs all leafs of a possibly nested schema into logger.
        
        Traverses through all leafs of a `MetricSchema` object provided it has the same schema as
        the instantiated logger
        """
        pass

    def push_value(self, key: Path, value: Any) -> None:
        # TODO narrow down Any to stricter type annotation
        """Logs a new value or item under a (strictly existing) path to the logger """
        try:
            metric = self.__refs__[key]
        except KeyError:
            raise KeyError(f"Unknown logger path: {key}") from None
        metric.push(value)

    def peek(self, key: Path) -> Any:
        # TODO narrow down Any to stricter type annotation
        # NOTE this does not work for sub trees as of now !
        """
        Reads a metric value given its path without destructively reducing it
        """
        try:
            metric = self.__refs__[key]
        except KeyError:
            raise KeyError(f"Unknown logger path: {key}") from None
        metric.peek()

    def reduce(self) -> MetricSchema:
        """
        Reduces all logged values based on their settings and returns a MetricSchema object.
        """
        def _reduce(path: Path, metric: Metric):
            try:
                return metric.reduce()
            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error reducing metrics {metric} at path {path}."
                ) from e
        reduced = tree.map_structure_with_path(
            _reduce,
            self.__tree__,
        )
        return self.__schema__.model_validate(reduced)
            

    def compile(self) -> dict:
        """
        Compiles all current values and throughputs into a single dictionary.
        """
        return self.reduce().model_dump()

    def reset(self) -> None:
        """
        Resets all data stored in this MetricLogger.
        """
        for path, metric in self.__refs__:
            try:
                metric.flush()
            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error flushing metrics {metric} at path {path}."
                ) from e