from typing import Self, Any, ClassVar, Union
from typing import TypeAlias
from collections import deque
import tree # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.base import Metric

Path: TypeAlias = tuple[str, ...]
Node = Metric | dict[str, "Node"] # validate recursive imports


class MetricLogger:
    _TOKEN: ClassVar[object] = object()
    _schema: type[MetricSchema]
    _refs: dict[Path, Metric]
    _root: str
    _tree: dict[str, Node]

    def __new__(cls, *, _token: object | None = None) -> "MetricLogger":
        if _token is not cls._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )
        return super().__new__(cls)

    def __init__(self, *, _token: object | None = None) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )

    # TODO immutability
    @classmethod
    def from_schema(cls, schema: type[MetricSchema]) -> "MetricLogger":
        tree, refs = cls._build_refs_from_schema(schema)
        self = cls.__new__(cls, _token=cls._TOKEN)
        self._schema = schema
        self._root = schema.__name__
        self._tree = tree
        self._refs = refs
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
            
            protocol = extra.get("reduce", ReduceProtocol.MEAN)
            metric = MetricFactory.create(protocol)
            tree[field_name] = metric
            refs[path] = metric

        return tree, refs


    def push_data(self, data: MetricSchema, prefix: Path = ()) -> None:
        """Push all leaf values of a MetricSchema into their corresponding metrics."""

        if type(data) is not self._schema:
            raise TypeError(
                f"Expected {self._schema.__name__}, "
                f"got {type(data).__name__}."
            )

        for field_name in type(data).model_fields:
            path = prefix + (field_name,)
            value = getattr(data, field_name)

            if isinstance(value, MetricSchema):
                self.push_data(value, prefix=path)
                continue

            self.push_value(key=field_name, value=value)


    def push_value(self, key: Path, value: Any) -> None:
        # TODO narrow down Any to stricter type annotation
        """Logs a new value or item under a (strictly existing) path to the logger """
        try:
            metric = self._refs[key]
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
            metric = self._refs[key]
        except KeyError:
            raise KeyError(f"Unknown logger path: {key}") from None
        return metric.peek()

    def reduce(self) -> MetricSchema:
        """
        Reduces all logged values based on their settings and returns a MetricSchema object.
        """
        def _reduce(path: Path, metric: Metric):
            try:
                return metric.reduce(compile=True)
            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error reducing metrics {metric} at path {path}."
                ) from e
        reduced = tree.map_structure_with_path(
            _reduce,
            self._tree,
        )
        return self._schema.model_validate(reduced)
            

    def compile(self) -> dict:
        """
        Compiles all current values and throughputs into a single dictionary.
        """
        return self.reduce().model_dump()

    def reset(self) -> None:
        """
        Resets all data stored in this MetricLogger.
        """ 
        for path, metric in self._refs.items():
            try:
                metric.flush()
            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error flushing metrics {metric} at path {path}."
                ) from e