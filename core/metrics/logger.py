from __future__ import annotations
from abc import ABC
import builtins
from dataclasses import dataclass
from typing import Generic, MutableMapping, Self, Any, ClassVar, TypeVar, Union, get_args, get_origin
from typing import TypeAlias
from collections import deque
import tree # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.base import Metric

Path: TypeAlias = tuple[str, ...]

@dataclass(slots=True)
class Node:
    schema: type[MetricSchema]
    children: dict[str, Node | Metric]
    dynamic: bool = False


class MetricLogger(ABC):
    _TOKEN: ClassVar[object] = object()
    _schema: type[MetricSchema]
    _refs: dict[Path, Metric]
    _root: str
    _tree: Node

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
        tree, refs = cls._build_from_schema(schema)
        self = cls.__new__(cls, _token=cls._TOKEN)
        self._schema = schema
        self._root = schema.__name__
        self._tree = tree
        self._refs = refs
        return self

    @classmethod
    def _build_from_schema(
        cls, 
        schema: type[MetricSchema],
        *,
        prefix: Path = (),
        dynamic: bool = False,
    ) -> tuple[Node, dict[Path, Metric]]:
        # TODO guardrails when Metric isnt well formatted
        refs: dict[Path, Metric] = {}
        node = Node(schema=schema, children={}, dynamic=dynamic)

        for field_name, field in schema.__pydantic_fields__.items():
            path = prefix + (field_name,)
            ann = field.annotation
            extra = field.json_schema_extra or {}
            
            if isinstance(ann, type) and issubclass(ann, MetricSchema):
                child, child_ref = cls._build_from_schema(ann, prefix=path)
                node.children[field_name] = child
                refs.update(child_ref)
                continue

            # CASE WHEN dict[ID, MetricSchema]
            if get_origin(ann) is dict:
                _, value_ann = get_args(ann)

                if not (isinstance(value_ann, type) and issubclass(value_ann, MetricSchema)):
                    raise TypeError(
                        f"{schema.__name__}.{field_name} must be "
                        f"dict[ID, MetricSchema], got {ann!r}"
                    )
                child, _ = cls._build_from_schema(value_ann, prefix=path, dynamic=True)
                node.children[field_name] = child
                continue
            
            protocol = extra.get("reduce", ReduceProtocol.MEAN)
            metric = MetricFactory.create(protocol)
            node.children[field_name] = metric
            refs[path] = metric

        return node, refs

    def _register_refs(
        self,
        node: Node,
        *,
        prefix: Path,
    ) -> None:

        for field_name, child in node.children.items():
            path = prefix + (field_name,)

            if isinstance(child, Metric):
                if path not in self._refs:
                    self._refs[path] = child.empty_copy()
                continue

            if child.dynamic:
                continue

            self._register_refs(
                child,
                prefix=path,
            )

    def _resolve_path(
            self, 
            path: Path,
            *,
            node: Node,
            index: int,
            prefix: Path
            ) -> Metric:

        if node.dynamic:
            if index >= len(path):
                raise KeyError(f"Expected dynamic ID at path: {path}")
            dynamic_id = path[index]
            prefix = prefix + (dynamic_id,)
            index += 1

        if index >= len(path):
            raise KeyError(f"Logger path does not point to a metric: {path}")
        field_name = path[index]

        try:
            child = node.children[field_name]
        except KeyError:
            raise KeyError(f"Unknown logger path: {path}") from None

        child_path = prefix + (field_name,)

        # leaf
        if isinstance(child, Metric):
            if index != len(path) - 1:
                raise KeyError(f"Path continues beyond metric leaf: {path}")

            # static refs exist
            metric = self._refs.get(child_path)
            if metric is not None:
                return metric

            # register refs for dynamic ID
            if node.dynamic:
                self._register_refs(node, prefix=prefix)
                try:
                    return self._refs[child_path]
                except KeyError:
                    raise RuntimeError(
                        f"Metric path was valid but not registered: {child_path}"
                    ) from None
            raise RuntimeError(f"Static metric path was not registered: {child_path}")

        metric = self._resolve_path(
            node=child,
            path=path,
            index=index + 1,
            prefix=child_path,
        )
        if node.dynamic:
            self._register_refs(
                node,
                prefix=prefix,
            )

            return self._refs[path]

        return metric

    # TODO refactor into one push function
    def push_data(self, data: MetricSchema, prefix: Path = ()) -> None:
        """Push all leaf values of a MetricSchema into their corresponding metrics."""

        if not prefix and type(data) is not self._schema:
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

            if isinstance(value, dict):
                for dynamic_id, child in value.items():
                    self.push_data(child, prefix=path + (dynamic_id,))
                continue

            self.push(key=path, value=value)


    def push(self, key: Path, value: Any) -> None:
        # TODO narrow down Any to stricter type annotation
        """Logs a new value or item under a (strictly existing) path to the logger """

        metric = self._refs.get(key)
        if metric is None:
            self._resolve_path(path=key, node=self._tree, index=0, prefix=())

            try:
                metric = self._refs.get(key)
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
            metric = self._refs.get(key)
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