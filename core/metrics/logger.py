from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, get_args, get_origin
from typing import TypeAlias
import tree # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.base import Metric

Path: TypeAlias = tuple[str, ...]

class Node(dict[str, "Node | Metric"]):
    schema: type[MetricSchema]
    children: dict[str, Node | Metric]
    dynamic: bool = False

    def __init__(
        self,
        *args,
        schema: type[MetricSchema] | None = None,
        children: dict[str, Node | Metric] | None = None,
        dynamic: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.schema = schema
        self.children = children or {}
        self.dynamic = dynamic

    def construct(self, data: dict[str, Any]):
        if self.dynamic:
            return {
                dynamic_id: child.construct(data[dynamic_id])
                for dynamic_id, child in self.items()
            }

        values: dict[str, Any] = {}

        for field_name, child in self.items():
            value = data[field_name]

            if isinstance(child, Metric):
                values[field_name] = value
            else:
                values[field_name] = child.construct(value)

        if self.schema is None:
            raise RuntimeError("Runtime Node has no schema.")

        return self.schema.model_construct(**values)

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
                if not node.dynamic: 
                    node[field_name] = child
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
                if not node.dynamic: node[field_name] = child
                continue
            
            protocol = extra.get("reduce", ReduceProtocol.MEAN)
            metric = MetricFactory.create(protocol)
            node.children[field_name] = metric
            if not node.dynamic:
                node[field_name] = metric
                refs[path] = metric

        return node, refs

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
            runtime_child = node.get(dynamic_id)

            if runtime_child is None:
                runtime_child, refs = self._build_from_schema(
                    node.schema,
                    prefix=prefix,
                )
                node[dynamic_id] = runtime_child
                self._refs.update(refs)

            return self._resolve_path(
                path=path,
                node=runtime_child,
                index=index + 1,
                prefix=prefix,
            )

        if index >= len(path):
            raise KeyError(f"Logger path does not point to a metric: {path}")
        field_name = path[index]

        try:
            child = node[field_name]
        except KeyError:
            raise KeyError(f"Unknown logger path: {path}") from None

        child_path = prefix + (field_name,)

        # leaf
        if isinstance(child, Metric):
            if index != len(path) - 1:
                raise KeyError(f"Path continues beyond metric leaf: {path}")

            return child

        return self._resolve_path(
            node=child,
            path=path,
            index=index + 1,
            prefix=child_path,
        )

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
            metric = self._resolve_path(path=key, node=self._tree, index=0, prefix=())
        metric.push(value)

    def peek_value(self, key: Path) -> Any:
        # TODO narrow down Any to stricter type annotation
        # NOTE this does not work for sub trees as of now !
        """
        Reads a metric value given its path without destructively reducing it
        """
        metric = self._refs.get(key)
        if metric is None:
            raise KeyError(f"Unknown logger path: {key}")
        return metric.peek()

    def peek(self) -> MetricSchema:
        """
        Returns all accumulated values as a MetricSchema
        without destructively reducing them.
        """
        def _peek(path: Path, metric: Metric):
            try:
                return metric.peek(compile=False)
            except Exception as e:
                raise ValueError(
                    f"Error peeking metric {metric} at path {path}."
                ) from e
        peeked = tree.map_structure_with_path(
            _peek,
            self._tree,
        )
        return self._tree.construct(peeked)

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

    def flush(self, key: Path) -> None:
        """
        Flush all accumulated values for the metric at `key`.
        """
        metric = self._refs.get(key)

        if metric is None: raise KeyError(f"Unknown logger path: {key}")

        try:
            metric.flush()
        except Exception as e:
            raise ValueError(f"Error flushing metric {metric} at path {key}.") from e