from abc import ABC
from dataclasses import dataclass
from typing import TypeAlias

Path: TypeAlias = tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Query(ABC):
    """Defines a reporting query over a MetricSchema."""

    title: str
    x: Path
    y: Path