from abc import ABC
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

Path: TypeAlias = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Query(ABC):
    """Defines a reporting query over a MetricSchema."""

    title: str
    x: Path
    y: Path | tuple[Path, ...]
    reduce: Literal["none", "mean"] = "none"
    error: Literal[
        "none",
        "std",
    ] = "none"

    def __post_init__(self) -> None:
        if self.error != "none" and self.reduce == "none":
            raise ValueError(
                "Query error requires a reduction. "
                "For example: reduce='mean', error='std'."
            )

    @property
    def y_paths(
        self,
    ) -> tuple[Path, ...]:
        if self.y and isinstance(self.y[0], tuple):
            return cast(tuple[Path, ...], self.y)
        return (cast(Path, self.y),)
