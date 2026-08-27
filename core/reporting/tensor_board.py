"""TensorBoard reporter: one scalar tag per y series, indexed by the integer x.

Requires the optional ``tensorboard`` package (``uv sync --extra tensorboard``).
Multi-series queries become one tag per series (``<title>/<series path>``);
mean/std reductions become ``<title>/mean`` and ``<title>/std``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.utils import sanitize_key

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TensorBoardConfig(ReporterConfig):
    """Configuration of :class:`TensorBoardReporter` (``log_dir/project/<label>/``)."""

    def __init__(self, *, project: str, log_dir: str = "runs") -> None:
        super().__init__(project=project)
        self.log_dir = Path(log_dir)

    def build(self, *, label: Optional[str] = None) -> TensorBoardReporter:
        name = f"{self.world}-{label}" if label is not None else str(self.world)
        return TensorBoardReporter(log_dir=self.log_dir / self.project_name / name)


class TensorBoardReporter(Reporter):
    """Log resolved queries as TensorBoard scalars."""

    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer: SummaryWriter | None = None

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def _get_writer(self) -> SummaryWriter:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:  # pragma: no cover - depends on the extra
                raise ImportError(
                    "TensorBoardReporter needs the 'tensorboard' package: "
                    "uv sync --extra tensorboard"
                ) from e
            self._writer = SummaryWriter(log_dir=str(self._log_dir))
        return self._writer

    @staticmethod
    def _step(x_value: PrimitiveType, query: Query) -> int:
        step = int(x_value)
        if step != x_value:
            raise TypeError(
                f"TensorBoard x-axis must be integer-valued: {query.x} contains {x_value!r}"
            )
        return step

    def _series(
        self, query: Query, ys: list[list[PrimitiveType]]
    ) -> dict[str, list[float]]:
        if query.reduce == "none":
            return {
                "/".join(path): list(values)
                for path, values in zip(query.y_paths, ys, strict=True)
            }
        values = np.asarray(ys, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Mean reduction expects a 2D collection of y series.")
        series = {"mean": values.mean(axis=0).tolist()}
        if query.error == "std":
            series["std"] = values.std(axis=0).tolist()
        return series

    def _report(
        self, query: Query, x: list[PrimitiveType], ys: list[list[PrimitiveType]]
    ) -> None:
        if not ys:
            return
        writer = self._get_writer()
        title = sanitize_key(query.title)
        for name, values in self._series(query, ys).items():
            tag = f"{title}/{name}"
            for x_value, y_value in zip(x, values, strict=True):
                writer.add_scalar(
                    tag=tag,
                    scalar_value=y_value,
                    global_step=self._step(x_value, query),
                )
        writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
