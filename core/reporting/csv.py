"""CSV reporter: one long-form file per query, rewritten on every report.

Each row is ``(query, x, series, value)``; ``series`` is the ``/``-joined
metric path of the y series so that multi-series queries stay distinguishable
when reloaded with pandas. Mean/std reductions are written as two series,
``mean`` and ``std``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.utils import sanitize_key


class CSVConfig(ReporterConfig):
    """Configuration of :class:`CSVReporter` (``output_dir/project/<label>/``)."""

    def __init__(self, *, project: str, output_dir: str = "results") -> None:
        super().__init__(project=project)
        self.output_dir = Path(output_dir)

    def build(self, *, label: Optional[str] = None) -> CSVReporter:
        name = f"{self.world}-{label}" if label is not None else str(self.world)
        return CSVReporter(output_dir=self.output_dir / self.project_name / name)


class CSVReporter(Reporter):
    """Write every resolved query to ``<output_dir>/<query title>.csv``."""

    HEADER = ("query", "x", "series", "value")

    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def path_for(self, query: Query) -> Path:
        return self._output_dir / f"{sanitize_key(query.title)}.csv"

    def _rows(
        self, query: Query, x: list[PrimitiveType], ys: list[list[PrimitiveType]]
    ):
        if query.reduce == "none":
            for path, values in zip(query.y_paths, ys, strict=True):
                series = "/".join(path)
                for x_value, y_value in zip(x, values, strict=True):
                    yield (query.title, x_value, series, y_value)
            return

        values = np.asarray(ys, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Mean reduction expects a 2D collection of y series.")
        mean = values.mean(axis=0)
        for x_value, m in zip(x, mean.tolist(), strict=True):
            yield (query.title, x_value, "mean", m)
        if query.error == "std":
            std = values.std(axis=0)
            for x_value, s in zip(x, std.tolist(), strict=True):
                yield (query.title, x_value, "std", s)

    def _report(
        self, query: Query, x: list[PrimitiveType], ys: list[list[PrimitiveType]]
    ) -> None:
        if not ys:
            return
        with self.path_for(query).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(self.HEADER)
            writer.writerows(self._rows(query, x, ys))

    def close(self) -> None:
        pass
