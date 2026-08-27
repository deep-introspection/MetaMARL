"""CSV reporter: one long-form file per query, rewritten on every report.

Each row is ``(query, x, series, value, error)``; ``series`` is the label of
the resolved series (metric path with wildcards bound, or the group id of a
reduced series) and ``error`` the standard deviation when the query asked for
one (empty otherwise).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query, Series
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

    HEADER = ("query", "x", "series", "value", "error")

    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def path_for(self, query: Query) -> Path:
        return self._output_dir / f"{sanitize_key(query.title)}.csv"

    @staticmethod
    def _rows(query: Query, series: list[Series]):
        for s in series:
            errors = s.error if s.error is not None else [""] * len(s.y)
            for x_value, y_value, err in zip(s.x, s.y, errors, strict=True):
                yield (query.title, x_value, s.label, y_value, err)

    def _report(self, query: Query, series: list[Series]) -> None:
        if not series:
            return
        with self.path_for(query).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(self.HEADER)
            writer.writerows(self._rows(query, series))

    def close(self) -> None:
        pass
