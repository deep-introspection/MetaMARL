from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from core.metrics.metric.series import SeriesMetric
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query


class CSVConfig(ReporterConfig):
    def __init__(
        self,
        *,
        project: str,
        output_dir: str = "results",
    ) -> None:
        super().__init__(project=project)
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        label: Optional[str] = None,
    ) -> CSVReporter:
        name = f"{self.world}-{label}" if label is not None else self.world

        return CSVReporter(
            output_dir=self.output_dir / self.project_name / name,
        )


class CSVReporter(Reporter):
    def __init__(
        self,
        *,
        output_dir: Path,
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _report(
        self,
        query: Query,
        x: SeriesMetric,
        y: SeriesMetric,
    ) -> None:
        x_values = x.peek(compile=False)
        y_values = y.peek(compile=False)

        x_name = "/".join(query.x)
        y_name = "/".join(query.y)

        filename = f"{'__'.join(query.y)}_vs_{'__'.join(query.x)}.csv"

        path = self._output_dir / filename

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    x_name,
                    y_name,
                ]
            )

            writer.writerows(
                zip(
                    x_values,
                    y_values,
                    strict=True,
                )
            )

    def close(self) -> None:
        pass
