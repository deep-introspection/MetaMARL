from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from core.metrics.metric.series import SeriesMetric
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query


class TensorBoardConfig(ReporterConfig):

    def __init__(
        self,
        *,
        project: str,
        log_dir: str = "runs",
    ) -> None:
        super().__init__(project=project)
        self.log_dir = Path(log_dir)

    def build(
        self,
        *,
        label: Optional[str] = None,
    ) -> TensorBoardReporter:

        name = (
            f"{self.world}-{label}"
            if label is not None
            else self.world
        )

        return TensorBoardReporter(
            log_dir=self.log_dir / self.project_name / name,
        )


class TensorBoardReporter(Reporter):

    def __init__(
        self,
        *,
        log_dir: Path,
    ) -> None:
        self._log_dir = log_dir
        self._writer: SummaryWriter | None = None

    def _get_writer(self) -> SummaryWriter:
        if self._writer is None:
            self._writer = SummaryWriter(
                log_dir=str(self._log_dir),
            )

        return self._writer

    def _report(
        self,
        query: Query,
        x: SeriesMetric,
        y: SeriesMetric,
    ) -> None:
        writer = self._get_writer()

        x_values = x.peek(compile=False)
        y_values = y.peek(compile=False)

        x_name = "/".join(query.x)
        y_name = "/".join(query.y)

        for x_value, y_value in zip(
            x_values,
            y_values,
            strict=True,
        ):
            step = int(x_value)

            if step != x_value:
                raise TypeError(
                    f"TensorBoard x-axis must be integer-valued: "
                    f"{query.x} contains {x_value!r}"
                )

            writer.add_scalar(
                tag=f"{y_name}_vs_{x_name}",
                scalar_value=y_value,
                global_step=step,
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None