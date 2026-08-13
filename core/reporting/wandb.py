from __future__ import annotations

from typing import Any, Optional
import uuid
import wandb

from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query

class WandbConfig(ReporterConfig):

    def __init__(
            self, 
            *,
            project: str,
            x_disable_stats: Optional[bool] = True,
            x_disable_meta: Optional[bool] = True,
            quiet: Optional[bool]= True,
            max_end_of_run_summary_metrics: Optional[int] = 0,
            max_end_of_run_history_metrics: Optional[int] = 0,
            **kwargs
        ):
        super().__init__(project=project)
        self.settings = {
            "x_disable_stats": x_disable_stats,
            "x_disable_meta": x_disable_meta,
            "quiet": quiet,
            "max_end_of_run_summary_metrics": max_end_of_run_summary_metrics,
            "max_end_of_run_history_metrics": max_end_of_run_history_metrics,
        }

    def build(
            self, 
            *,
            label: Optional[str] = None
        ) -> WandbReporter:

        name = (
            f"{self.world}-{label}"
            if label is not None
            else self.world
        )

        return WandbReporter(
                project = self.project_name,
                run_id = uuid.uuid4().hex,
                group = self.world,
                name = name,
                config = {
                    "outer_iters": self.outer_iters,
                    "world_name": self.world,
                },
                settings=self.settings,
            )

class WandbReporter(Reporter):
    """
    Ray actor that owns a single W&B run.

    Other actors/processes should never receive the raw wandb.Run object.
    They only send serializable payloads to this actor.
    """

    def __init__(
        self,
        *,
        project: str,
        name: str,
        run_id: str,
        group: str,
        config: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        self._defined_prefixes: set[str] = set()
        self._run: wandb = None
        self._project = project
        self._name = name
        self._run_id = run_id
        self._group = group
        self._config = config
        self._settings = settings

    def _init_run(self):
        if self._run is None:
            self._run = wandb.init(
                project=self._project,
                id=self._run_id,
                group=self._group,
                name=self._name,
                config=self._config or {},
                reinit="create_new",
                settings=wandb.Settings(**(self._settings or {})),
            )

    def _report(
        self,
        query: Query,
        x: list[PrimitiveType],
        y: list[PrimitiveType],
    ) -> None:
        self._init_run()

        x_name = "/".join(query.x)
        y_name = "/".join(query.y)

        self._run.log({
            f"{y_name}_vs_{x_name}": wandb.plot.line_series(
                xs=x,
                ys=[y],
                keys=[y_name],
                title=f"{y_name} vs {x_name}",
                xname=x_name,
            )
        })
        
    def close(self) -> None:
        self._init_run()
        if run is not None:
            run.finish()
            run = None