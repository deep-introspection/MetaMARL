from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
import wandb

from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.utils import sanitize_key


class WandbConfig(ReporterConfig):
    def __init__(
        self,
        *,
        project: str,
        x_disable_stats: Optional[bool] = True,
        x_disable_meta: Optional[bool] = True,
        quiet: Optional[bool] = True,
        max_end_of_run_summary_metrics: Optional[int] = 0,
        max_end_of_run_history_metrics: Optional[int] = 0,
        **kwargs,
    ):
        super().__init__(project=project)
        self.settings = {
            "x_disable_stats": x_disable_stats,
            "x_disable_meta": x_disable_meta,
            "quiet": quiet,
            "max_end_of_run_summary_metrics": max_end_of_run_summary_metrics,
            "max_end_of_run_history_metrics": max_end_of_run_history_metrics,
        }

    def build(self, *, label: Optional[str] = None) -> WandbReporter:

        name = f"{self.world}-{label}" if label is not None else self.world

        return WandbReporter(
            project=self.project_name,
            run_id=uuid.uuid4().hex,
            group=self.world,
            name=name,
            config={
                "outer_iters": self.outer_iters,
                "world_name": self.world,
            },
            settings=self.settings,
        )


class WandbReporter(Reporter):
    """Reporter rendering each query as a Plotly figure logged to one W&B run.

    The run is created lazily on the first ``report`` call so that building
    a reporter never touches the network.
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

    @staticmethod
    def _path_name(path: tuple[str, ...]) -> str:
        return "/".join(path)

    @staticmethod
    def _series_label(path: tuple[str, ...]) -> str:
        return "/".join(path)

    def _raw_series_figure(
        self,
        query: Query,
        x: list[PrimitiveType],
        ys: list[list[PrimitiveType]],
    ) -> go.Figure:

        fig = go.Figure()

        for path, values in zip(
            query.y_paths,
            ys,
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=values,
                    mode="lines+markers",
                    name=self._series_label(path),
                )
            )

        return fig

    def _mean_figure(
        self,
        query: Query,
        x: list[PrimitiveType],
        ys: list[list[PrimitiveType]],
    ) -> go.Figure:
        fig = go.Figure()
        values = np.asarray(ys, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError("Mean reduction expects a 2D collection of y series.")

        mean = np.mean(values, axis=0)

        if query.error == "std":
            std = np.std(values, axis=0)
            upper = mean + std
            lower = mean - std
            fig.add_trace(
                go.Scatter(
                    x=list(x) + list(x)[::-1],
                    y=upper.tolist() + lower[::-1].tolist(),
                    mode="lines",
                    fill="toself",
                    line=dict(
                        width=0,
                    ),
                    name="±1 std",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean.tolist(),
                mode="lines+markers",
                line=dict(
                    width=3,
                ),
                marker=dict(size=4),
                name="mean",
            )
        )

        return fig

    def _report(
        self,
        query: Query,
        x: list[PrimitiveType],
        ys: list[list[PrimitiveType]],
    ) -> None:

        self._init_run()
        if self._run is None:
            raise RuntimeError("W&B run failed to initialize.")
        if not ys:
            return

        if query.reduce == "none":
            fig = self._raw_series_figure(query=query, x=x, ys=ys)
        elif query.reduce == "mean":
            fig = self._mean_figure(query=query, x=x, ys=ys)
        else:
            raise ValueError(f"Unknown query reduction: {query.reduce!r}")

        x_name = self._path_name(query.x)

        fig.update_layout(
            title=query.title,
            xaxis_title=x_name,
            yaxis_title="value",
            hovermode="x unified",
            template="plotly_white",
            height=650,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_xaxes(rangeslider_visible=False)
        plot_name = sanitize_key(query.title)
        self._run.log({f"plots/{plot_name}": fig})

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
