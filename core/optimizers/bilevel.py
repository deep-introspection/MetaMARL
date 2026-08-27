"""Bilevel optimizer: an outer mechanism search wrapping an inner policy optimizer.

The outer level (``ESConfig`` / ``ESOptimizer``) searches the mechanism
parameters; the inner level (``APPOptimizerConfig`` / ``RayOptimizer``) trains
the agents' policies against each candidate mechanism. Both share one
``World`` Ray actor through which candidates and step records are exchanged.

``BilevelConfig`` is the composition root: it starts Ray, creates the
reporting and World actors, injects the mechanism template and the seeds
into both levels, builds them and ties the ES population size to the number
of inner environments. ``BilevelOptimizer.run`` is the outer loop.

Example
-------
>>> cfg = (
...     BilevelConfig()
...     .world(world_name="fishery")
...     .mechanism(mechanism=ChainedMechanism(children=(quota, subsidy)))
...     .training(outer_iters=100)
...     .outer(ESConfig().training(sigma=0.15).environment(env=FisheryRegulatorEnv, ...))
...     .inner(APPOptimizerConfig().environment(env=FisheryRegulatedEnv, ...))
... )
>>> result = cfg.build_optimizer().run()

See ``examples/bilevel_fishery/debug.py`` for a complete configuration.
"""

import logging
import uuid
from typing import Any, Literal, Optional, Self

import ray
from ray.actor import ActorHandle

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.mechanism.base import Mechanism
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.reporting.enums import ReporterType
from core.reporting.wandb import WandbReporter
from core.world.base import World

logger = logging.getLogger(__name__)


class BilevelConfig(OptimizerConfig):
    """Fluent configuration of a bilevel run (see module docstring).

    Builder methods: :meth:`world`, :meth:`mechanism`, :meth:`training`,
    :meth:`ray`, :meth:`reporting`, :meth:`inner`, :meth:`outer`. Every
    method returns ``self``.
    """

    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or BilevelOptimizer)
        self.outer_cfg = None
        self.inner_cfg = None
        self.outer_iters = 10

        # TODO what is the point of having seed here
        self.seed = None
        self.world_name: Optional[str] = None
        self.ray_cfg = None
        self.mechanism_template: Optional[Mechanism] = None
        self.output_dir: str | None = None

        # TODO generalize
        self.wandb_cfg: dict[str, Any] | None = None
        self._reporter: ActorHandle[WandbReporter] | None = None

    # @override(OptimizerConfig)
    # def _get_logger_schema(self):
    #     return LoggerSchema(
    #         inner: self.inner_cfg._get_logger_schema
    #         outer: self.inner_cfg._get_logger_schema
    #     )

    @property
    def reporter(self) -> ActorHandle[WandbReporter] | None:
        return self._reporter

    def inner(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.inner_cfg = cfg
        return self

    def outer(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.outer_cfg = cfg
        return self

    def world(self, *, world_name: str, **kwargs) -> Self:
        """Name the shared ``World`` actor (a random suffix keeps runs distinct)."""
        if world_name is not None:
            self.world_name = f"{world_name}_{uuid.uuid4().hex[:8]}"
        return self

    def mechanism(self, *, mechanism: Mechanism, **kwargs) -> Self:
        """Set the mechanism template shared by the inner and outer optimizers.

        The template fixes the optimizer space (``mechanism.dimension``,
        ``encode``/``decode``) and acts as the default mechanism of the
        regulated environments until a candidate is published.
        """
        if mechanism is not None:
            if not isinstance(mechanism, Mechanism):
                raise TypeError(
                    f"mechanism must be a Mechanism instance, got {type(mechanism).__name__}"
                )
            self.mechanism_template = mechanism
        return self

    def training(
        self, *, outer_iters: int, output_dir: str | None = None, **kwargs
    ) -> Self:
        """Set the number of outer (ES) generations and an optional output directory."""
        if outer_iters is not None:
            self.outer_iters = outer_iters
        self.output_dir = output_dir
        return self

    def ray(
        self,
        *,
        device: DeviceType = "cpu",
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        omp_threads: int = 1,
        logging_level: str = "ERROR",
        runtime_env: Optional[dict] = None,
        **kwargs,
    ) -> Self:
        """Configure the local Ray runtime (device, CPU/GPU counts, runtime env)."""
        self.ray_cfg = RayRuntimeConfig(
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            omp_threads=omp_threads,
            logging_level=logging_level,
            runtime_env=runtime_env,
            init_kwargs=kwargs,
        )
        return self

    # TODO remote actor access to credentials
    def reporting(
        self,
        reporter: Literal["wandb", "local"],
        project_name: str,
        config: Optional[dict[str, Any]] = None,
        settings_dict: Optional[dict[str, Any]] = None,
    ) -> Self:
        """Select the reporting backend. Only Weights & Biases is available."""
        if ReporterType(reporter) == ReporterType.wandb:
            self.wandb_cfg = {
                "project_name": project_name,
                "config": config,
                "settings": settings_dict or {},
            }
        elif ReporterType(reporter) == ReporterType.local:
            raise TypeError("Local reporting is not available yet.")

        return self

    @override(OptimizerConfig)
    def build_optimizer(self):
        """Start Ray, create the actors, build both levels and return a ``BilevelOptimizer``.

        The ES population size is set to the inner optimizer's ``batch_capacity``
        (number of regulated environments divided by the number of seeds), so
        every candidate is evaluated by exactly one environment per seed.
        """
        if self.mechanism_template is None:
            raise ValueError(
                "BilevelConfig requires .mechanism(mechanism=...) to be set"
            )
        if self.inner_cfg is None or self.outer_cfg is None:
            raise ValueError("BilevelConfig requires both .inner(...) and .outer(...)")

        RayRuntime.ensure_initialized(self.ray_cfg or RayRuntimeConfig())

        if self.wandb_cfg:
            project = self.wandb_cfg["project_name"]
            extra_cfg = self.wandb_cfg.get("config") or {}
            self._reporter = WandbReporter.options(
                name=f"{self.world_name}_wandb"
            ).remote(
                project=project,
                name=f"{project}-{self.world_name}",
                config={
                    "outer_iters": self.outer_iters,
                    "world_name": self.world_name,
                    **extra_cfg,
                },
                settings=self.wandb_cfg["settings"],
            )

        world = World.options(name=self.world_name).remote(reporting=self.reporter)

        inner_cfg = self.inner_cfg.copy()
        outer_cfg = self.outer_cfg.copy()

        if self.mechanism_template is None:
            raise ValueError(
                "BilevelConfig requires .mechanism(mechanism=...) to be set"
            )

        outer_cfg.dimension = self.mechanism_template.dimension
        inner_cfg = inner_cfg._merge_env_config({"mechanism": self.mechanism_template})

        # Assign see to outer cfg for looping
        if inner_cfg.seeds is not None:
            outer_cfg._merge_env_config(
                {
                    "seeds": inner_cfg.seeds,
                }
            )
            # inner_cfg.seed = None

        if inner_cfg.eval_seeds is not None:
            outer_cfg._merge_env_config(
                {
                    "eval_seeds": inner_cfg.eval_seeds,
                }
            )

        outer_cfg = outer_cfg._merge_env_config({"mechanism": self.mechanism_template})

        inner_opt = inner_cfg.build_optimizer(
            world=world, world_name=self.world_name, reporting=self.reporter
        )
        outer_opt = outer_cfg.build_optimizer(
            world=world, inner_opt=inner_opt, reporting=self.reporter
        )

        # what if outer_opt does not have that property ??
        # override outer_opt population size with inner_opt batch_size
        # set inner batch capacity to be the same as inner for batch sampling
        outer_opt.batch_capacity = inner_opt.batch_capacity
        return self.opt_class(config=self, outer=outer_opt, inner=inner_opt)


class BilevelOptimizer(Optimizer):
    """Outer loop: run the outer optimizer ``outer_iters`` times, stop early on convergence.

    Parameters
    ----------
    config : BilevelConfig
    outer : Optimizer
        Optimizer whose ``run()`` returns a dict with ``best_fitness`` and an
        optional ``converged`` flag (the ES).
    inner : Optimizer
        Inner optimizer, driven by the outer env; kept for lifecycle access.
    """

    def __init__(self, config: BilevelConfig, outer: Optimizer, inner: Optimizer):
        super().__init__(config)

        self.world_name = config.world_name
        self.max_outer_iters = config.outer_iters
        self.outer = outer
        self.inner = inner
        self.output_dir = config.output_dir
        self.mechanism_template = config.mechanism_template

        self.outer_iter = 0
        self.converged = False
        self.best_trajectory: list[dict] | None = None
        self.all_trajectories: list[tuple[int, float, list[dict]]] = []
        self.population_history: list[tuple[int, list]] = []
        self.es_metrics_history: list[dict] = []

        # TODO avoid hardcoding prefix
        # self.wandb_run.define_metric("ppo/ppo_step")
        # self.wandb_run.define_metric("ppo/*", step_metric="ppo/ppo_step")

    def run(self) -> dict[str, Any]:
        """Run the outer generations and return a summary dict.

        Returns
        -------
        dict
            ``converged``, ``outer_iters`` (generations actually run),
            ``best_fitness``, ``best_mechanism`` (encoded vector) and history
            fields (``best_trajectory``, ``all_trajectories``,
            ``population_history``).
        """
        logger.info(
            "[Bilevel] Starting run | max_outer_iters=%d | world=%s",
            self.max_outer_iters,
            self.world_name,
        )

        for i in range(self.max_outer_iters):
            self.outer_iter = i
            logger.info(
                "[Bilevel] Outer iteration %d / %d started",
                i + 1,
                self.max_outer_iters,
            )

            outer_metrics = self.outer.run()

            self.metrics.log_dict(
                {
                    "bilevel/outer_iter": i,
                    "bilevel/best_fitness": outer_metrics.get(
                        "best_fitness", -float("inf")
                    ),
                }
            )

            if outer_metrics.get("converged", False):
                self.converged = True
                logger.info(
                    "[Bilevel] EARLY STOP | "
                    "outer optimizer converged | "
                    "iter=%d | best_fitness=%.4f",
                    i,
                    outer_metrics["best_fitness"],
                )
                break

        logger.info(
            "[Bilevel] Run finished | iters=%d | converged=%s | best_fitness=%.4f",
            self.outer_iter + 1,
            self.converged,
            self.outer.best_fitness,
        )

        # TODO fig reporter with the new wandb reporter actor
        if self.config.reporter is not None:
            ray.get(self.config.reporter.finish.remote())

        return {
            "converged": self.converged,
            "outer_iters": self.outer_iter + 1,
            "best_fitness": self.outer.best_fitness,
            "best_mechanism": self.outer.best_candidate,
            "best_trajectory": self.best_trajectory,
            "all_trajectories": self.all_trajectories,
            "population_history": self.population_history,
        }
