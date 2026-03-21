import logging
import uuid
from typing import Optional, Self

import wandb
import matplotlib.pyplot as plt

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.visualizers.bilevel_viz_reporter import BilevelVizReporter
from core.world.base import World

logger = logging.getLogger(__name__)


class BilevelConfig(OptimizerConfig):
    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or BilevelOptimizer)
        self.outer_cfg = None
        self.inner_cfg = None
        self.outer_iters = 10

        # TODO what is the point of having seed here
        self.seed = None
        self.world_name: Optional[str] = None
        self.ray_cfg = None
        self.mechanism_space: Optional[MechanismSpace] = None
        self.default_mechanism: Optional[Mechanism] = None
        self.output_dir: str | None = None

    # @override(OptimizerConfig)
    # def _get_logger_schema(self):
    #     return LoggerSchema(
    #         inner: self.inner_cfg._get_logger_schema
    #         outer: self.inner_cfg._get_logger_schema
    #     )
    def inner(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.inner_cfg = cfg
        return self

    def outer(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.outer_cfg = cfg
        return self

    def world(self, *, world_name: str, **kwargs) -> Self:
        if world_name is not None:
            self.world_name = f"{world_name}_{uuid.uuid4().hex[:8]}"
        return self

    def mechanism(
        self, *, space: MechanismSpace, default: Mechanism = None, **kwargs
    ) -> Self:
        if space is not None:
            self.mechanism_space = space
            self.default_mechanism = default or space.default()
        return self

    def training(
        self, *, outer_iters: int, seed=None, output_dir: str | None = None, **kwargs
    ) -> Self:
        if outer_iters is not None:
            self.outer_iters = outer_iters
        if seed is not None:
            self.seed = seed
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
    
    def reporting(
            self,
    ) -> Self:
        # TODO
        pass

    @override(OptimizerConfig)
    def build_optimizer(self):
        RayRuntime.ensure_initialized(self.ray_cfg or RayRuntimeConfig())
        world = World.options(name=self.world_name).remote()

        inner_cfg = self.inner_cfg.copy()
        outer_cfg = self.outer_cfg.copy()

        if self.mechanism_space is not None:
            outer_cfg.dimension = self.mechanism_space.dimension

            inner_cfg = inner_cfg._merge_env_config(
                {
                    "mechanism_space": self.mechanism_space,
                    "default_mechanism": self.default_mechanism,
                }
            )

        outer_cfg = outer_cfg._merge_env_config(
            {
                "mechanism_space": self.mechanism_space,
                "default_mechanism": self.default_mechanism,
            }
        )

        inner_opt = inner_cfg.build_optimizer(world=world, world_name=self.world_name)
        outer_opt = outer_cfg.build_optimizer(world=world, inner_opt=inner_opt)

        # what if outer_opt does not have that property ??
        # override outer_opt population size with inner_opt batch_size
        # set inner batch capacity to be the same as inner for batch sampling
        outer_opt.batch_capacity = inner_opt.batch_capacity
        return self.opt_class(config=self, outer=outer_opt, inner=inner_opt)


class BilevelOptimizer(Optimizer):
    def __init__(self, config: BilevelConfig, outer: Optimizer, inner: Optimizer):
        super().__init__(config)

        self.world_name = config.world_name
        self.max_outer_iters = config.outer_iters
        self.outer = outer
        self.inner = inner
        self.output_dir = config.output_dir
        self.mechanism_space = config.mechanism_space

        self.outer_iter = 0
        self.converged = False
        self.best_trajectory: list[dict] | None = None
        self.all_trajectories: list[tuple[int, float, list[dict]]] = []
        self.population_history: list[tuple[int, list]] = []
        self.es_metrics_history: list[dict] = []

        # W&B
        # TODO have proper initialization for this
        # wandb tracking
        self.wandb_run = wandb.init(
            project="bilevel",
            name=f"bilevel-{self.world_name}",
            config={
                "outer_iters": config.outer_iters,
                "world_name": self.world_name,
            },
            reinit=True,
            settings=wandb.Settings(
                x_disable_stats=True,
                x_disable_meta=True,
                quiet=True,
                max_end_of_run_history_metrics=0,
                max_end_of_run_summary_metrics=0,
            )
        )

        # TODO temporary
        self.viz = BilevelVizReporter(
            wandb_run=self.wandb_run,
            output_dir=self.output_dir,  # optional
            mechanism_space=self.mechanism_space,  # optional
            optimize_params=getattr(self.mechanism_space, "optimize_params", None),
            num_mechanisms=16,  # TODO make this dynamic
        )

        # Attach run handle to inner/outer (simple, no interface changes)
        setattr(self.outer, "wandb_run", self.wandb_run)
        setattr(self.inner, "wandb_run", self.wandb_run)
        setattr(self.outer, "bilevel", self)
        setattr(self.inner, "bilevel", self)

        wandb.define_metric("ppo/ppo_step")
        wandb.define_metric("ppo/*", step_metric="ppo/ppo_step")

    def _wandb_log_fig(self, key: str, fig, step: int):
        if not self.wandb_run:
            return
        self.wandb_run.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)  # IMPORTANT: prevent memory leak

    def run(self) -> None:
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

            # one single call that logs everything
            self.viz.on_outer_iteration_end(
                outer_iter=i,
                outer=self.outer,
                outer_metrics=outer_metrics,
            )

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

        self.viz.finish()
        wandb.finish()
        return {
            "converged": self.converged,
            "outer_iters": self.outer_iter + 1,
            "best_fitness": self.outer.best_fitness,
            "best_mechanism": self.outer.best_candidate,
            "best_trajectory": self.best_trajectory,
            "all_trajectories": self.all_trajectories,
            "population_history": self.population_history,
        }
