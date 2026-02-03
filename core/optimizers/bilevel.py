import logging
import uuid
from typing import Optional, Self

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
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

    @override(OptimizerConfig)
    def build_optimizer(self):
        RayRuntime.ensure_initialized(self.ray_cfg or RayRuntimeConfig())
        world = World.options(name=self.world_name).remote()

        inner_cfg = self.inner_cfg.copy()
        outer_cfg = self.outer_cfg.copy()

        if self.mechanism_space is not None:
            outer_cfg.dimension = self.mechanism_space().dimension

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

            trajectory = outer_metrics.get("best_trajectory")
            fitness = outer_metrics.get("best_fitness", -float("inf"))
            pop_history = outer_metrics.get("population_history", [])

            if pop_history:
                self.population_history.append((i, pop_history[-1]))

            # Collect ES metrics for plotting
            self._collect_es_metrics(i, fitness)

            if trajectory:
                self.best_trajectory = trajectory
                self.all_trajectories.append((i, fitness, trajectory))
                self._save_intermediate_plot(i, fitness, trajectory)

            self._save_parameter_plots()
            self._save_es_metrics_plot()

            self.metrics.log_dict(
                {
                    "bilevel/outer_iter": i,
                    "bilevel/best_fitness": fitness,
                }
            )

            # ---- Early stopping ----
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

        return {
            "converged": self.converged,
            "outer_iters": self.outer_iter + 1,
            "best_fitness": self.outer.best_fitness,
            "best_mechanism": self.outer.best_candidate,
            "best_trajectory": self.best_trajectory,
            "all_trajectories": self.all_trajectories,
            "population_history": self.population_history,
        }

    def _save_intermediate_plot(
        self, iteration: int, fitness: float, trajectory: list[dict]
    ) -> None:
        """Save intermediate visualization for this iteration."""
        if not self.output_dir:
            return

        try:
            from pathlib import Path

            from examples.bilevel_fishery.visualization import (
                plot_combined_trial_analysis,
            )

            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            mechanism_params = None
            if self.mechanism_space is not None and self.outer.best_candidate is not None:
                candidate = self.outer.best_candidate
                # Decode candidate using mechanism space to get actual values
                mechanism = self.mechanism_space.decode(candidate)
                # Only show optimized params
                optimize_params = getattr(self.mechanism_space, "optimize_params", None)
                if optimize_params:
                    mechanism_params = {p: getattr(mechanism, p) for p in optimize_params}
                else:
                    mechanism_params = {
                        "min_stock": mechanism.min_stock,
                        "fine_amount": mechanism.fine_amount,
                    }

            save_path = output_path / f"iter_{iteration:03d}.png"
            plot_combined_trial_analysis(
                trajectory,
                mechanism_params=mechanism_params,
                title=f"Iteration {iteration} (fitness={fitness:.4f})",
                save_path=str(save_path),
            )
            logger.info("[Bilevel] Saved intermediate plot to %s", save_path)

        except Exception as e:
            logger.warning("[Bilevel] Failed to save intermediate plot: %s", e)

    def _collect_es_metrics(self, iteration: int, fitness: float) -> None:
        """Collect ES metrics from outer optimizer's env for plotting."""
        metrics = {
            "generation": iteration,
            "best_fitness": fitness,
            "total_fines": 0.0,
            "mean_fish": 0.0,
            "min_fish": 1.0,
            "mean_collapse_rate": 0.0,
        }

        # Try to get metrics from the regulator env
        if hasattr(self.outer, "env") and hasattr(self.outer.env, "last_metrics"):
            env_metrics = self.outer.env.last_metrics
            if env_metrics:
                import numpy as np
                metrics["total_fines"] = sum(m.get("total_fines", 0.0) for m in env_metrics)
                metrics["mean_fish"] = float(np.mean([m.get("mean_fish", 0.0) for m in env_metrics]))
                metrics["min_fish"] = float(min(m.get("min_fish", 1.0) for m in env_metrics))
                metrics["mean_collapse_rate"] = float(np.mean([m.get("collapse_rate", 0.0) for m in env_metrics]))

        self.es_metrics_history.append(metrics)

    def _save_es_metrics_plot(self) -> None:
        """Save ES metrics plot (fines, fish, collapse rate)."""
        if not self.output_dir or not self.es_metrics_history:
            return

        try:
            from pathlib import Path

            from examples.bilevel_fishery.visualization import plot_es_metrics

            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            plot_es_metrics(
                self.es_metrics_history,
                save_path=str(output_path / "es_metrics.png"),
            )
            logger.info("[Bilevel] Saved ES metrics plot")

        except Exception as e:
            logger.warning("[Bilevel] Failed to save ES metrics plot: %s", e)

    def _save_parameter_plots(self) -> None:
        """Save parameter evolution and fitness plots."""
        if not self.output_dir or not self.population_history:
            return

        try:
            from pathlib import Path

            from examples.bilevel_fishery.visualization import (
                plot_fitness_vs_parameters,
                plot_parameter_evolution,
            )

            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Get optimize_params from mechanism space if available
            optimize_params = None
            if hasattr(self, "mechanism_space") and hasattr(self.mechanism_space, "optimize_params"):
                optimize_params = self.mechanism_space.optimize_params

            plot_fitness_vs_parameters(
                self.population_history,
                save_path=str(output_path / "fitness_vs_params.png"),
                optimize_params=optimize_params,
            )
            logger.info("[Bilevel] Saved fitness vs parameters plot")

            plot_parameter_evolution(
                self.population_history,
                save_path=str(output_path / "param_evolution.png"),
                optimize_params=optimize_params,
            )
            logger.info("[Bilevel] Saved parameter evolution plot")

        except Exception as e:
            logger.warning("[Bilevel] Failed to save parameter plots: %s", e)
