import logging
import uuid
from typing import Any, Literal, Optional, Self

import ray
from ray.actor import ActorHandle

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.reporting.wandb import WandbReporter
from core.reporting.enums import ReporterType
from core.world.base import World

logger = logging.getLogger(__name__)


class BilevelConfig(OptimizerConfig):
    """Fluent builder for the bilevel optimization graph.

    ``BilevelConfig`` follows the method-chaining builder pattern: each
    configuration method returns ``self`` so that calls can be composed in a
    single expression.  Calling :meth:`build_optimizer` finalises the
    configuration, initialises Ray, optionally spawns a WandB reporter actor,
    and wires the outer ES optimizer to the inner RL optimizer.

    The typical usage pattern is::

        optimizer = (
            BilevelConfig()
            .inner(PPOptimizerConfig()...)
            .outer(ESConfig()...)
            .world(world_name="fishery_run")
            .mechanism(space=mech_space)
            .training(outer_iters=50)
            .ray(device="cpu", num_cpus=8)
            .reporting("wandb", project_name="bilevel_fishery")
            .build_optimizer()
        )

    Attributes
    ----------
    outer_cfg : OptimizerConfig or None
        Configuration for the outer (meta) optimizer — typically an
        :class:`~core.optimizers.es.config.ESConfig`.
    inner_cfg : OptimizerConfig or None
        Configuration for the inner (policy) optimizer — typically a
        :class:`~core.optimizers.ppo.config.PPOptimizerConfig` or
        :class:`~core.optimizers.appo.config.APPOptimizerConfig`.
    outer_iters : int
        Number of outer-loop iterations (ES generations) to run.
        Defaults to 10.
    seed : int or None
        Global random seed (currently reserved for future use).
    world_name : str or None
        Human-readable name for the Ray named actor group; a random hex
        suffix is appended at build time to ensure uniqueness.
    ray_cfg : RayRuntimeConfig or None
        Ray cluster configuration.  If ``None``, a default CPU-only
        ``RayRuntimeConfig`` is used.
    mechanism_space : MechanismSpace or None
        Search space over regulatory mechanisms (quotas, fines, thresholds,
        ban periods).  Its dimension is forwarded to the outer optimizer.
    default_mechanism : Mechanism or None
        Fallback mechanism used before the first ES generation completes.
    output_dir : str or None
        Optional directory for writing checkpoints and artefacts.
    wandb_cfg : dict or None
        WandB initialisation parameters set via :meth:`reporting`.
    """

    def __init__(self, opt_class=None):
        """Initialise ``BilevelConfig`` with sensible defaults.

        Parameters
        ----------
        opt_class : type[BilevelOptimizer], optional
            Concrete optimizer class to instantiate in :meth:`build_optimizer`.
            Defaults to :class:`BilevelOptimizer`.
        """
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
        """Ray actor handle for the WandB reporter, or ``None`` if not configured.

        Returns
        -------
        ActorHandle[WandbReporter] or None
            A remote handle to the WandB reporter actor created during
            :meth:`build_optimizer`, or ``None`` when WandB reporting has not
            been configured via :meth:`reporting`.
        """
        return self._reporter

    def inner(self, cfg: OptimizerConfig = None) -> Self:
        """Set the inner (policy-level) optimizer configuration.

        The inner optimizer is responsible for training N fishing agents via
        multi-agent RL under a fixed regulatory mechanism proposed by the outer
        optimizer.  Typically this is a
        :class:`~core.optimizers.ppo.config.PPOptimizerConfig` or
        :class:`~core.optimizers.appo.config.APPOptimizerConfig`.

        Parameters
        ----------
        cfg : OptimizerConfig, optional
            Configuration for the inner RL optimizer.  If ``None``, the
            existing ``inner_cfg`` is left unchanged.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
        if cfg is not None:
            self.inner_cfg = cfg
        return self

    def outer(self, cfg: OptimizerConfig = None) -> Self:
        """Set the outer (mechanism-search) optimizer configuration.

        The outer optimizer searches the mechanism space (quotas, fines, stock
        thresholds, ban periods) using Evolution Strategies.  Its candidate
        population size is automatically synchronised with the inner optimizer's
        batch capacity during :meth:`build_optimizer`.  Typically an
        :class:`~core.optimizers.es.config.ESConfig`.

        Parameters
        ----------
        cfg : OptimizerConfig, optional
            Configuration for the outer ES optimizer.  If ``None``, the
            existing ``outer_cfg`` is left unchanged.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
        if cfg is not None:
            self.outer_cfg = cfg
        return self

    def world(self, *, world_name: str, **kwargs) -> Self:
        """Configure the shared Ray World actor for this run.

        The World actor acts as a shared message bus / context store between
        the outer and inner optimizers.  A unique 8-character hex suffix is
        appended to ``world_name`` at call time to prevent name collisions
        between concurrent runs.

        Parameters
        ----------
        world_name : str
            Human-readable base name for the Ray named actor group.  The
            final name stored in ``self.world_name`` will be
            ``"<world_name>_<hex8>"``.
        **kwargs : Any
            Reserved for future options; currently unused.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
        if world_name is not None:
            self.world_name = f"{world_name}_{uuid.uuid4().hex[:8]}"
        return self

    def mechanism(
        self, *, space: MechanismSpace, default: Mechanism = None, **kwargs
    ) -> Self:
        """Configure the regulatory mechanism search space.

        The mechanism space defines the dimensionality and bounds of the ES
        search problem (e.g. quota levels, fine magnitudes, stock thresholds,
        seasonal ban durations).  Its dimension is forwarded to the outer
        optimizer's config at build time.

        Parameters
        ----------
        space : MechanismSpace
            The mechanism space object that defines the search domain and
            provides a ``default()`` factory.
        default : Mechanism, optional
            Default mechanism used as the initial condition for the inner
            optimizer before the first ES generation.  Falls back to
            ``space.default()`` if not provided.
        **kwargs : Any
            Reserved for future options; currently unused.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
        if space is not None:
            self.mechanism_space = space
            self.default_mechanism = default or space.default()
        return self

    def training(
        self, *, outer_iters: int, seed=None, output_dir: str | None = None, **kwargs
    ) -> Self:
        """Set bilevel training loop hyperparameters.

        Parameters
        ----------
        outer_iters : int
            Total number of outer-loop iterations (ES generations) to execute
            before the run terminates.  Early stopping may occur before this
            limit if the outer optimizer converges.
        seed : int, optional
            Global random seed.  Forwarded to the outer optimizer's RNG when
            set.
        output_dir : str or None, optional
            Filesystem path for writing run artefacts (checkpoints, logs).
            ``None`` disables local output.
        **kwargs : Any
            Additional keyword arguments forwarded to the parent
            :meth:`~core.optimizers.config.OptimizerConfig.training`.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
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
        """Configure the Ray cluster runtime used for distributed execution.

        This method constructs a :class:`~core.adaptors.ray.runtime.RayRuntimeConfig`
        and stores it for use in :meth:`build_optimizer`, which calls
        :meth:`~core.adaptors.ray.runtime.RayRuntime.ensure_initialized` before
        building any Ray actors.

        Parameters
        ----------
        device : {"cpu", "gpu"}, optional
            Target device for the Ray workers.  Defaults to ``"cpu"``.
        num_cpus : int, optional
            Total number of CPUs to allocate to the Ray cluster.  ``None``
            lets Ray auto-detect available CPUs.
        num_gpus : int, optional
            Total number of GPUs to allocate.  ``None`` disables GPU usage.
        omp_threads : int, optional
            Value for the ``OMP_NUM_THREADS`` environment variable inside Ray
            workers.  Defaults to ``1`` to prevent OpenMP thread oversubscription.
        logging_level : str, optional
            Ray internal logging level (e.g. ``"ERROR"``, ``"WARNING"``).
            Defaults to ``"ERROR"`` to suppress verbose Ray logs.
        runtime_env : dict, optional
            Ray runtime environment specification (pip packages, environment
            variables, working directory, etc.).  See Ray documentation for
            the full schema.
        **kwargs : Any
            Additional keyword arguments forwarded to ``ray.init()``.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).
        """
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
        """Configure experiment reporting / telemetry.

        Currently only WandB reporting is supported.  When ``"wandb"`` is
        selected, a :class:`~core.reporting.wandb.WandbReporter` Ray actor is
        created during :meth:`build_optimizer` and passed to both the inner and
        outer optimizers.

        Parameters
        ----------
        reporter : {"wandb", "local"}
            Reporting backend.  ``"local"`` is reserved for future use and
            raises :class:`TypeError` if specified.
        project_name : str
            WandB project name used for grouping runs in the WandB dashboard.
        config : dict, optional
            Arbitrary hyperparameter dictionary logged to the WandB run at
            initialisation.  Merged with bilevel-level metadata
            (``outer_iters``, ``world_name``).
        settings_dict : dict, optional
            WandB ``Settings`` overrides (e.g. ``{"mode": "offline"}``).
            Defaults to an empty dict if not provided.

        Returns
        -------
        Self
            This ``BilevelConfig`` instance (for method chaining).

        Raises
        ------
        TypeError
            If ``reporter="local"`` is requested (not yet implemented).
        """
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
        """Finalise the configuration and construct the :class:`BilevelOptimizer`.

        Performs the following steps in order:

        1. Ensures Ray is initialised (idempotent).
        2. Spawns a named :class:`~core.reporting.wandb.WandbReporter` Ray actor
           if WandB reporting was configured.
        3. Creates the shared :class:`~core.world.base.World` Ray actor.
        4. Copies inner and outer configs and injects mechanism-space metadata.
        5. Builds the inner RL optimizer via ``inner_cfg.build_optimizer()``.
        6. Builds the outer ES optimizer via ``outer_cfg.build_optimizer()``.
        7. Synchronises the outer optimizer's population size with the inner
           optimizer's batch capacity.
        8. Returns a fully wired :class:`BilevelOptimizer`.

        Returns
        -------
        BilevelOptimizer
            A ready-to-run bilevel optimizer with inner and outer sub-optimizers
            connected through the shared World actor.
        """
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

        inner_opt = inner_cfg.build_optimizer(
            world=world, world_name=self.world_name, reporting=self.reporter
        )
        outer_opt = outer_cfg.build_optimizer(world=world, inner_opt=inner_opt)

        # what if outer_opt does not have that property ??
        # override outer_opt population size with inner_opt batch_size
        # set inner batch capacity to be the same as inner for batch sampling
        outer_opt.batch_capacity = inner_opt.batch_capacity
        return self.opt_class(config=self, outer=outer_opt, inner=inner_opt)


class BilevelOptimizer(Optimizer):
    """Top-level bilevel optimizer that coordinates the outer ES and inner RL loops.

    ``BilevelOptimizer`` implements the bilevel optimisation structure for
    sustainable fishery regulation:

    * **Outer loop (ES):** At each generation the outer
      :class:`~core.optimizers.es.optimizer.ESOptimizer` samples a population
      of candidate regulatory mechanisms from a search distribution and
      evaluates them by delegating to the inner loop.
    * **Inner loop (RL):** For each candidate mechanism the inner Ray RLlib
      optimizer trains N fishing agents (PPO or APPO) and returns aggregate
      episode fitness scores back to the outer optimizer.

    The optimisation terminates after ``max_outer_iters`` generations or
    earlier if the outer optimizer signals convergence.

    Attributes
    ----------
    world_name : str
        Name of the shared Ray World actor group.
    max_outer_iters : int
        Maximum number of outer-loop generations before forced termination.
    outer : Optimizer
        The outer (ES) optimizer instance.
    inner : Optimizer
        The inner (RL) optimizer instance.
    output_dir : str or None
        Filesystem path for run artefacts, or ``None`` if disabled.
    mechanism_space : MechanismSpace or None
        The regulatory mechanism search space.
    outer_iter : int
        Index of the current (or last completed) outer-loop iteration.
    converged : bool
        ``True`` if the outer optimizer signalled early convergence.
    best_trajectory : list[dict] or None
        Trajectory (sequence of environment observations/rewards) produced by
        the best mechanism found so far.
    all_trajectories : list[tuple[int, float, list[dict]]]
        History of ``(iteration, fitness, trajectory)`` tuples for all
        completed outer iterations.
    population_history : list[tuple[int, list]]
        History of ``(iteration, population)`` tuples for post-hoc analysis.
    es_metrics_history : list[dict]
        Sequence of per-generation ES metric dictionaries.
    """

    def __init__(self, config: BilevelConfig, outer: Optimizer, inner: Optimizer):
        """Initialise the bilevel optimizer.

        Parameters
        ----------
        config : BilevelConfig
            Fully built bilevel configuration object.
        outer : Optimizer
            Pre-built outer (ES) optimizer.
        inner : Optimizer
            Pre-built inner (RL) optimizer.
        """
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

        # TODO avoid hardcoding prefix
        # self.wandb_run.define_metric("ppo/ppo_step")
        # self.wandb_run.define_metric("ppo/*", step_metric="ppo/ppo_step")

    def run(self) -> dict:
        """Execute the bilevel optimisation loop.

        Iterates over at most ``max_outer_iters`` outer generations.  At each
        generation the outer ES optimizer proposes a new population of
        mechanisms, evaluates them via the inner RL optimizer, and updates its
        search distribution.  The loop terminates early when the outer optimizer
        sets ``converged=True`` in its returned metrics.

        Returns
        -------
        dict
            A summary dictionary with the following keys:

            ``"converged"`` : bool
                Whether the outer optimizer converged before exhausting
                ``max_outer_iters``.
            ``"outer_iters"`` : int
                Number of completed outer iterations.
            ``"best_fitness"`` : float
                Highest fitness score observed across all generations.
            ``"best_mechanism"`` : numpy.ndarray
                Parameter vector of the best regulatory mechanism found.
            ``"best_trajectory"`` : list[dict] or None
                Environment trajectory produced by the best mechanism.
            ``"all_trajectories"`` : list[tuple[int, float, list[dict]]]
                Full trajectory history indexed by outer iteration.
            ``"population_history"`` : list[tuple[int, list]]
                ES population history for post-hoc analysis.
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
