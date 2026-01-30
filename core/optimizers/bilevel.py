import uuid
from typing import Optional, Self

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.world.base import World


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

    def training(self, *, outer_iters: int, seed=None, **kwargs) -> Self:
        if outer_iters is not None:
            self.outer_iters = outer_iters
        if seed is not None:
            self.seed = seed
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
        self.outer_iters = config.outer_iters
        self.outer = outer
        self.inner = inner

    def run(self) -> None:
        for _ in range(self.outer_iters):
            self.outer.run()
