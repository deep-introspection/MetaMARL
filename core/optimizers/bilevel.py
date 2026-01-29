import uuid
from typing import Optional, Self

from core.adaptors.ray.config import DeviceType, RayRuntimeConfig
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.world.base import World


class BilevelConfig(OptimizerConfig):
    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or BilevelOptimizer)
        self.outer_cfg = None
        self.inner_cfg = None
        self.outer_iters = 10
        self.seed = None
        self.world_name: Optional[str] = None
        self.ray_cfg = None

    def inner(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.inner_cfg = cfg
        return self

    def outer(self, cfg: OptimizerConfig = None) -> Self:
        if cfg is not None:
            self.outer_cfg = cfg
        return self

    def world(self, *world_name: str, **kwargs) -> Self:
        if world_name is not None:
            self.world_name = f"{world_name}_{uuid.uuid4().hex[:8]}"
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
        world = World.options(name=self.world_name).remote()

        if self.ray_cfg is not None:
            self.ray_cfg.initialize()
        inner_opt = self.inner_cfg.build_optimizer(
            world=world, world_name=self.world_name
        )
        outer_opt = self.outer_cfg.build_optimizer(world=world, inner_opt=inner_opt)

        return self.opt_class(cfg=self, outer=outer_opt, inner=inner_opt)


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
