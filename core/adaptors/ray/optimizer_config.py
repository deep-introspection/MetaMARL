from typing import Optional, Self

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.envs.base import BaseEnv
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.world.base import World


class RayOptimizerConfig(OptimizerConfig):
    # TODO review this
    # must be overriden in subclasses
    algo_class: type[Algorithm] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")
        super().__init__(opt_class=RayOptimizer)

        self.ray_cfg: AlgorithmConfig = (
            self.algo_class.get_default_config()
            .environment(None)
            .env_runners(num_env_runners=0)
        )

    def validate(self) -> None:
        self.ray_cfg.validate()

    def get_config_for_module(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.get_config_for_module(**kwargs)

    def python_environment(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.python_environment(**kwargs)

    # @override(OptimizerConfig)
    def resources(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.resources(**kwargs)
        return self

    def framework(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.framework(**kwargs)
        return self

    def api_stack(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.api_stack(**kwargs)
        return self

    def env_runners(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.env_runners(**kwargs)
        return self

    def learners(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.learners(**kwargs)
        return self

    def callbacks(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.callbacks(**kwargs)
        return self

    def evaluation(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.evaluation(**kwargs)
        return self

    def offline_data(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.offline_data(**kwargs)
        return self

    def multi_agent(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.multi_agent(**kwargs)
        return self

    def reporting(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.reporting(**kwargs)
        return self

    def reporting(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.reporting(**kwargs)
        return self

    def checkpointing(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.checkpointing(**kwargs)
        return self

    def fault_tolerance(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.fault_tolerance(**kwargs)
        return self

    def rl_module(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.rl_module(**kwargs)
        return self

    def experimental(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.experimental(**kwargs)
        return self

    @override(OptimizerConfig)
    def build_optimizer(self, *, world: Optional[World] = None, inner_opt: Optional[Optimizer] = None, **kwargs):
        cfg = self.copy(copy_frozen=True)
        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")
        
        algo = self.ray_cfg.build(**kwargs)
        opt = RayOptimizer(algo=algo, config=cfg)

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt_id = world.register_optimizer(opt)
            opt.set_id(opt_id)

        env = cfg._env_creator(world=world, inner_opt=inner_opt)
        env.set_opt_id(opt.id)
        opt.env = env
        
        return 

    @override(OptimizerConfig)
    def freeze(self) -> None:
        self.ray_cfg.freeze()

    @override(OptimizerConfig)
    def training(self, **kwargs) -> Self:
        self.ray_cfg = self.ray_cfg.training(**kwargs)
        return self

    @override(OptimizerConfig)
    def environment(self, *, env=None, **kwargs) -> Self:
        self.env = env
        return self
