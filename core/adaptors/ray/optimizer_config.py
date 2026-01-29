import uuid
from typing import Optional, Self

import ray
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.registry import register_env

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.utils import generate_uuid
from core.world.base import World


class RayOptimizerConfig(OptimizerConfig):
    # TODO review this
    # must be overriden in subclasses
    algo_class: type[Algorithm] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")
        super().__init__(opt_class=RayOptimizer)

        # TODO termporary setting until find out how to share world context accross runners
        self.ray_cfg: AlgorithmConfig = (
            self.algo_class.get_default_config()
            .environment(None)
            .env_runners(
                num_env_runners=0,
                create_local_env_runner=True,
                create_env_on_local_worker=True,
            )
        )
        self.agent_specs: Optional[dict] = None  # TODO default
        self.world_name: Optional[str] = None

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

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict) -> Self:
        # TODO : default agent
        if agents is not None:
            self.agent_specs = agents
            policies = {}
            agent_type_map = {}

            for agent_type, spec in self.agent_specs.items():
                policy_name = spec["policy"]

                policies[policy_name] = (
                    None,
                    spec["observation_space"],
                    spec["action_space"],
                    {},
                )

                for i in range(spec["count"]):
                    agent_type_map[f"{agent_type}:{i}"] = policy_name

            def policy_mapping_fn(agent_id, *args, **kwargs):
                return agent_type_map[agent_id]

            self.ray_cfg = self.ray_cfg.multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=list(set(agent_type_map.values())),
            )

    # @override(OptimizerConfig)
    # def _env_creator(
    #     self,
    #     *,
    #     world: Optional[World] = None,
    #     opt_id: Optional[OptimizerID] = None,
    #     **kwargs,
    # ) -> BaseEnv:
    #     return self.env(
    #         world=world,
    #         opt_id=opt_id,
    #         train_iters=self.train_iters,
    #         eval_iters=self.eval_iters,
    #         **self.env_config,
    #     )

    @override(OptimizerConfig)
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        world_name: Optional[str] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ):
        cfg = self.copy(copy_frozen=True)
        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        env_name = f"regulated_env_{uuid.uuid4().hex}"

        if world is not None:
            if world_name is None:
                raise ValueError(
                    "world_name must be provided when using Ray world actor"
                )
            self.world_name = world_name
            registry = ray.get(world.get_opt_registry.remote())
            # Register the new ID and get the result
            opt_id = ray.get(
                world._set_new_opt_id.remote(opt_id=generate_uuid(registry))
            )

        def env_creator(env_ctx):
            world = ray.get_actor(env_ctx["world_name"])
            opt_id = env_ctx["opt_id"]

            return cfg._env_creator(
                world=world,
                inner_opt=inner_opt,
                opt_id=opt_id,
                agent_populations={k: v["count"] for k, v in self.agent_specs.items()}
                if self.agent_specs
                else None,
                **{
                    k: v
                    for k, v in env_ctx.items()
                    if k not in ("world_name", "opt_id")
                },
            )

        register_env(env_name, env_creator)
        self.ray_cfg = self.ray_cfg.environment(
            env=env_name, env_config={"world_name": self.world_name, "opt_id": opt_id}
        )

        algo = self.ray_cfg.build_algo(**kwargs)
        opt = RayOptimizer(algo=algo, config=cfg)

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt.set_id(opt_id)

        return opt

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
