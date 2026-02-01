import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Self

import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import register_env

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.utils import generate_uuid
from core.world.base import World

# TODO override environment to attach docstrings


@dataclass
class AgentSpec:
    count: int
    policy: str
    observation_space: Space
    action_space: Space


class RayOptimizerConfig(OptimizerConfig):
    # TODO review this
    # must be overriden in subclasses
    algo_class: type[Algorithm] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")
        super().__init__(opt_class=RayOptimizer)

        # TODO termporary setting until find out how to share world context accross runners
        self._cfg_ops: list[Callable[[AlgorithmConfig], AlgorithmConfig]] = []
        self.rllib_cfg: AlgorithmConfig | None = None
        self.agent_specs: Optional[dict] = None  # TODO default
        self.world_name: Optional[str] = None
        self.eval_episodes: Optional[int] = None
        self.eval_base_seed: Optional[int] = None
        self.rollout_fragment_length: Optional[int] = None

    def rllib_config_mutator(fn):
        def wrapper(self, *args, **kwargs):
            self._cfg_ops.append(lambda cfg: fn(cfg, *args, **kwargs))
            return self

        return wrapper

    @rllib_config_mutator
    def validate(cfg, **kwargs) -> None:
        return cfg.validate(**kwargs)

    @rllib_config_mutator
    def get_config_for_module(cfg, **kwargs) -> None:
        return cfg.get_config_for_module(**kwargs)

    @rllib_config_mutator
    def python_environment(cfg, **kwargs) -> None:
        """Sets the config's python environment settings.

        Args:
            extra_python_environs_for_driver: Any extra python env vars to set in the
                algorithm's process, e.g., {"OMP_NUM_THREADS": "16"}.
            extra_python_environs_for_worker: The extra python environments need to set
                for worker processes.
        """
        return cfg.python_environment(**kwargs)

    @rllib_config_mutator
    def resources(cfg, **kwargs) -> None:
        """Specifies resources allocated for an Algorithm and its ray actors/workers."""
        return cfg.resources(**kwargs)

    @rllib_config_mutator
    def framework(cfg, **kwargs) -> None:
        """Sets the config's DL framework settings."""
        return cfg.framework(**kwargs)

    @rllib_config_mutator
    def api_stack(cfg, **kwargs) -> None:
        """Sets the config's API stack settings."""
        return cfg.api_stack(**kwargs)

    def model(self, **kwargs) -> Self:
        """Sets the model configuration."""
        def _set_model(cfg):
            cfg.model.update(kwargs)
            return cfg
        self._cfg_ops.append(_set_model)
        return self

    @rllib_config_mutator
    def env_runners(cfg, **kwargs) -> None:
        """Sets the rollout worker configuration."""
        return cfg.env_runners(**kwargs)

    @rllib_config_mutator
    def learners(cfg, **kwargs) -> None:
        """Sets LearnerGroup and Learner worker related configurations."""
        return cfg.learners(**kwargs)

    @rllib_config_mutator
    def callbacks(cfg, **kwargs) -> None:
        """Sets the callbacks configuration.
        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.callbacks(**kwargs)

    # def evaluation(self, *, episodes: int = None, rollout_fragment_length: int, base_seed: Optional[int]=None, **kwargs) -> Self:
    #     if episodes is not None:
    #         self.eval_episodes = episodes
    #     if base_seed is not None:
    #         self.eval_base_seed = base_seed
    #     # TODO to infer from horizon
    #     if rollout_fragment_length is not None:
    #         self.rollout_fragment_length = rollout_fragment_length
    #     return self

    @rllib_config_mutator
    def evaluation(cfg, **kwargs) -> None:
        return cfg.evaluation(**kwargs)

    @rllib_config_mutator
    def offline_data(cfg, **kwargs) -> None:
        return cfg.offline_data(**kwargs)

    @rllib_config_mutator
    def multi_agent(cfg, **kwargs) -> None:
        """Sets the config's multi-agent settings."""
        return cfg.multi_agent(**kwargs)

    @rllib_config_mutator
    def reporting(cfg, **kwargs) -> None:
        """Sets the config's reporting settings.
        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.reporting(**kwargs)

    @rllib_config_mutator
    def checkpointing(cfg, **kwargs) -> None:
        return cfg.checkpointing(**kwargs)

    @rllib_config_mutator
    def fault_tolerance(cfg, **kwargs) -> None:
        """Sets the config's fault tolerance settings.
        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.fault_tolerance(**kwargs)

    @rllib_config_mutator
    def rl_module(cfg, **kwargs) -> None:
        """Sets the config's RLModule settings.
        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.rl_module(**kwargs)

    @rllib_config_mutator
    def experimental(cfg, **kwargs) -> None:
        """Sets the config's experimental settings.
        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.experimental(**kwargs)

    def _apply_agents_to_rllib(self) -> list[AgentID]:
        policies = {}
        agent_type_map = {}
        agents: list[AgentID] = []
        observation_spaces = {}
        action_spaces = {}

        for agent_type, spec in self.agent_specs.items():
            obs_space = spec.get("observation_space")
            act_space = spec.get("action_space")

            policies[spec.get("policy")] = (
                None,
                spec.get("observation_space"),
                spec.get("action_space"),
                {},
            )

            for i in range(spec.get("count")):
                agent_id = f"{agent_type}:{i}"
                agents.append(agent_id)
                agent_type_map[agent_id] = spec.get("policy")
                observation_spaces[agent_id] = obs_space
                action_spaces[agent_id] = act_space

        self.env_config.update({"observation_spaces": observation_spaces})
        self.env_config.update({"action_spaces": action_spaces})

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return agent_type_map[agent_id]

        self.rllib_cfg = self.rllib_cfg.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(set(agent_type_map.values())),
        )
        return agents

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict[str, AgentSpec]) -> Self:
        self.agent_specs = agents
        return self

    @override(OptimizerConfig)
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        world_name: Optional[str] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ):
        if self.rllib_cfg is None:
            self.rllib_cfg = self.algo_class.get_default_config()
            for op in self._cfg_ops:
                self.rllib_cfg = op(self.rllib_cfg)

        if self.opt_class is None:
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
        if self.agent_specs:
            agents = self._apply_agents_to_rllib()

        def env_creator(env_ctx):
            return self._env_creator(
                world=world, opt_id=opt_id, agents=agents, **dict(env_ctx)
            )

        register_env(env_name, env_creator)
        self.rllib_cfg = self.rllib_cfg.environment(
            env=env_name, env_config=self.env_config
        )

        # Building defferred to policyActor
        # algo = self.rllib_cfg.build_algo(**kwargs)

        cfg = self.copy(copy_frozen=True)
        # TODO do not give world to ray optimizer. temp solution until environment factory
        opt = RayOptimizer(config=cfg, world=world)

        # TODO refactor to env Factory later
        opt.world = world

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt.set_id(opt_id)

        return opt

    @rllib_config_mutator
    @override(OptimizerConfig)
    def freeze(cfg, **kwargs) -> None:
        return cfg.freeze(**kwargs)

    @rllib_config_mutator
    @override(OptimizerConfig)
    def training(cfg, **kwargs) -> Self:
        """Sets the training related configuration."""
        return cfg.training(**kwargs)
