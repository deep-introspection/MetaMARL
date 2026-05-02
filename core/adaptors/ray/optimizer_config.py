from functools import cached_property
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Self, TypeAlias, Any
import torch

import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import register_env

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.reporting.wandb import WandbReporter
from core.utils import generate_uuid
from core.world.base import World
# from core.adaptors.ray.protocols import PolicyResultMapper, from_new_api, from_old_api

# TODO override environment to attach docstrings


@dataclass
class AgentSpec:
    count: int
    policy: str
    observation_space: Space
    action_space: Space

FnID : TypeAlias = str

@dataclass
class RLlibConfigOp:
    fn : Callable[..., AlgorithmConfig]
    args : tuple[Any, ...] # TODO remove any
    kwargs : dict[ str, Any] # TODO remove any

    def __call__(self, cfg: AlgorithmConfig) -> AlgorithmConfig:
        return self.fn(cfg, *self.args, **self.kwargs)


class RayOptimizerConfig(OptimizerConfig):
    # TODO review this
    # must be overriden in subclasses
    algo_class: type[Algorithm] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")
        super().__init__(opt_class=RayOptimizer)

        # TODO termporary setting until find out how to share world context accross runners
        self._cfg_ops: dict[FnID, RLlibConfigOp] = {}
        self.rllib_cfg: AlgorithmConfig | None = None
        self.agent_specs: Optional[dict] = None  # TODO default
        self.world_name: Optional[str] = None
        self.eval_episodes: Optional[int] = None
        self.eval_base_seed: Optional[int] = None
        self.rollout_fragment_length: Optional[int] = None
        # self._result_mapper: ResultMapper = None

    # TODO let mutator accept an explicit ID
    def rllib_config_mutator(fn):
        def wrapper(self, *args, **kwargs):
            self._cfg_ops[fn.__name__] = RLlibConfigOp(
                    fn = fn,
                    args = args,
                    kwargs = kwargs
                )
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

        self._cfg_ops["model"] = RLlibConfigOp(
            fn = _set_model,
            args=(),
            kwargs={}
        )
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

    def _seeded_xavier_uniform(self, seed: Optional[int]):
        if seed is None:
            return "xavier_uniform_"
        counter = {"i": 0}
        def init_(tensor, **kwargs):
            layer_seed = int(seed) + counter["i"]
            counter["i"] += 1

            state = torch.random.get_rng_state()
            torch.manual_seed(layer_seed)
            torch.nn.init.xavier_uniform_(tensor, **kwargs)
            torch.random.set_rng_state(state)
        return init_

    def _apply_agents_to_rllib(self) -> list[AgentID]:
        policies = {}
        agent_type_map = {}
        agents: list[AgentID] = []
        observation_spaces = {}
        action_spaces = {}

        # Get number of envs and seeds
        num_envs = self.rllib_cfg.num_envs_per_env_runner or 1
        num_seeds = len(self.seed) if self.seed is not None else 1

        if num_envs % num_seeds != 0:
            raise ValueError(
                f"num_envs_per_env_runner={num_envs} must be divisible by num_seeds={num_seeds}"
            )
        
        # Get number of mechanisms (one policy per mechanism, per seed)
        num_mechanisms = num_envs // num_seeds

        module_specs = {}

        for agent_type, spec in self.agent_specs.items():
            obs_space = spec.get("observation_space")
            act_space = spec.get("action_space")
            base_policy = spec.get("policy")

            # TODO (nadinemgh) this does not guarantee tht different mechanism's policy will be
                # initiated with the same seed !
                # what we want :
                # run mechanism 0, seed 101
                # run mechanism 1, seed 101
                # run mechanism 2, seed 101
                # run mechanism 0, seed 202
                # run mechanism 1, seed 202
                # run mechanism 2, seed 202
            for seed_idx in range(num_seeds):
                seed_value = self.seed[seed_idx] if self.seed is not None else None
            
                for m_idx in range(num_mechanisms):
                    policy_id = f"{base_policy}_m{m_idx}_s{seed_idx}"
                    policies[policy_id] = (
                        None,
                        spec.get("observation_space"),
                        spec.get("action_space"),
                        {},
                    )

                    module_specs[policy_id] = RLModuleSpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config=DefaultModelConfig(
                            vf_share_layers=False,
                            fcnet_kernel_initializer=self._seeded_xavier_uniform(seed_value),
                            fcnet_bias_initializer="zeros_",
                            head_fcnet_kernel_initializer=self._seeded_xavier_uniform(seed_value),
                            head_fcnet_bias_initializer="zeros_",
                        ),
                    )

            for i in range(spec.get("count")):
                agent_id = f"{agent_type}:{i}"
                agents.append(agent_id)
                agent_type_map[agent_id] = base_policy
                observation_spaces[agent_id] = obs_space
                action_spaces[agent_id] = act_space
        
        self.rllib_cfg = self.rllib_cfg.rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs=module_specs
                )
            )

        self.env_config.update({"observation_spaces": observation_spaces})
        self.env_config.update({"action_spaces": action_spaces})

        def policy_mapping_fn(agent_id, episode, *_, **__):
            base_policy = agent_type_map[agent_id]

            # Old Api Stack Route to policy based on environment index
            env_idx = getattr(episode, "env_id", None)

            # New Api fallback
            if env_idx is None:
                env_idx = int(episode.id_.split("|")[0])

            if env_idx is None:
                raise RuntimeError(
                    "No environment index found on episode. "
                    "Expected episode.env_id (old stack) or "
                    "episode.custom_data['env_idx'] (new stack)."
                )
            mechanism_idx = int(env_idx) % num_mechanisms
            seed_idx = int(env_idx) // num_mechanisms
            return f"{base_policy}_m{mechanism_idx}_s{seed_idx}"

        all_policies = list(policies.keys())
        self.rllib_cfg = self.rllib_cfg.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=all_policies,
        )
        return agents

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict[str, AgentSpec]) -> Self:
        self.agent_specs = agents
        return self

    # lazy resolution : better encapsulation ?
    # @cached_property
    # def result_mapper(self) -> ResultMapper:
    #     return self._resolve_result_mapper()

    # def _resolve_result_mapper(cfg: AlgorithmConfig) -> ResultMapper:
    #     uses_new_stack = bool(
    #         getattr(cfg, "enable_rl_module_and_learner", False)
    #         and getattr(cfg, "enable_env_runner_and_connector_v2", False)
    #     )
    #     return from_new_api if uses_new_stack else from_old_api

    @override(OptimizerConfig)
    def build_optimizer(
        self,
        *,
        world: ActorHandle[World],
        world_name: Optional[str] = None,
        reporting: ActorHandle[WandbReporter],
    ):
        if self.rllib_cfg is None:
            self.rllib_cfg = self.algo_class.get_default_config()
            for op in self._cfg_ops.values():
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
        opt = RayOptimizer(config=cfg, world=world, reporting=reporting)

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
    def training(cfg, **kwargs) -> None:
        return cfg.training(**kwargs)
    
    @rllib_config_mutator
    def _debugging_rllib(cfg, seed: Optional[int] = None, **kwargs):
        return cfg.debugging(seed=seed, **kwargs)
    
    
    @override(OptimizerConfig)
    def debugging(
        self,
        *,
        seed: Optional[int] = None,
        num_seeds: int = 3,
        **kwargs,
    ) -> Self:
        """Sets the debugging related configuration."""
        
        super().debugging(seed=seed, num_seeds=num_seeds)
        if seed is not None:
            env_runners_op = self._cfg_ops.get("env_runners")

            if env_runners_op is not None:
                env_runners_op.kwargs["num_envs_per_env_runner"] = (
                    env_runners_op.kwargs.get("num_envs_per_env_runner", 1) * num_seeds
                )
                
        # Lazy construction
        return self._debugging_rllib(seed=seed, **kwargs)