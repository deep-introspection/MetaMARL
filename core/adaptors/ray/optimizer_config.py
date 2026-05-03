from functools import cached_property
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
from core.reporting.wandb import WandbReporter
from core.utils import generate_uuid
from core.world.base import World
# from core.adaptors.ray.protocols import PolicyResultMapper, from_new_api, from_old_api

# TODO override environment to attach docstrings


@dataclass
class AgentSpec:
    """Specification for a homogeneous group of agents sharing a single policy.

    Parameters
    ----------
    count : int
        Number of agents of this type in the environment.
    policy : str
        Base policy ID string.  One policy per mechanism will be derived from
        this (e.g. ``"fisher_0"``, ``"fisher_1"``, …).
    observation_space : Space
        Gymnasium observation space shared by all agents of this type.
    action_space : Space
        Gymnasium action space shared by all agents of this type.
    """

    count: int
    policy: str
    observation_space: Space
    action_space: Space


class RayOptimizerConfig(OptimizerConfig):
    """Configuration for a Ray/RLlib-backed optimizer.

    Acts as a fluent builder that accumulates RLlib ``AlgorithmConfig``
    mutations (stored in ``_cfg_ops``) and applies them lazily when
    ``build_optimizer`` is called.  Subclasses must set the class-level
    attribute ``algo_class`` to the concrete RLlib ``Algorithm`` class to
    use (e.g. ``APPO`` or ``PPO``).

    The deferred-mutation pattern avoids constructing the heavy
    ``AlgorithmConfig`` object until the full set of options is known,
    which is important when configs are assembled across multiple files
    or passed between processes.

    Parameters
    ----------
    (no constructor arguments — configuration is built via the fluent
    methods below, e.g. ``training()``, ``env_runners()``, etc.)

    Attributes
    ----------
    algo_class : type[Algorithm]
        RLlib Algorithm class; **must** be set on the concrete subclass.
    _cfg_ops : list[Callable[[AlgorithmConfig], AlgorithmConfig]]
        Ordered list of pending mutations to apply to the RLlib config.
    rllib_cfg : AlgorithmConfig or None
        The materialised RLlib config; ``None`` until ``build_optimizer``
        is called.
    agent_specs : dict or None
        Mapping of agent-type name to spec dict (see ``agents()``).
    world_name : str or None
        Identifier for the Ray actor that owns the simulation world.
    eval_episodes : int or None
        Number of evaluation episodes per ``evaluate()`` call.
    eval_base_seed : int or None
        Base random seed used to initialise evaluation environments.
    rollout_fragment_length : int or None
        Number of environment steps per rollout fragment.
    """

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
        # self._result_mapper: ResultMapper = None

    def rllib_config_mutator(fn):
        """Decorator that converts a config-mutation function into a fluent builder method.

        Wraps ``fn`` so that, instead of executing immediately, it is appended
        to ``self._cfg_ops`` as a deferred lambda.  Each deferred operation
        receives the ``AlgorithmConfig`` object and must return it (possibly
        modified) so that operations can be chained.

        The decorator is intentionally defined as a plain function (not a
        ``staticmethod``) so that it can be used directly in the class body
        during class construction.

        Parameters
        ----------
        fn : Callable[[AlgorithmConfig, ...], AlgorithmConfig]
            A function whose first positional argument is the
            ``AlgorithmConfig`` instance to mutate.

        Returns
        -------
        Callable
            A wrapper method that records the call as a pending operation and
            returns ``self`` for method chaining.
        """

        def wrapper(self, *args, **kwargs):
            self._cfg_ops.append(lambda cfg: fn(cfg, *args, **kwargs))
            return self

        return wrapper

    @rllib_config_mutator
    def validate(cfg, **kwargs) -> None:
        """Validate the accumulated RLlib configuration.

        Delegates to ``AlgorithmConfig.validate(**kwargs)``.  Call this after
        all other builder methods to catch configuration errors early.
        """
        return cfg.validate(**kwargs)

    @rllib_config_mutator
    def get_config_for_module(cfg, **kwargs) -> None:
        """Return the per-module config subset from the RLlib config.

        Delegates to ``AlgorithmConfig.get_config_for_module(**kwargs)``.
        """
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
        """Configure offline data settings in the RLlib config.

        Delegates to ``AlgorithmConfig.offline_data(**kwargs)``.
        """
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
        """Configure checkpoint settings in the RLlib config.

        Delegates to ``AlgorithmConfig.checkpointing(**kwargs)``.
        """
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
        """Materialise agent specs into the RLlib multi-agent configuration.

        For each agent type in ``agent_specs``, creates one policy per
        mechanism (``num_envs_per_env_runner``) following the naming
        convention ``"{base_policy}_{mechanism_index}"``.  The resulting
        ``policy_mapping_fn`` routes each agent to the policy that
        corresponds to its environment index, enabling independent
        per-mechanism policies within a single RLlib run.

        Also populates ``env_config`` with per-agent observation and action
        spaces so that the environment factory can resolve spaces at
        creation time without a direct reference to the config object.

        Returns
        -------
        list[AgentID]
            Flat list of all agent IDs (e.g. ``["fisher:0", "fisher:1"]``)
            in the order they were created.
        """
        policies = {}
        agent_type_map = {}
        agents: list[AgentID] = []
        observation_spaces = {}
        action_spaces = {}

        # Get number of mechanisms (one policy per mechanism)
        num_mechanisms = self.rllib_cfg.num_envs_per_env_runner or 1

        for agent_type, spec in self.agent_specs.items():
            obs_space = spec.get("observation_space")
            act_space = spec.get("action_space")
            base_policy = spec.get("policy")

            # Create one policy per mechanism
            for m in range(num_mechanisms):
                policy_id = f"{base_policy}_{m}"
                policies[policy_id] = (
                    None,
                    spec.get("observation_space"),
                    spec.get("action_space"),
                    {},
                )

            for i in range(spec.get("count")):
                agent_id = f"{agent_type}:{i}"
                agents.append(agent_id)
                agent_type_map[agent_id] = base_policy
                observation_spaces[agent_id] = obs_space
                action_spaces[agent_id] = act_space

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
            env_idx = int(env_idx) % num_mechanisms
            return f"{base_policy}_{env_idx}"

        all_policies = list(policies.keys())
        self.rllib_cfg = self.rllib_cfg.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=all_policies,
        )
        return agents

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict[str, AgentSpec]) -> Self:
        """Register the agent type specifications for multi-agent training.

        Parameters
        ----------
        agents : dict[str, AgentSpec]
            Mapping from agent-type name (e.g. ``"fisher"``) to its
            ``AgentSpec`` describing count, policy base ID, and spaces.

        Returns
        -------
        Self
            This config object for method chaining.
        """
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
        """Construct and return a configured ``RayOptimizer`` instance.

        Applies all deferred ``_cfg_ops`` to the base ``AlgorithmConfig``,
        registers a unique Gymnasium environment with Ray's global registry
        (so that remote env-runner workers can instantiate it), wires up the
        multi-agent policy mapping, and builds the ``RayOptimizer`` without
        yet constructing the heavy ``Algorithm`` object (that is deferred to
        the ``PolicyActor``).

        Parameters
        ----------
        world : ActorHandle[World]
            Ray remote actor handle for the simulation world.  Used to
            generate a unique optimizer ID and to pass context to the
            environment factory.
        world_name : str, optional
            Human-readable name for the world actor; required when
            ``world`` is not ``None``.
        reporting : ActorHandle[WandbReporter]
            Ray remote actor handle for the W&B reporting backend.

        Returns
        -------
        RayOptimizer
            A fully configured but not yet started optimizer instance.

        Raises
        ------
        ValueError
            If ``opt_class`` is ``None`` or ``world_name`` is missing when
            ``world`` is provided.
        """
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
    def training(cfg, **kwargs) -> Self:
        """Sets the training related configuration."""
        return cfg.training(**kwargs)
