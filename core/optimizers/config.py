from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING, Optional, Self, Type, Union

import ray
from gymnasium import Space
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.envs.base import BaseEnv
from core.types import EnvConfigDict, EnvType
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    """Private base class for all configuration objects in the optimizer graph.

    Provides a minimal interface that every config subclass must implement.
    """

    def to_dict(self) -> dict:
        """Serialise this configuration to a plain dictionary.

        Returns
        -------
        dict
            A dictionary representation of the configuration.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
    """Abstract base configuration for all optimizer nodes.

    ``OptimizerConfig`` follows the same fluent builder pattern used across the
    codebase: each mutating method returns ``self`` so calls can be chained.
    Concrete subclasses (e.g. :class:`~core.optimizers.es.config.ESConfig`,
    :class:`~core.adaptors.ray.optimizer_config.RayOptimizerConfig`) extend
    this class with algorithm-specific parameters.

    The config object can be frozen via :meth:`freeze` once an optimizer has
    been built, making it read-only for the lifetime of the run.

    Attributes
    ----------
    opt_class : type[Optimizer] or None
        Concrete optimizer class to instantiate in :meth:`build_optimizer`.
    _is_frozen : bool
        When ``True``, attribute assignment raises :class:`AttributeError`.
    env : str or type or None
        Environment class or registered string identifier.
    env_config : dict
        Keyword arguments forwarded to the environment constructor.
    horizon : int or None
        Episode length (number of environment steps per rollout).
    seed : int or None
        Global random seed for reproducibility.
    evaluation_config : OptimizerConfig or None
        Optional sub-configuration used during evaluation rollouts.
    stats_cls_lookup : dict
        Mapping used by the Ray ``MetricsLogger`` to resolve statistic classes.
    """

    # TODO registry to allow opt_class str
    # TODO runtime checking of opt_class
    def __init__(self, opt_class: Optional[Type[Optimizer]] = None):
        """Initialise an ``OptimizerConfig`` instance.

        Parameters
        ----------
        opt_class : type[Optimizer], optional
            Optimizer class that this config belongs to.  When provided,
            :meth:`build_optimizer` instantiates ``opt_class(config=self)``.
        """
        self.opt_class = opt_class

        # -- lifecycle --
        self._is_frozen = False

        # --- world / environment ---
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.horizon: int = None  # TODO default value

        # --- training ---
        self.seed: int = None

        # --- eval ---
        self.evaluation_config: Optional["OptimizerConfig"] = None

        # TODO
        # --- reporting ---
        self.stats_cls_lookup = DEFAULT_STATS_CLS_LOOKUP

    def __setattr__(self, name, value):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if name not in ["_is_frozen"]:
                raise AttributeError(
                    f"Cannot set attribute ({name}) of an already frozen "
                    "OptimizerConfig!"
                )
        super().__setattr__(name, value)

    # TODO generalize this function
    def _merge_env_config(self, extra: dict) -> Self:
        """Merge additional key-value pairs into the environment config dict.

        Existing keys are preserved; ``extra`` values take precedence on
        collision.

        Parameters
        ----------
        extra : dict
            Key-value pairs to merge into ``self.env_config``.

        Returns
        -------
        Self
            This config instance (mutated in-place; returned for chaining).
        """
        self.env_config = {
            **(self.env_config or {}),
            **extra,
        }
        return self

    # TODO freezing for nested configs
    def freeze(self) -> None:
        """Freeze this config object, such that no attributes can be set anymore.

        Optimizers should use this method to make sure their config objects
        remain read-only after this.
        """
        if self._is_frozen:
            return
        self._is_frozen = True

    def copy(self, copy_frozen: Optional[bool] = None) -> Self:
        """Creates a deep copy of this config and (un)freezes if necessary.

        Args:
            copy_frozen: Whether the created deep copy is frozen or not, If None,
                keep the same frozen status that 'self' currently has.

        Returns:
            A deep copy of 'self' that is (un)frozen.
        """
        cp = copy.deepcopy(self)
        if copy_frozen is True:
            cp.freeze()
        elif copy_frozen is False:
            cp._is_frozen = False
            if isinstance(cp.evaluation_config, OptimizerConfig):
                cp.evaluation_config._is_frozen = False
        return cp

    # TODO review this
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialise a configuration object from a plain dictionary.

        Only keys that correspond to existing attributes are applied; unknown
        keys are silently ignored.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict` or a compatible mapping.

        Returns
        -------
        Self
            A new config instance populated with values from ``data``.
        """
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # TODO review this
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Deserialise a configuration object from a YAML file.

        Loads the YAML document at ``path`` and delegates to
        :meth:`from_dict`.

        Parameters
        ----------
        path : str
            Filesystem path to a YAML file containing the serialised config.

        Returns
        -------
        Self
            A new config instance populated with values from the YAML file.
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def _env_creator(
        self,
        **env_ctx,
    ) -> BaseEnv:
        """Instantiate the environment using the stored ``env`` class.

        Parameters
        ----------
        **env_ctx : Any
            Keyword arguments forwarded verbatim to the ``env`` constructor
            (e.g. ``world``, ``opt_id``, ``agents``, and any ``env_config``
            key-value pairs).

        Returns
        -------
        BaseEnv
            A freshly constructed environment instance.
        """
        return self.env(**env_ctx)

    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    # TODO move optimizer registration to executor in future
    # TODO enable multiple world registration
    def build_optimizer(
        self,
        *,
        world: Optional[World] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof)."""
        cfg = self.copy(copy_frozen=True)
        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        # TODO remove this in the future and create registry for world and optimizer. keep for now as safety guard
        opt = cfg.opt_class(config=cfg)

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt_id = ray.get(world.register_optimizer.remote(opt))
            opt.set_id(opt_id)

        env = cfg._env_creator(
            world=world, opt_id=opt_id, optimizer=inner_opt, **self.env_config
        )
        opt.env = env

        return opt

    # TODO EnvConfigDict
    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
        train_iters: Optional[int] = None,
        horizon: Optional[int] = None,
        *,
        env_config: Optional[EnvConfigDict] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        disable_env_checking: Optional[bool] = None,
    ):
        """Sets the config's RL-environment settings.

        Args:
            env: The environment specifier. This can either be a tune-registered env,
                via `tune.register_env([name], lambda env_ctx: [env object])`,
                or a string specifier of an RLlib supported type. In the latter case,
                RLlib tries to interpret the specifier as either an Farama-Foundation
                gymnasium env, a PyBullet env, or a fully qualified classpath to an Env
                class, e.g. "ray.rllib.examples.envs.classes.random_env.RandomEnv".
            env_config: Arguments dict passed to the env creator as an EnvContext
                object (which is a dict plus the properties: `num_env_runners`,
                `worker_index`, `vector_index`, and `remote`).
            observation_space: The observation space for the Policies of this Algorithm.
            action_space: The action space for the Policies of this Algorithm.
            horizon: Rollout steps taken by the environment before termination.
            render_env: If True, try to render the environment on the local worker or on
                worker 1 (if num_env_runners > 0). For vectorized envs, this usually
                means that only the first sub-environment is rendered.
                In order for this to work, your env has to implement the
                `render()` method which either:
                a) handles window generation and rendering itself (returning True) or
                b) returns a numpy uint8 image of shape [height x width x 3 (RGB)].
            clip_rewards: Whether to clip rewards during Policy's postprocessing.
                None (default): Clip for Atari only (r=sign(r)).
                True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
                False: Never clip.
                [float value]: Clip at -value and + value.
                Tuple[value1, value2]: Clip at value1 and value2.
            normalize_actions: If True, RLlib learns entirely inside a normalized
                action space (0.0 centered with small stddev; only affecting Box
                components). RLlib unsquashes actions (and clip, just in case) to the
                bounds of the env's action space before sending actions back to the env.
            clip_actions: If True, the RLlib default ModuleToEnv connector clips
                actions according to the env's bounds (before sending them into the
                `env.step()` call).
            disable_env_checking: Disable RLlib's env checks after a gymnasium.Env
                instance has been constructed in an EnvRunner. Note that the checks
                include an `env.reset()` and `env.step()` (with a random action), which
                might tinker with your env's logic and behavior and thus negatively
                influence sample collection- and/or learning behavior.
            is_atari: This config can be used to explicitly specify whether the env is
                an Atari env or not. If not specified, RLlib tries to auto-detect
                this.
            action_mask_key: If observation is a dictionary, expect the value by
                the key `action_mask_key` to contain a valid actions mask (`numpy.int8`
                array of zeros and ones). Defaults to "action_mask".

        Returns:
            This updated AlgorithmConfig object.
        """
        self.env_config: dict = {}
        if env is not None:
            self.env = env
        if train_iters is not None:
            self.env_config.update({"train_iters": train_iters})
        if observation_space is not None:
            self.env_config.update({"observation_space": observation_space})
        if action_space is not None:
            self.env_config.update({"action_space": action_space})
        if horizon is not None:
            self.env_config.update({"horizon": horizon})
        if env_config is not None:
            self.env_config.update(env_config)
        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking

        return self

    def training(self, *, seed: Optional[float] = None) -> Self:
        """Set base training hyperparameters shared by all optimizer types.

        Parameters
        ----------
        seed : float, optional
            Global random seed for reproducible runs.  Forwarded to the
            optimizer's RNG (e.g. ``numpy.random.default_rng(seed)``).

        Returns
        -------
        Self
            This config instance (for method chaining).
        """
        if seed is not None:
            self.seed = seed
        return self

    # TODO Docstring explanation
    # @abstractmethod
    # def ressources(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def evaluation(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def reporting(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def checkpointing(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def fault_tolerance(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def experimental(self):
    #     raise NotImplementedError
