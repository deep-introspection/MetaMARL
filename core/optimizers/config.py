"""Base configuration objects for optimizers.

``OptimizerConfig`` follows the builder pattern of RLlib's ``AlgorithmConfig``:
a mutable object configured through chained method calls (``environment``,
``debugging``, ``training`` ...), then frozen and turned into an ``Optimizer``
by ``build_optimizer``. Concrete configs (``RayOptimizerConfig``, ``ESConfig``)
extend it with backend-specific builders.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Self, Type, Union

import numpy as np
import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.envs.base import BaseEnv
from core.reporting.wandb import WandbReporter
from core.types import EnvConfigDict, EnvType
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    """Minimal interface shared by all configuration objects."""

    def to_dict(self) -> dict:
        """Converts this configuration to dict format."""
        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
    """Fluent, freezable configuration from which an ``Optimizer`` is built.

    Contract
    --------
    - *Fluent*: builder methods mutate ``self`` and return it, so calls
      chain. Subclasses implement ``training`` (abstract) and may add more.
    - *Freeze*: ``freeze()`` flips ``_is_frozen``; afterwards any attribute
      assignment raises ``AttributeError`` (enforced in ``__setattr__``).
      Freezing is shallow: nested objects such as ``env_config`` stay
      mutable.
    - *Copy*: ``copy(copy_frozen=...)`` deep-copies the config and sets the
      frozen flag of the copy. ``build_optimizer`` always works on
      ``self.copy(copy_frozen=True)``, so the original stays editable and the
      optimizer owns an immutable snapshot.

    Parameters
    ----------
    opt_class : type[Optimizer], optional
        Optimizer class instantiated by ``build_optimizer``.

    Attributes
    ----------
    env : str or EnvType or None
        Environment class (or registered name) set by ``environment``.
    env_config : dict
        Keyword arguments passed to the environment constructor.
    horizon : int or None
        Episode length; also copied into ``env_config["horizon"]``.
    base_seed : int or None
        Root seed given to ``debugging``.
    seeds : list of int
        Training seeds derived from ``base_seed`` (empty when unseeded).
    eval_seeds : list of int or None
        Evaluation seeds; set by subclasses (see
        ``RayOptimizerConfig.evaluation``).
    evaluation_config : OptimizerConfig or None
        Optional nested evaluation config; ``copy(copy_frozen=False)`` also
        unfreezes it.
    stats_cls_lookup : dict
        RLlib ``Stats`` class lookup handed to the optimizer's
        ``MetricsLogger``.
    """

    # TODO registry to allow opt_class str
    # TODO runtime checking of opt_class
    def __init__(self, opt_class: Optional[Type[Optimizer]] = None):
        """Initializes an OptimizerConfig instance.

        Args:
            optimizer_class: An optional Optimizer class that this config class belongs to.
                Used (if provided) to build a respective Optimizer instance from this
                config.
        """
        self.opt_class = opt_class

        # -- lifecycle --
        self._is_frozen = False

        # --- world / environment ---
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.horizon: int = None  # TODO default value

        # --- debugging ---
        self.base_seed: Optional[int] = None
        self.seeds: list[int] = []

        # --- eval ---
        self.evaluation_config: Optional["OptimizerConfig"] = None
        self.eval_seeds: Optional[list[int]] = None

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
        """Shallow-merge ``extra`` into ``env_config`` (``extra`` wins).

        Note that ``environment`` resets ``env_config`` to ``{}`` before
        filling it, so merges done before that call are lost.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
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
        """Serialization from dict"""
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # TODO review this
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Serialization from yaml"""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def _env_creator(
        self,
        **env_ctx,
    ) -> BaseEnv:
        """Instantiate ``self.env`` with the given keyword arguments.

        Parameters
        ----------
        **env_ctx
            Constructor arguments: ``build_optimizer`` passes ``world``,
            ``opt_id``, ``optimizer`` and the ``env_config`` entries;
            ``RayOptimizerConfig`` adds ``agents``, ``mechanism_id``, ``seed``,
            ``policy_seed`` and ``mode``.

        Returns
        -------
        BaseEnv
            A new environment instance. ``self.env`` must be a callable class;
            string specifiers are not resolved here.
        """
        return self.env(**env_ctx)

    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    # TODO move optimizer registration to executor in future
    # TODO enable multiple world registration
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        inner_opt: Optional[Optimizer] = None,
        reporting: Optional[ActorHandle[WandbReporter]] = None,
        **kwargs,
    ) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof)."""
        cfg = self.copy(copy_frozen=True)
        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        # TODO remove this in the future and create registry for world and optimizer. keep for now as safety guard
        opt: Optimizer = cfg.opt_class(config=cfg)

        opt.world = world
        opt.reporting = reporting

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

    @abstractmethod
    def training(self):
        """Set the optimizer's training hyperparameters (backend specific).

        Subclasses define the accepted keyword arguments and return ``self``.
        """
        raise NotImplementedError

    def debugging(
        self,
        *,
        seed: Optional[int] = None,  # base seed
        num_seeds: int = 3,
    ) -> Self:
        """Derive the training seeds from a base seed.

        Parameters
        ----------
        seed : int or None, optional
            Base seed stored in ``base_seed``. A
            ``numpy.random.SeedSequence(seed)`` is created and
            ``generate_state(num_seeds)`` yields ``num_seeds`` independent
            32-bit integers, stored as Python ints in ``seeds``. The same base
            seed always yields the same list, and different base seeds give
            well-separated streams (the SeedSequence hashing spreads them).
            ``None`` clears ``seeds`` to ``[]``.
        num_seeds : int, optional
            Number of training seeds to derive (default 3).

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """
        if seed is not None:
            self.base_seed = seed
            ss = np.random.SeedSequence(seed)
            self.seeds = ss.generate_state(num_seeds).tolist()
        else:
            self.seeds = []
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
