from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING, Optional, Self, Type, Union

from gymnasium import Space
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.envs.base import BaseEnv
from core.types import EnvConfigDict, EnvType
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    def to_dict(self) -> dict:
        """Converts this configuration to dict format."""
        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
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
        self.train_iters: Optional[int] = None
        self.eval_iters: Optional[int] = None
        self.env_config: dict = {}
        self.observation_space: Optional[Space] = None
        self.action_space: Optional[Space] = None
        self.disable_env_checking: bool = False

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
        *,
        world: Optional[World] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ) -> BaseEnv:
        return self.env(
            world=world,
            optimizer=inner_opt,
            train_iters=self.train_iters,
            eval_iters=self.eval_iters,
            **self.env_config,
        )

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
            opt_id = world.register_optimizer(opt)
            opt.set_id(opt_id)

        env = cfg._env_creator(world=world, inner_opt=inner_opt)
        env.set_opt_id(opt.id)
        opt.env = env

        return opt

    # TODO EnvConfigDict
    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
        train_iters: Optional[int] = None,
        eval_iters: Optional[int] = None,
        *,
        env_config: Optional[EnvConfigDict] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        disable_env_checking: Optional[bool] = None,
    ):
        """Defines the environment interface for the Optimizer

        Args:
            env: Environment identifier or callable. May be a Gymnasium env, a Ray-registered env
                name, or a custom environment class.
            env_config: Domain-specific configuration passed to the environment constructor.
            observation_space: Observation space describing environment outputs. Optional for
                for optimizers that do not consume observation.
            action_space: Action space describing valid environment inputs.
            disable_env_checking: If True, disable environment validation checks. Userful for
                custom or partially compliant environments.
        """
        if env is not None:
            self.env = env
        if train_iters is not None:
            self.train_iters = train_iters
        if eval_iters is not None:
            self.eval_iters = eval_iters
        if env_config is not None:
            self.env_config = env_config
        if observation_space is not None:
            self.observation_space = observation_space
        if action_space is not None:
            self.action_space = action_space
        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking
        return self

    def training(self, *, seed: Optional[float] = None) -> Self:
        """

        Args:
            seed:
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
