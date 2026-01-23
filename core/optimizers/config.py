from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING, Optional, Self, Type, Union

from gymnasium import Space

from core.types import EnvType
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
        self._world: Optional[World] = None
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.observation_space: Optional[Space] = None
        self.action_space: Optional[Space] = None
        self.disable_env_checking: bool = False

        # --- training ---
        self.seed: int = 0

        # --- eval ---
        self.evaluation_config: Optional["OptimizerConfig"] = None

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

    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    # TODO move optimizer registration to executor in future
    # TODO enable multiple world registration
    def build_optimizer(self) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof).

        Args:
            world: Name of the world (gymnasium Env type) #review
            logger_creator: Callable that creates a logger object. If unspecified, a default logger is created.

        """
        # TODO Executer to enforce guardrails
        # if world is None:
        #     raise ValueError("Optimizer requires a World instance")

        cfg = self.copy()

        # TODO : world is passed to optimizer, but also stored in config. must only have one source of truth
        # if world is not None:
        #     cfg._world = world

        cfg.freeze()  # attention this would freeze the cfg even if the world is None !
        opt_class = self.opt_class
        return opt_class(config=cfg)

    # TODO Docstring explanation
    # @abstractmethod
    # def world(self, world: Optional[World] = None):
    #     """
    #     Docstring for world

    #     :param self: Description
    #     :param world: Description
    #     """
    #     raise NotImplementedError

    # TODO EnvConfigDict
    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
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
        if env_config is not None:
            self.env_config = env_config
        if observation_space is not None:
            self.observation_space = observation_space
        if action_space is not None:
            self.action_space = action_space
        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking
        return self

    def training(self, *, seed=Optional[float]) -> Self:
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
