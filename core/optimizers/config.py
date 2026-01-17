import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

from core.optimizers.base import Optimizer
from core.utils import WorldType


class _Config(ABC):
    def to_dict(self) -> dict:
        """Converts this configuration to dict format."""


# TODO this has to be immutable
# TODO optimizer registry


class OptimizerConfig(_Config):
    def __init__(self, opt_class: Optional[type] = None):
        """Initializes an OptimizerConfig instance.

        Args:
            optimizer_class: An optional Optimizer class that this config class belongs to.
                Used (if provided) to build a respective Optimizer instance from this
                config.
        """
        self.opt_class = opt_class

        # World specs
        self._world = None

    def build_optimizer(
        self,
        world: Optional[Union[str, WorldType]] = None,
        # logger_creator: Optional[Callable[[], Logger]] = None,
        use_copy: bool = True,
    ) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof).

        Args:
            world: Name of the world (gym.Env type) #review
            logger_creator: Callable that creates a logger object. If unspecified, a default logger is created.

        """
        if world is not None:
            self.world = world

        # if logger_creator is not None:
        #         self.logger_creator = logger_creator

        if isinstance(self.opt_class, str):
            opt_class = get_trainable_cls(self.opt_class)

        return opt_class(
            config=self if not use_copy else copy.deepcopy(self),
            # logger_creator=self.logger_creator,
        )

    # TODO Docstring explanation
    @abstractmethod
    def world(self, world=Optional[Union[str, WorldType]]):
        """
        Docstring for world

        :param self: Description
        :param world: Description
        """
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def training(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def ressources(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def evaluation(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def reporting(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def checkpointing(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def fault_tolerance(self):
        raise NotImplementedError

    # TODO Docstring explanation
    @abstractmethod
    def experimental(self):
        raise NotImplementedError
