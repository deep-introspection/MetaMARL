import copy
from abc import ABC, abstractmethod
from typing import Optional, Union, Type, Self

from core.optimizers.base import Optimizer
from core.utils import WorldType


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

        # World specs
        self._world = None

        # Has this config object been frozen (cannot alter its attributes anymore).
        self._is_frozen = False

        # Evaluation
        self.evaluation_config : Optional["OptimizerConfig"] = None

    def __setattr__(self, name, value):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if name not in ["_is_frozen"]:
                raise AttributeError(
                    f"Cannot set attribute ({name}) of an already frozen "
                    "OptimizerConfig!"
                )
        super().__setattr__(name, value)

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
    
    # TODO freezing for nested configs
    def freeze(self) -> None:
        """Freeze this config object, such that no attributes can be set anymore.

        Optimizers should use this method to make sure their config objects 
        remain read-only after this.
        """
        if self._is_frozen:
            return
        self._is_frozen = True


    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    def build_optimizer(
        self,
        world: Optional[Union[str, WorldType]] = None,
    ) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof).

        Args:
            world: Name of the world (gym.Env type) #review
            logger_creator: Callable that creates a logger object. If unspecified, a default logger is created.

        """
        cfg = self.copy() 
        if world is not None:
            cfg._world = world
        cfg.freeze() # attention this would freeze the cfg even if the world is None !

        opt_class = self.opt_class
        
        return opt_class(config=cfg)

    # TODO Docstring explanation
    @abstractmethod
    def world(self, world: Optional[Union[str, WorldType]]):
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
