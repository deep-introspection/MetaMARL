from abc import ABC, abstractmethod
from logging import Logger
from typing import Callable, Optional, Any

from core.optimizers.config import OptimizerConfig
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from core.types import OptimizerID
from core.world.base import World
from core.optimizers.base import Optimizer


class Optimizer(ABC):
    """
    Abstract optimizer node.

    Represents a single logical optimizer in a possibly hierarchical
    (bilevel / multilevel) optimization graph.
    """

    # data owned by the optimizer
    config: OptimizerConfig

    opt_id: OptimizerID

    metrics: Optional[MetricsLogger]

    # TODO ability to save data offline
    # offline_data: Optional[OfflineData]

    logger_creator: Optional[Callable[[], Logger]]

    # this is the default configuration as soon as an optimizer is created

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        # TODO
        # logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        self.config: OptimizerConfig = config

        # Assigned by World or Orchestrator
        self.opt_id: OptimizerID | None = None

        # Optional environment (may be None for meta-optimizers)
        # TODO review
        self.env = getattr(config, "_env", None)

        # Optional metrics hook
        self.metrics = None

        # Optimizer Graph connectivity
        self._downstream: set[Optimizer] = set()
        self._upstream: set[Optimizer] = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.opt_id})"

    @property
    def id(self) -> OptimizerID:
        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID not set")
        return self.opt_id

    # TODO make id immutable
    # def set_id(self, id: OptimizerID) -> None:
    #     if self.opt_id is not None:
    #         raise RuntimeError("Optimizer ID not set")
    #     self.opt_id = id

    def set_downstream(self, opt: Optimizer) -> None:
        self._downstream.add(opt)

    def set_upstream(self, opt: Optimizer) -> None:
        # this is only for checking!
        # and also to call the upstream optimizer
        self._upstream.add(opt)

    # what is a class method decorator doing really
    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "Optimizer":
        """Instantiate optimizer from config."""
        return cls(config=config)

    # TODO default config logic
    @classmethod
    def get_default_config(cls) -> OptimizerConfig:
        """ """
        raise NotImplementedError(
            "Optimizers must define a default config explicitly"
        )
    @classmethod
    def from_checkpoint(cls, file_path: Any) -> "Optimizer":
        """Restore optimizer from config."""
        raise NotImplementedError

    # Accessors
    # def __getattribute__(self, name):
    #     return super().__getattribute__(name)

    # replace with get context but probably this is in env
    # def get_signal(self) -> Signal:
    #     return self.signal

    # Mutators
    # def __setattr__(self, name, value) -> Any:
    #     return super().__setattr__(name, value)

    def run(self, world: Optional[World] = None) -> None:
        # TODO do we have the entire loop in run ?
        """Execute ONE optimization iteration."""

        if world is None:
            raise ValueError("Optimizer.run requires a World instance")
        
        if self.opt_id is None:
            raise RuntimeError("Optimizer must have an ID before running")

        self._publish(world)

        for opt in self._downstream:
            opt.run(world)

        self._aggregate(world)

    @abstractmethod
    def _publish(self, world: Optional[World] = None) -> None:
        """Publish context to world before downstream optimizers run"""
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, world: Optional[World] = None) -> None:
        """Aggregate and process context published by downstream optimizers"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, world: Optional[World]) -> None:
        """Evaluate Optimizer Performance"""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self) -> None:
        """Persist Optimizer State"""
        raise NotImplementedError
