from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Optional

# TODO move ray dependencies out of ray optimizer
from ray.actor import ActorHandle
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from core.envs.base import BaseEnv
from core.metrics.schemas import MetricSchema
from core.optimizers.config import OptimizerConfig
from core.reporting.base import Reporter
from core.types import OptimizerID
from core.world.base import World


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
        world: Optional[ActorHandle[World]] = None,
        reporting: Optional[Reporter] = None,
        # TODO
        # logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        from core.optimizers.config import OptimizerConfig

        self.config: OptimizerConfig = config

        self.world = world  # TODO replace by envFactory
        self.reporting: Optional[Reporter] = reporting

        # Assigned by World or Orchestrator
        self.opt_id: OptimizerID | None = None

        # Optional environment (may be None for meta-optimizers)
        # TODO review
        self._env: BaseEnv | None = config.env

        # Optimizer Graph connectivity
        self._downstream: set["Optimizer"] = set()
        self._upstream: set["Optimizer"] = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.opt_id})"

    # TODO setup accessors and mutators
    # @property
    # def world(self) -> ActorHandle[World]:
    #     return self._world

    # @world.setter
    # def world(self, world: ActorHandle[World]) -> None:
    #     self._world = world

    # @property
    # def reporting(self) -> ActorHandle[WandbReporter]:
    #     return self._reporting

    # @reporting.setter
    # def world(self, reporting: ActorHandle[WandbReporter]) -> None:
    #     self._reporting = reporting

    @property
    def env(self) -> BaseEnv | None:
        return self._env

    @env.setter
    def env(self, value: BaseEnv | None) -> None:
        self._env = value

        if value is not None:
            self._on_env_init(value)

    def _on_env_init(self, env: BaseEnv) -> None:
        """Hook called after a runtime env is attached"""
        pass

    @property
    def id(self) -> OptimizerID:
        if self.opt_id is None:
            raise RuntimeError("Optimizer ID not set")
        return self.opt_id

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    # TODO make id immutable
    def set_id(self, id: OptimizerID) -> None:
        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID already set")
        self.opt_id = id

    def set_downstream(self, opt: "Optimizer") -> None:
        self._downstream.add(opt)

    def set_upstream(self, opt: "Optimizer") -> None:
        # this is only for checking!
        # and also to call the upstream optimizer
        self._upstream.add(opt)

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "Optimizer":
        """Instantiate optimizer from config."""
        return cls(config=config)

    # TODO default config logic
    @classmethod
    def get_default_config(cls) -> OptimizerConfig:
        """ """
        raise NotImplementedError("Optimizers must define a default config explicitly")

    @classmethod
    def from_checkpoint(cls, file_path: Any) -> "Optimizer":
        """Restore optimizer from config."""
        raise NotImplementedError

    def reduce_metrics(self) -> MetricSchema:
        """
        Destructively reduce and return all accumulated optimizer metrics.
        """
        if self.logger is None:
            raise RuntimeError(f"{type(self).__name__} has no MetricLogger.")

        return self.logger.reduce()

    def flush_metrics(self) -> None:
        """
        Flush all accumulated optimizer metrics.
        """
        if self.logger is None:
            return

        self.logger.reset()

    def report_metrics(self) -> None:
        """
        plot all accumulated metrics given queries
        """
        metrics = self.logger.peek()
        self.reporting.report(metrics)

    # Accessors
    # def __getattribute__(self, name):
    #     return super().__getattribute__(name)

    # replace with get context but probably this is in env
    # def get_signal(self) -> Signal:
    #     return self.signal

    # Mutators
    # def __setattr__(self, name, value) -> Any:
    #     return super().__setattr__(name, value)

    # TODO change this to training step
    @abstractmethod
    def run(self) -> None:
        """
        Implementations may publish Context objects to the World, invoke downstream
        optimizers via `self._downstream`, and retrieve or aggregate contexts from
        the World in any order. The framework does not impose a fixed execution flow.

        Example
        -------
        >>> def run(self, world: World) -> None:
        >>>     # Publish contexts
        >>>     ctx = Context(
        >>>         id=None,
        >>>         opt_id=self.id,
        >>>         payload=SignalContext(value=1.0),
        >>>     )
        >>>     world.set_new_context(ctx)
        >>>
        >>>     # Execute downstream optimizers
        >>>     for opt in self._downstream:
        >>>         opt.run(world)
        >>>
        >>>     # Retrieve downstream contexts ---
        >>>     for opt in self._downstream:
        >>>         ctx_ids = world.get_opt_ctx_ids(opt.id)
        >>>         for ctx_id in ctx_ids:
        >>>             ctx = world.get_context(ctx_id)
        >>>             # aggregate or process ctx.payload here
        >>>
        >>>     # Optional: update or overwrite own context ---
        >>>     ctx.payload.value += 1.0
        >>>     world.update_context(ctx)

        """
        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate Optimizer Performance"""
        pass

    def save(self) -> None:
        """Persist Optimizer State"""
        pass

    def reset(self) -> None:
        """Reset optimizer state (e.g., policy weights)."""
        pass

    def stop(self) -> None:
        pass
