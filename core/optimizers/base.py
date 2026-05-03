from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Optional

from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from core.envs.base import BaseEnv
from core.optimizers.config import OptimizerConfig
from core.types import OptimizerID


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
        """Initialize the base optimizer node.

        Sets up shared state that every concrete optimizer requires: a reference
        to its config, an unset optimizer ID (assigned later by the World or an
        orchestrator), an optional environment handle, a metrics logger, and the
        upstream / downstream adjacency sets used for multi-level graph wiring.

        Parameters
        ----------
        config : OptimizerConfig, optional
            Configuration object carrying hyperparameters, environment settings,
            and lifecycle flags.  If ``None``, the optimizer is constructed
            without a config (useful in testing or when overriding).
        **kwargs : Any
            Additional keyword arguments forwarded to subclass constructors.
        """
        from core.optimizers.config import OptimizerConfig

        self.config: OptimizerConfig = config

        # Assigned by World or Orchestrator
        self.opt_id: OptimizerID | None = None

        # Optional environment (may be None for meta-optimizers)
        # TODO review
        self.env: BaseEnv | None = config.env

        # Optional metrics hook
        self.metrics: MetricsLogger = MetricsLogger(
            root=True, stats_cls_lookup=config.stats_cls_lookup
        )

        # Optimizer Graph connectivity
        self._downstream: set["Optimizer"] = set()
        self._upstream: set["Optimizer"] = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.opt_id})"

    @property
    def id(self) -> OptimizerID:
        """Unique identifier of this optimizer node.

        Returns
        -------
        OptimizerID
            The identifier assigned by the World or orchestrator.

        Raises
        ------
        RuntimeError
            If the ID has not yet been set via :meth:`set_id`.
        """
        if self.opt_id is None:
            raise RuntimeError("Optimizer ID not set")
        return self.opt_id

    @property
    def batch_capacity(self) -> int:
        """Maximum number of candidates or rollout episodes the optimizer
        can process in a single step.

        Returns
        -------
        int
            The batch capacity configured for this optimizer.
        """
        return self._batch_capacity

    # TODO make id immutable
    def set_id(self, id: OptimizerID) -> None:
        """Assign a unique identifier to this optimizer node.

        Parameters
        ----------
        id : OptimizerID
            The identifier to assign.

        Raises
        ------
        RuntimeError
            If the optimizer already has an ID (IDs are immutable once set).
        """
        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID already set")
        self.opt_id = id

    def set_downstream(self, opt: "Optimizer") -> None:
        """Register a downstream optimizer that this node can invoke.

        Parameters
        ----------
        opt : Optimizer
            The optimizer node placed downstream in the optimization graph.
        """
        self._downstream.add(opt)

    def set_upstream(self, opt: "Optimizer") -> None:
        """Register an upstream optimizer that this node is subordinate to.

        Used for graph traversal and cycle checking; the upstream node is
        responsible for invoking the current optimizer's :meth:`run` method.

        Parameters
        ----------
        opt : Optimizer
            The optimizer node placed upstream in the optimization graph.
        """
        # this is only for checking!
        # and also to call the upstream optimizer
        self._upstream.add(opt)

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "Optimizer":
        """Instantiate an optimizer from a configuration object.

        Parameters
        ----------
        config : OptimizerConfig
            Fully specified configuration for the optimizer.

        Returns
        -------
        Optimizer
            A new optimizer instance of type ``cls``.
        """
        return cls(config=config)

    # TODO default config logic
    @classmethod
    def get_default_config(cls) -> OptimizerConfig:
        """Return the default configuration for this optimizer class.

        Returns
        -------
        OptimizerConfig
            A default-populated configuration object for this optimizer.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method to provide a concrete default.
        """
        raise NotImplementedError("Optimizers must define a default config explicitly")

    @classmethod
    def from_checkpoint(cls, file_path: Any) -> "Optimizer":
        """Restore an optimizer from a persisted checkpoint.

        Parameters
        ----------
        file_path : Any
            Path or identifier pointing to the checkpoint artifact.

        Returns
        -------
        Optimizer
            A restored optimizer instance.

        Raises
        ------
        NotImplementedError
            Subclasses must implement checkpoint restoration logic.
        """
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
        """Evaluate optimizer performance on a held-out or evaluation split.

        Subclasses should override this method to run evaluation rollouts,
        compute performance metrics, and log results.  The base implementation
        is a no-op.
        """
        pass

    def save(self) -> None:
        """Persist the optimizer state to disk or a remote checkpoint store.

        Subclasses should override this method to serialize policy weights,
        distribution parameters, and any other runtime state required for
        resumption.  The base implementation is a no-op.
        """
        pass

    def reset(self) -> None:
        """Reset optimizer state (e.g., policy weights, search distribution).

        Subclasses should override this method to reinitialise all mutable
        runtime state to its starting values.  The base implementation is a
        no-op.
        """
        pass

    def stop(self) -> None:
        """Perform cleanup when the optimizer is shut down.

        Subclasses should override this method to release resources such as
        Ray actors, file handles, or network connections.  The base
        implementation is a no-op.
        """
        pass
