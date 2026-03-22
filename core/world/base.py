from __future__ import annotations

from typing import TYPE_CHECKING, KeysView, Optional

import ray
from core.reporting.wandb import WandbReporter
from core.types import ContextID, OptimizerID
from core.utils import generate_uuid
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


@ray.remote
class World:
    """
    Shared runtime container for optimizer-produced contexts.

    The World acts as the single source of truth for:
    - Context lifecycles (creation, update, removal)
    - Association between optimizers and their contexts
    - Enforcing global constraints (e.g. singleton contexts)

    The World does NOT own optimizers or environments.
    It only tracks identifiers and context payloads.
    """

    # TODO the reporting type annotation to add
    def __init__(self, reporting: WandbReporter = None):
        # Maps optimizer IDs to the list of context IDs they own
        # TODO replace with registry
        self._opt_ctx_map: dict[OptimizerID, list[ContextID]] = {}

        # Maps context IDs to Context objects
        self._contexts: dict[ContextID, Context] = {}

        # Mechanism registry
        self._mechanism_registry: dict[int, MechanismContext] = {}

        # TODO wrap this into generic logger class - for now use W&B
        self.reporting: WandbReporter = reporting

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # Accessors
    def get_ctx_registry(self) -> dict[ContextID, Context]:
        return self._contexts

    def get_mechanism_registry(self) -> dict[int, MechanismContext]:
        return self._mechanism_registry

    def get_opt_registry(self) -> KeysView[OptimizerID]:
        return self._opt_ctx_map.keys()

    def get_context(self, ctx_id: ContextID) -> Context | None:
        """Access a context stored in world with an ID"""
        return self._contexts.get(ctx_id, None)

    def get_opt_ctx_ids(self, opt_id: OptimizerID) -> list[ContextID]:
        """
        Return all context IDs registered under a given optimizer.
        """
        return list(self._opt_ctx_map.get(opt_id, []))

    def get_ctx_ids(self) -> set[ContextID]:
        """
        Return all context IDs registered in the world.
        """
        return set(self._contexts.keys())

    def get_opt_ids(self) -> set[OptimizerID]:
        """
        Return all optimizer IDs known to the world.
        """
        return set(self._opt_ctx_map.keys())

    def get_mechanism(self) -> MechanismContext:
        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned
                return m_ctx

        raise RuntimeError("no available mechanisms to train")

    def try_get_mechanism(self) -> MechanismContext | None:
        """Try to get a published mechanism, return None if none available."""
        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned
                return m_ctx
        return None

    # TODO fix this function. now the primary key is ctx_id
    def get_mechanism_by_index(self, index: int) -> MechanismContext:
        return self._mechanism_registry[index]

    def _validate_ctx_schema_exists(self, schema: type[ContextSchema]) -> None:
        """
        Ensure a singleton ContextSchema is not already present in the world.
        """
        for ctx in self._contexts.values():
            if type(ctx.payload) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in world."
                )

    # Mutators
    def append_context(self, ctx: Context, *, singleton: bool = False):
        # Enforce singleton schemas if requested
        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None and ctx.id in self._contexts:
            raise ValueError(f"ContextID '{ctx.id}' already exists")

        ctx.id = generate_uuid(self._contexts)
        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            self._mechanism_registry[ctx.id] = ctx.payload

        # Track optimizer → context mapping
        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)
            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        # if env-step-context call reporter actor
        if self.reporting is not None and isinstance(ctx.payload, EnvStepContext):
            self.reporting.plot_env_step.remote(
                ctx=ctx,
                obs_keys_skip=(
                    "fixed_quota",
                    "prop_quota",
                    "min_stock",
                    "fine_amount",
                    "ban_period",
                    "catch_prob",
                ),
            )

        return ctx.id

    def register_optimizer(self, opt: Optimizer) -> OptimizerID:
        """
        Register a new optimizer ID in the world.

        This initializes an empty context set for the optimizer.
        """
        return self._set_new_opt_id(opt_id=opt.opt_id)

    def _set_new_opt_id(self, opt_id: OptimizerID) -> OptimizerID:
        if opt_id is None:
            opt_id = generate_uuid(registry=self._opt_ctx_map.keys())

        if opt_id not in self._opt_ctx_map:
            self._opt_ctx_map[opt_id] = []

        return opt_id

    def set_new_context(self, ctx: Context, singleton: bool = False) -> ContextID:
        """
        Register a new context in the world.

        Args:
            ctx: Context object containing optimizer ID and schema payload
            singleton: Whether this context schema must be unique globally

        Returns:
            The generated ContextID
        """

        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None:
            if ctx.id in self._contexts:
                raise ValueError(f"ContextID '{ctx.id}' already exists")
        else:
            ctx.id = generate_uuid(registry=self._contexts.keys())

        self._contexts[ctx.id] = ctx

        # Track latest mechanism globally
        # Track per-env mechanism
        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")
            self._mechanism_registry[ctx.id] = ctx.payload

        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)

            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        return ctx.id

    def update_context(self, ctx: Context) -> None:
        """
        Update an existing context payload.
        """
        if ctx.id not in self._contexts:
            raise KeyError(f"Context {ctx.id} not registered")

        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")
            self._mechanism_registry[ctx.id] = ctx.payload

    def remove_context(self, ctx: Context) -> None:
        """
        Remove a context from the world.
        """
        if ctx.id in self._contexts:
            self._contexts.pop(ctx.id)

        if ctx.opt_id in self._opt_ctx_map:
            lst = self._opt_ctx_map[ctx.opt_id]
            if ctx.id in lst:
                lst.remove(ctx.id)

            if not lst:
                del self._opt_ctx_map[ctx.opt_id]

    # TODO fix this function. now the primary key is ctx_id
    def flush(self, job: Optional[MechanismStatus] = None) -> None:
        to_delete = []

        for ctx_id, m_ctx in self._mechanism_registry.items():
            if job is not None and m_ctx.job != job:
                continue
            to_delete.append(ctx_id)

        for ctx_id in to_delete:
            del self._mechanism_registry[ctx_id]

    def flush_ctx(self, ctx_ids: list[ContextID]):
        for cid in ctx_ids:
            self._contexts.pop(cid, None)
