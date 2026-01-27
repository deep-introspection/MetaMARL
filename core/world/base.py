from __future__ import annotations

from typing import TYPE_CHECKING, KeysView

from core.types import ContextID, OptimizerID
from core.utils import generate_uuid
from core.world.context import Context, ContextSchema

import ray

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

    def __init__(self):
        # Maps optimizer IDs to the set of context IDs they own
        # TODO replace with registry
        self._opt_ctx_map: dict[OptimizerID, set[ContextID]] = {}

        # Maps context IDs to Context objects
        self._contexts: dict[ContextID, Context] = {}

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # Accessors
    def get_opt_registry(self) -> KeysView[OptimizerID]:
        return self._opt_ctx_map.keys()
    
    def get_context(self, ctx_id: ContextID) -> Context | None:
        """Access a context stored in world with an ID"""
        return self._contexts.get(ctx_id, None)

    def get_opt_ctx_ids(self, opt_id: OptimizerID) -> set[ContextID]:
        """
        Return all context IDs registered under a given optimizer.
        """
        return self._opt_ctx_map.get(opt_id, set())

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
            self._opt_ctx_map[opt_id] = set()
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

        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)
            self._opt_ctx_map[ctx.opt_id].add(ctx.id)

        return ctx.id

    def update_context(self, ctx: Context) -> None:
        """
        Update an existing context payload.
        """
        if ctx.id not in self._contexts:
            raise KeyError(f"Context {ctx.id} not registered")

        self._contexts[ctx.id] = ctx

    def remove_context(self, ctx: Context) -> None:
        """
        Remove a context from the world.
        """
        if ctx.id in self._contexts:
            self._contexts.pop(ctx.id)

        if ctx.opt_id in self._opt_ctx_map:
            self._opt_ctx_map[ctx.opt_id].discard(ctx.id)
            if not self._opt_ctx_map[ctx.opt_id]:
                del self._opt_ctx_map[ctx.opt_id]
