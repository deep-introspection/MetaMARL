from abc import ABC
from typing import Union
from core.types import ContextID, OptimizerID
from core.world.context import Context, ContextSchema
from core.utils import generate_uuid


class World(ABC):
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

    # Accessors
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
            if type(ctx.schema) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in world."
                )

    # Mutators
    def set_new_optimizer(self, opt_id: Union[OptimizerID | None]) -> None:
        """
        Register a new optimizer ID in the world.

        This initializes an empty context set for the optimizer.
        """
        if opt_id is None:
            opt_id = generate_uuid(registry=self._opt_ctx_map.keys())
        self._opt_ctx_map[opt_id] = set()

    def set_new_context(self, ctx: Context, singleton: bool = False) -> ContextID:
        """
        Register a new context in the world.

        Args:
            ctx: Context object containing optimizer ID and schema payload
            singleton: Whether this context schema must be unique globally

        Returns:
            The generated ContextID
        """
        if ctx.id is not None:
            raise ValueError("New context must not alreay have an id")

        if singleton:
            self._validate_ctx_schema_exists(ctx.schema)

        if ctx.opt_id not in self._opt_ctx_map.keys():
            self.set_new_optimizer(ctx.opt_id)

        ctx.id = generate_uuid(registry=self._contexts.keys())

        self._contexts[ctx.id] = ctx
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
        self._contexts.pop(ctx.id)
        self._opt_ctx_map[ctx.opt_id].remove(ctx.id)

        if not self._opt_ctx_map[ctx.opt_id]:
            del self._opt_ctx_map[ctx.opt_id]
