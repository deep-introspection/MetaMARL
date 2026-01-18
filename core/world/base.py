from abc import ABC

from core.utils.types import ContextID, OptimizerID
from core.world.context import Context, ContextSchema

import uuid

# class Context(Enum):
#     # what do we want to enum exactly ? parent of child ? types of contexts ±?
#     STATE = auto()
#     ACTION = auto()

# TODO accesor for context
# TODO this should be a base class with the ability to generate its own uuid
# TODO must be immutable
# TODO Enums for Context to access different Context Schemas.


# why not inherit from _Generic ?
# TODO context wrapper for envs will be required
# TODO Enums for different contexts
# TODO contexts should be immutable
# TODO a schema stores only the att, but how does the payload get stored and passed ?


class World(ABC):
    def __init__(self):
        # TODO replace with registry
        self._opt_ctx_map: dict[OptimizerID, set[ContextID]] = {}
        self._contexts: dict[ContextID, Context] = {}

    def get_opt_ctx_ids(self, opt_id) -> set[ContextID]:
        return self._opt_ctx_map[opt_id]

    def get_ctx_ids(self) -> set[ContextID]:
        return set(self._contexts.keys())

    def get_opt_ids(self) -> set[OptimizerID]:
        return set(self._opt_ctx_map.keys())

    # TODO method must be able to validate unique uuid against registry
    # TODO extend this function for the genreation of any UUID
    def _generate_uuid(self) -> ContextID:
        """
        Generates a unique uuid key for the context
        """
        while True:
            ctx_id = str(uuid.uuid4())
            if ctx_id not in self._contexts.keys():
                return ctx_id

    def _validate_new_context(self, ctx: Context) -> None:
        """
        Valides if the context has a uuid
        """
        if ctx.id is not None:
            raise ValueError("New context must not alreay have an id")

    def _validate_ctx_schema_exists(self, schema: type[ContextSchema]) -> None:
        for ctx in self._contexts.values():
            if type(ctx.schema) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in world."
                )

    def set_new_context(self, ctx: Context, singleton: bool = False) -> uuid.UUID:
        """
        Register a new context in the world
        """

        self._validate_new_context(ctx)

        if singleton:
            self._validate_ctx_schema_exists(ctx.schema)

        ctx.id = self._generate_uuid()

        self._contexts[ctx.id] = ctx
        self._opt_ctx_map.setdefault(ctx.opt_id, set()).add(ctx.id)

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
