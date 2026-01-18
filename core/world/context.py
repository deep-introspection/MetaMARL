from dataclasses import dataclass
from core.utils.types import ContextID, OptimizerID
from pydantic import BaseModel


# TODO some world contexts are singletons (mutable) others are simply mutable.
# TODO for now singleton/or no is deffered to world
# TODO Enums for Context to access different Context Schemas.
class ContextSchema(BaseModel):
    """Base schema for shared world context."""

    pass


@dataclass
class Context:
    """
    Runtime instance of a context
    """

    id: ContextID | None
    opt_id: OptimizerID
    schema: type[ContextSchema]
    payload: ContextSchema
