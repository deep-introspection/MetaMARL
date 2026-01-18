from abc import ABC
from enum import Enum, auto
from pydantic import BaseModel
from utils import ContextID, OptimizerID
import uuid

# class Context(Enum):
#     # what do we want to enum exactly ? parent of child ? types of contexts ±?
#     STATE = auto()
#     ACTION = auto()


# TODO this should be a base class with the ability to generate its own uuid
class Context(BaseModel):
    ctx_id: ContextID
    opt_id: OptimizerID


# why not inherit from _Generic ?
# TODO context wrapper for envs will be required
# TODO Enums for different contexts


class World(ABC):
    def __init__(self):
        self._contexts: dict[OptimizerID, dict[ContextID, Context]] = None

    def _generate_uuid(self) -> uuid.UUID:
        return uuid.getnode()

    def add_context(self, opt_id: OptimizerID, ctx: Context) -> None:
        ctx_id = self._generate_uuid()
        self._contexts[opt_id][ctx_id] = ctx

    def update_context(self, context: Context) -> None:
        pass

    def remove_context(self, ContextID) -> None:
        pass
