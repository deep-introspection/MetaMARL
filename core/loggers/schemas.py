from dataclasses import Field

from pydantic import BaseModel
from loggers.enums import ReduceProtocol


class LoggerSchema(BaseModel):
    """Base Pydantic schema for structured logger payloads.

    Subclass this to define experiment-specific logging schemas.  Each field
    may carry a ``reduce`` hint in ``json_schema_extra`` (a
    :class:`~loggers.enums.ReduceProtocol` value) that downstream reducers use
    to aggregate multiple observations before writing to the logging backend.

    Fields
    ------
    iter : int
        Current iteration counter.  Reduced with
        :attr:`~loggers.enums.ReduceProtocol.LAST` — only the final value seen
        within an aggregation window is kept.  Defaults to ``0``.
    """

    iter: int = Field(default=0, json_schema_extra={"reduce": ReduceProtocol.LAST})
