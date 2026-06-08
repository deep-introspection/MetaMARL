from dataclasses import Field

from pydantic import BaseModel
from core.reporting.loggers.enums import ReduceProtocol


class LoggerSchema(BaseModel):
    iter: int = Field(default=0, json_schema_extra={"reduce": ReduceProtocol.LAST})
