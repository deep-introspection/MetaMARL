from dataclasses import Field

from pydantic import BaseModel
from core.metrics.enums import ReduceProtocol


class MetricSchema(BaseModel):
    iter: int = Field(default=0, json_schema_extra={"reduce": ReduceProtocol.LAST})
