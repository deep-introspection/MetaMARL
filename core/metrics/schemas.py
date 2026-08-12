from pydantic import BaseModel, Field
from core.metrics.enums import ReduceProtocol


class MetricSchema(BaseModel):
    iter: int = Field(default=0, json_schema_extra={"reduce": ReduceProtocol.LAST})
