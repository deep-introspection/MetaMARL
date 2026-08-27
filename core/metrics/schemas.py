"""Root of the metric schema hierarchy.

Every logged structure derives from :class:`MetricSchema`; it carries the
``iter`` counter (last value kept) shared by all loggers.
"""

from typing import Optional

from pydantic import BaseModel, Field

from core.metrics.enums import ReduceProtocol


class MetricSchema(BaseModel):
    iter: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
