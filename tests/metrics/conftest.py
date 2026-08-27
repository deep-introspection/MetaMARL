"""Schemas shared by the metrics and reporting tests."""

from typing import Optional

import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema


class LeafSchema(MetricSchema):
    mean_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    series_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    last_value: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    sum_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    )
    min_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    max_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
    default_value: Optional[float] = None  # no metadata -> MEAN


class RichLeafSchema(LeafSchema):
    """Runtime subtype of ``LeafSchema`` with one extra field."""

    extra: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class GroupSchema(MetricSchema):
    by_id: dict[str, LeafSchema] = Field(default_factory=dict)


class RootSchema(MetricSchema):
    static: LeafSchema
    group: GroupSchema
    optional_static: Optional[LeafSchema] = None
    inner: Optional[MetricSchema] = None


@pytest.fixture
def schemas():
    return LeafSchema, RichLeafSchema, GroupSchema, RootSchema
