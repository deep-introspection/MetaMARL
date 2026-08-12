from typing import TypeAlias

from pydantic import Field

from core.adaptors.ray.schema import AgentEnvStepSchema, EnvStepSchema
from core.metrics.enums import ReduceProtocol

AgentID: TypeAlias = str

class FisheryAgentMetricSchema(AgentEnvStepSchema):
    """Fishery-specific metrics that vary by agent."""

    requested_harvest: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    delivered_harvest: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    requested_frac: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_violation: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_penalty: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    risk_penalty: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

class FisheryMetricSchema(EnvStepSchema):
    """Fishery-specific environment-level metrics."""

    # TODO move this into logging for mechanism
    # max_demand_frac: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )
    quota_stress: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    allowed_harvest: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_stock: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth_noise: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_attempted: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_realized: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    total_usage_norm: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    B_msy: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    MSY: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    F_msy: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    fish_stock_next: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm_next: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # full_required_harvest: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    # realized_harvest: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    # harvest_to_msy: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    by_agent: dict[AgentID, FisheryAgentMetricSchema] = Field(
        default_factory=dict,
    )