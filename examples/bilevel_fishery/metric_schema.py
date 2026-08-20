from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol

AgentID: TypeAlias = str

class FisheryAgentMetricSchema(AgentEnvStepSchema):
    """Fishery-specific metrics that vary by agent."""

    requested_harvest: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    delivered_harvest: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    requested_frac: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_violation: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_penalty: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    risk_penalty: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

class FisheryMetricSchema(EpisodeRolloutSchema):
    """Fishery-specific environment-level metrics."""

    # TODO move this into logging for mechanism
    # max_demand_frac: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )
    quota_stress: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    allowed_harvest: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_stock: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth_noise: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_attempted: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_realized: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    total_usage_norm: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    B_msy: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    MSY: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    F_msy: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    fish_stock_next: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm_next: Optional[float] = Field(
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