from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

MechanismID: TypeAlias = str


class ESParameterSchema(MetricSchema):
    value: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )


class ESCandidateSchema(MetricSchema):
    fitness: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )

    # Parameter names are runtime-defined by MechanismSpace.
    by_parameter: dict[str, ESParameterSchema] = Field(default_factory=dict)


class ESSchema(MetricSchema):
    sigma: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    population_size: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    fitness_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    fitness_best: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    best_mechanism_idx: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    best_fitness_global: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )

    by_mechanism: dict[MechanismID, ESCandidateSchema] = Field(default_factory=dict)

    search_mean: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    global_best: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    inner: Optional[MetricSchema] = None
