"""Metric schema of one Evolution Strategies generation.

Series fields (``generation``, ``sigma``, ``fitness_mean``, ...) grow by one
value per generation; ``by_mechanism`` holds each candidate's fitness and
parameter values, ``search_mean``/``global_best``/``generation_best`` are keyed
by parameter name, and ``inner`` is the reduced metric schema of the inner
optimizer, specialized at runtime (``RaySchema`` for RLlib).
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

MechanismID: TypeAlias = str
ParameterName: TypeAlias = str


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
    generation: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
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

    search_mean: dict[ParameterName, ESParameterSchema] = Field(default_factory=dict)
    global_best: dict[ParameterName, ESParameterSchema] = Field(default_factory=dict)
    generation_best: dict[ParameterName, ESParameterSchema] = Field(
        default_factory=dict
    )
    inner: Optional[MetricSchema] = None
