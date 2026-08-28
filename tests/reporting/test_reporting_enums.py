"""Unit tests for the small reporting and logger enumerations."""

from __future__ import annotations

import pytest

from core.loggers.enums import ReduceProtocol
from core.reporting.enums import ReporterType

pytestmark = pytest.mark.unit


def test_reduce_protocol_members_are_strings():
    assert {p.value for p in ReduceProtocol} == {"mean", "sum", "max", "min", "last"}
    assert ReduceProtocol("last") is ReduceProtocol.LAST
    assert isinstance(ReduceProtocol.MEAN, str)


def test_reporter_type_members():
    assert {r.value for r in ReporterType} == {"wandb", "local"}
    assert ReporterType("wandb") is ReporterType.wandb
