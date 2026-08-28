"""Unit tests for the small reporting enumerations."""

from __future__ import annotations

import pytest

from core.reporting.enums import ReporterType

pytestmark = pytest.mark.unit


def test_reporter_type_members():
    assert {r.value for r in ReporterType} == {"wandb", "local"}
    assert ReporterType("wandb") is ReporterType.wandb
