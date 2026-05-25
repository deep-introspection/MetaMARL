"""Sanity tests for the package skeleton."""

import logging

import pytest

import bilevel_fishery
from bilevel_fishery._logging import setup_logging


@pytest.mark.unit
def test_package_imports_and_exposes_version() -> None:
    assert bilevel_fishery.__version__ == "0.1.0"


@pytest.mark.unit
def test_setup_logging_is_idempotent() -> None:
    setup_logging("DEBUG")
    setup_logging("INFO")
    assert logging.getLogger().level == logging.INFO
