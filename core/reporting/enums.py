"""Enumerations of the reporting layer."""

from enum import Enum


class ReporterType(Enum):
    """Reporting backend named in a configuration.

    ``wandb`` streams figures to Weights & Biases; ``local`` is reserved for
    on-disk reporting, which this branch does not implement yet
    (``BilevelConfig.reporter`` raises ``TypeError`` when it is selected).
    """

    wandb: str = "wandb"
    local: str = "local"
