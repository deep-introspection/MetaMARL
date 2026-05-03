from enum import Enum


class ReporterType(Enum):
    """Backend to which experiment metrics are reported.

    Values
    ------
    wandb : str
        Metrics are streamed to Weights & Biases via the ``wandb`` Python SDK.
    local : str
        Metrics are written to a local destination (e.g. CSV file or stdout).
    """

    wandb: str = "wandb"
    local: str = "local"
