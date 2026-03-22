from enum import Enum


class ReporterType(Enum):
    wandb: str = "wandb"
    local: str = "local"
