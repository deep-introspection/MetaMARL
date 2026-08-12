from enum import Enum


class ReporterType(Enum):
    wandb: str = "wandb"
    local: str = "local"


class Resolution(Enum):
    env: str = "env_steps"
    inner: str = "train_iters"
    outer: str = "generation"
