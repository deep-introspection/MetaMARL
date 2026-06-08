from enum import Enum, auto
from typing import Any, Literal, Optional

import wandb


class DataParser(Enum):
    rllib_new_api = "rllib_new_api"
    rllib_old_api = "rllib_old_api"


class WandbLogger:
    def __init__(self, wandb_run: wandb.Run):
        self.wandb_run = wandb_run

        self._plots_to_log: dict = {}

    def _make_table_name(
        self,
        optimizer_name: str,
        metric_key: str,  # TODO replace with string alias
        metric_group: Optional[str] = None,  # TODO
    ) -> str:
        # TODO
        pass

    def _parse_data_rllib_new_api(self, data: dict[Any]) -> dict[Any]:
        pass
