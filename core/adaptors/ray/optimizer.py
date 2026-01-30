from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.train._internal.checkpoint_manager import _TrainingResult

from core.annotations import override
from core.optimizers.base import Optimizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    def __init__(
        self,
        algo: Algorithm,
        config: RayOptimizerConfig,
    ):
        super().__init__(config)
        self.algo = algo

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        return self.config.num_envs_per_env_runner

    @override(Optimizer)
    def run(self) -> None:
        self.algo.train()

    # @override(Optimizer)
    def evaluate(
        self,
        # parallel_train_future: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> float:
        # TODO if this lags implement this function manuallyu
        result = self.algo.evaluate()
        return result["evaluation"]

    # @override(Optimizer)
    def stop(self) -> None:
        self.algo.stop()

    # @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        return self.algo.save(checkpoint_dir)
