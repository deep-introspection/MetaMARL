import logging

import concurrent

from core.optimizers.base import Optimizer

from core.adaptors.ray.optimizer_config import RayOptimizerConfig
from ray.rllib.algorithms.algorithm import Algorithm

from core.annotations import override

from ray.rllib.utils.typing import ResultDict 
from ray.train._internal.checkpoint_manager import _TrainingResult

import logging


from typing import Optional

logger = logging.getLogger(__name__)

# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):

    def __init__(
            self,
            algo: Algorithm,
            config: RayOptimizerConfig,
        ):
        super().__init__(config)
        self.algo = algo
    
    @override(Optimizer)
    def run(self) -> ResultDict:
        result = self.algo.train()
        return result
    
    @override(Optimizer)
    def evaluate(
        self,
        parallel_train_future: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> ResultDict:
        return self.algo.evaluate(parallel_train_future)
    
    @override(Optimizer)
    def stop(self) -> None:
        self.algo.stop()
        
    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        return self.algo.save(checkpoint_dir)

    

        



    
