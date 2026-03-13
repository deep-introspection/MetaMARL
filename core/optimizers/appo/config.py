from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class APPOptimizerConfig(RayOptimizerConfig):
    algo_class: Algorithm = APPO
