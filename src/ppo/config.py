from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class PPOptimizerConfig(RayOptimizerConfig):
    algo_class: Algorithm = PPO
