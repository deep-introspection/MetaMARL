from typing import override

from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig

from gymnasium import spaces

import numpy as np


class DummyOptimizer(Optimizer):
    pass


class SuperDummyOptimizer(Optimizer):
    pass


class DummyOptimizerConfig(OptimizerConfig):
    @override
    def training(self):
        print("training test")
        return None


class SuperDummyOptimizerConfig(OptimizerConfig):
    pass

class DummyEnv(gym.Env):

    def __init__(self):
        self.action_space: spaces.Space(shape=(2,), dtype=np.float32
        )




def test_core_framework_end_to_end():
    # obviously we dont need all these for the optimizer, we jut need a few and then the
    # the adaptor takes care of the rest
    dum_opt_cfg = (
        DummyOptimizerConfig()
        .world(world="dummy_world")
        .training(...)
        .ressources(...)
        .evaluation(...)
        .reporting(...)
        .checkpointing(...)
        .fault_tolerance(...)
        .experimental(...)
    )

    super_dum_opt_cfg = SuperDummyOptimizerConfig()

    dummy_optimizer = dum_opt_cfg.build_optimizer()
    super_dummy_optimizer = super_dum_opt_cfg.build_optimizer()

    # what if the the super_dummy is done first
    # think about how we are going to set guardrails here
    super_dummy_optimizer.set_downstream(optimizer=dummy_optimizer)
    super_dummy_optimizer.run()

    # evaluate the entire system
    super_dummy_optimizer.evaluate()

    # save checkpoint and visualize results
    super_dummy_optimizer.save_checkpoint()

    super_dummy_optimizer.visualize_results()
