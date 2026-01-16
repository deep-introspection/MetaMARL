import pytest
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig


class DummyOptimizer(Optimizer):
    pass

class DummyOptimizerConfig(OptimizerConfig):
    pass 



def test_core_framework_end_to_end():
    # obviously we dont need all these for the optimizer, we jut need a few and then the
    # the adaptor takes care of the rest
    config = (
        DummyOptimizerConfig()
        .world(env="dummy_world")
        .framework(**kwargs) #config's DL framework settings
        .resources(**kwargs) #specifies the resources allocated for an Algorithm and its ray actors/works
        .env_runners(**kwargs) # Sets the rollout worker configuration - also this is a ray wrapper
        .training(**kwargs)
        .
    )