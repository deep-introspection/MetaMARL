"""Integration test for the RLlib-backed inner optimizer on CartPole."""

import pytest


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "RayOptimizerConfig.build_optimizer() now requires a World actor and a "
        "reporter and constructs a regulated env; a plain CartPole run is no "
        "longer expressible. Covered by the per-feature bilevel smoke tests."
    )
)
def test_ppo_cartpole_training():
    pass
