from typing import override

import numpy as np
from gymnasium import spaces

from core.wrappers.context_wrapper import ContextWrapper


class RegulatoryEnv(ContextWrapper):
    """Wrapper that injects mechanism parameters from World into FisheryEnv.

    This wrapper reads the current MechanismContext from the World and
    applies violation penalties to the reward based on quota violations.
    """

    def __init__(self, env: Env, world: World, meta_opt_id: str) -> None:
        super().__init__(env, world)
        self._meta_opt_id = meta_opt_id
        self._cached_mechanism: Optional[MechanismContext] = None

        # TODO the action spaces are the mechanism parameters that are being sampled from
        # TODO not the context

        # TODO dynmaically define the mechanism params to be inserted here (action space)
        # TODO action spaces to world mechanism context - how to generalize ?

        # Define ES Action Spaces
        self.action_spaces = {
            "fixed_quota": spaces.Box(low=0.0, high=1.0, shape=(1.0), dtype=np.float32),
            "prop_quota": spaces.Box(low=0.0, high=1.0, shape=(1.0), dtype=np.float32),
            "min_stock": spaces.Box(low=0.0, high=1.0, shape=(1.0), dtype=np.float32),
            "fine_amount": spaces.Box(low=0.0, high=1.0, shape=(2.0), dtype=np.float32),
            "fixed_quota": spaces.Box(low=0.0, high=1.0, shape=(10), dtype=np.float32),
        }
        self.observation_spaces = self.action_spaces

    # TODO step
    @override
    def step(self, action):
        # Publish regulator action -> world mechanism
        # actually the observation would be the mechanism params and the
        mechanism_ctx = map_unit_vector_to_mechanism(action)
        self._world.update_context(mechanism_ctx)

        # Run inner learning system (PPO)
        self.downstream_optimizer.run()

        # read context published by downstream optimizer from world
        fitness_ctx = self._world.get_context("fitness_context")

        # compute reward from fitness context
        reward = self._compute_fitness(fitness_ctx)

        terminated = False
        truncated = False

        # TODO what is an observation ???
        obs = np.zeros(1)

        return obs, reward, terminated, truncated, {}
