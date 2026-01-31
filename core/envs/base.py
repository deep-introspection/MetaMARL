from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.world.base import World
from core.world.context import Context, ContextSchema, EnvStepContext, MechanismContext


class BaseEnv(Env):
    """Base environment that directly interacts with the World."""

    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        mechanism_space: MechanismSpace = None,
        vector_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.world = world
        self._opt_id = opt_id
        self._t = 0
        self.m_space: MechanismSpace = mechanism_space()
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self.env_id = vector_index

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        self._opt_id = opt_id

    # private methods
    def _publish(self, payload: ContextSchema):
        ctx = Context(
            id=None,
            opt_id=self._opt_id,
            step=self._t,
            env=self.__class__.__name__,
            payload=payload,
        )
        ray.get(self.world.append_context.remote(ctx))

    @abstractmethod
    def _step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        raise NotImplementedError

    def _base_reset(self, *, seed=None):
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._pre_reset()

    @abstractmethod
    def _pre_reset(self) -> None:
        pass

    @abstractmethod
    def _reset(self):
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raw_obs, raw_reward, terminated, truncated, info = self._step(
            self.action(action)
        )

        obs = self.observation(raw_obs)
        reward = self.reward(raw_reward)

        if self.m_ctx is not None:
            m_idx = self.m_ctx.index
        m_idx = None
        
        # Publish env context to World
        self._publish(
            EnvStepContext(
                mechanism=m_idx,  # TODO link with mechanismID rather than mechanism
                observation=obs,
                reward=reward,
                action=action,
                info=info,
            )
        )
        self._t += 1
        return obs, reward, terminated, truncated, info

    @override(Env)
    def reset(self, *, seed=None, options=None):
        self._base_reset(seed=seed)
        obs = self._reset()
        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index,
                observation=obs,
                reward=0.0,
                action=None,
                info={},
            )
        )
        return obs

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        return observation

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        return reward

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        return action
