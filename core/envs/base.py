from abc import abstractmethod
from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override
from core.types import ContextID, OptimizerID
from core.world.base import World
from core.world.context import Context, ContextSchema, EnvStepContext


class BaseEnv(Env):
    """Base environment that directly interacts with the World."""

    def __init__(self, *, world: World, **kwargs) -> None:
        super().__init__()
        self.world = world
        self._ctx_id: ContextID | None = None
        self._opt_id: OptimizerID | None = None

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        self._opt_id = opt_id

    # private methods
    def _publish(self, payload: ContextSchema):
        ctx = Context(
            id=self._ctx_id,
            opt_id=self._opt_id,
            payload=payload,
        )

        if self._ctx_id is None:
            self._ctx_id = self.world.set_new_context(ctx)
        else:
            ctx.id = self._ctx_id
            self.world.update_context(ctx)

    @abstractmethod
    def _step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        raise NotImplementedError

    @abstractmethod
    def _reset(self):
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._step(self.action(action))

        # Publish env context to World
        self._publish(
            EnvStepContext(
                observation=obs,
                reward=reward,
                action=action,
            )
        )

        return (
            self.observation(obs),
            self.reward(reward),
            terminated,
            truncated,
            info,
        )

    @override(Env)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        obs = self._reset()

        self._publish(
            EnvStepContext(
                observation=obs,
                reward=0.0,
                action=None,
            )
        )

        return obs, {}

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
