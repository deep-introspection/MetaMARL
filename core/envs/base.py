from abc import abstractmethod
from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override
from core.world.base import World
from core.world.context import Context


class BaseEnv(Env):
    """Base environment that directly interacts with the World."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world

    def _publish(self, ctx: Context):
        # TODO get specific ctx id and opt_id (IMPORTANT)
        self.world.update_context(ctx)

    @abstractmethod
    def _step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        out = self._step(self.action(action))

        if out is None:
            obs = self.observation(None)
            reward = self.reward(0.0)
            terminated = False
            truncated = False
            info = {}
        else:
            obs, reward, terminated, truncated, info = out

        # Publish env context to World
        self._publish(
            Context(
                id=None,
                opt_id=None,
                payload={
                    "observation": obs,
                    "reward": reward,
                    "action": action,
                },
            )
        )

        return (
            self.observation(obs),
            self.reward(reward),
            terminated,
            truncated,
            info,
        )

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
