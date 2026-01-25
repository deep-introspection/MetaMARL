from core.world.base import World
from core.world.context import Context
from core.wrappers.context_wrapper import ContextWrapper


class RegulatedEnv(ContextWrapper):
    def __init__(self, world: World) -> None:
        super().__init__(world=world)

    def _get_violation_signal(self) -> float:
        raise NotImplementedError

    def _get_violation_penalty(self) -> float:
        raise NotImplementedError


from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override


class ContextEnv(Env):
    """Modify observations, rewards and actions from :meth:`Env.reset` and :meth:`Env.step` using :meth:`observation`,
    :meth:`rewards` and :meth:`action`functions by injecting the world's context into it.
    Helper functions may be added to extract different information from the world context.

    """

    def __init__(self, world: World) -> None:
        """
        Constructor for the context wrapper.

        Args:
            world: world from/to which contexts are published and retreived
        """
        super().__init__(self)
        self.world = world

    def _step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # retreive context from world goes here
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._step(
            self.action(action)
        )
        return (
            self.observation(observation),
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


class ContextWrapper(Wrapper):
    """Modify observations, rewards and actions from :meth:`Env.reset` and :meth:`Env.step` using :meth:`observation`,
    :meth:`rewards` and :meth:`action`functions by injecting the world's context into it.
    Helper functions may be added to extract different information from the world context.

    """

    def __init__(self, world: World) -> None:
        """
        Constructor for the context wrapper.

        Args:
            world: world from/to which contexts are published and retreived
        """
        super().__init__(self)
        self.world = world

    @override(Wrapper)
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        env_context = Context(
            id=self.env.__class__.__name__,
            payload={"observation": observation, "reward": reward, "action": action},
        )
        self.world.update_context(env_context)
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
