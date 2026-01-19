from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.world.base import World


# TODO do we really want the output of the hidden functions to be Context ?
class ContextWrapper(Wrapper, ABC):
    """Modify observations, rewards and actions from :meth:`Env.reset` and :meth:`Env.step` using :meth:`observation`,
    :meth:`rewards` and :meth:`action`functions by injecting the world's context into it.
    Helper functions may be added to extract different information from the world context.

    """

    def __init__(self, env: Env, world: World) -> None:
        """
        Constructor for the context wrapper.

        Args:
            env: Environment to be wrapped
        """
        super().__init__(env)
        self._world = world

    @abstractmethod
    def _get_violation_signal(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _get_violation_penalty(self) -> float:
        raise NotImplementedError

    # TODO abstract method for other signal types
    # @abstractmethod
    # def _get_signal_from_ctx(self) -> float:
    #     raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(
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
        raise NotImplementedError

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        penalty = self._get_violation_penalty()
        signal = self._get_violation_signal()
        return reward - penalty * signal

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError
