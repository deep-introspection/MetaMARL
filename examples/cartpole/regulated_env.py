import logging
from typing import SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ActType
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.world.context import EnvStepContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# Numerical stability constant
EPS = 1e-8

# TODO add multiagent state in types
# TODO ban proportional to violation severity


# TODO number of agents spawned dynamically as a byproduct of config stating number of agents
class CartpoleRegulatedEnv(MultiAgentRegulatedEnv):
    """Single-agent CartPole-v1 wrapped as a ``MultiAgentRegulatedEnv``.

    Bridges the standard Gymnasium ``CartPole-v1`` environment into the
    bilevel-fishery multi-agent regulated API so that the APPO/PPO inner
    optimizer can treat it as a drop-in regulated environment.  The
    regulatory mechanism stubs (penalty, violation signal, intrinsic
    utility) all return zero because CartPole is used purely as a sanity
    check for the bilevel plumbing, not for mechanism design research.

    Parameters
    ----------
    render_mode : str or None, optional
        Render mode forwarded to ``gymnasium.make("CartPole-v1")``.
        ``None`` disables rendering (default).
    **kwargs
        Forwarded to :class:`~core.envs.marl_regulated.MultiAgentRegulatedEnv`.
        Must include an ``agents`` mapping with exactly one agent entry.

    Raises
    ------
    ValueError
        If the number of agents provided is not exactly 1.
    """

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Ensure number of agents == 1
        if len(self.agents) != 1:
            raise ValueError(
                "CartPoleRegulatedEnv is a single-agent QC environment. "
                f"Got agents={self.agents}."
            )
        self.agent_id = self.agents[0]

        # Initialize Cartpole env
        self.env = gymnasium.make("CartPole-v1", render_mode=render_mode)

        # override observationa and action spaces
        self.action_spaces = {self.agent_id: self.env.action_space}
        self.observation_spaces = {self.agent_id: self.env.observation_space}
        self.action_space = spaces.Dict(self.action_space)
        self.observation_space = spaces.Dict(self.observation_space)

        self.S_t: np.ndarray | None = None
        self._last_reset_seed: int | None = None

    @override(MultiAgentRegulatedEnv)
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Reset the CartPole environment and return initial observations.

        Parameters
        ----------
        seed : int or None, optional
            Random seed forwarded to the underlying Gymnasium environment.
        options : dict or None, optional
            Additional reset options forwarded to ``CartPole-v1``.

        Returns
        -------
        tuple[MultiAgentDict, MultiAgentDict]
            ``(observations, infos)`` dictionaries keyed by ``agent_id``.
        """
        self._base_reset(seed=seed)
        self._last_reset_seed = seed

        obs, info = self.env.reset(seed=seed, options=options)
        self.S_t = np.asarray(obs, dtype=np.float32)

        observations = {self.agent_id: self.S_t}
        infos = {self.agent_id: info}
        return observations, infos

    def _reset(self) -> MultiAgentDict:
        """Internal reset used by the base class for episode restarts.

        Replays the last seed so that deterministic episode repetition is
        possible within a single outer-loop iteration.

        Returns
        -------
        MultiAgentDict
            Initial observation keyed by ``agent_id``.
        """
        obs, _ = self.env.reset(seed=self._last_reset_seed)
        self.S_t = np.asarray(obs, dtype=np.float32)
        return {self.agent_id: self.S_t}

    @override(MultiAgentRegulatedEnv)
    def step(self, action_dict: MultiAgentDict):
        """Advance the CartPole environment by one timestep.

        Parameters
        ----------
        action_dict : MultiAgentDict
            Dictionary mapping ``agent_id`` to the discrete action (0 or 1).

        Returns
        -------
        tuple
            ``(observations, rewards, terminateds, truncateds, infos)`` where
            each value is a dictionary keyed by ``agent_id`` (plus ``"__all__"``
            for the done flags).
        """
        action = int(action_dict[self.agent_id])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.S_t = np.asarray(obs, dtype=np.float32)

        observations = {self.agent_id: self.S_t}
        rewards = {self.agent_id: float(reward)}
        terminateds = {
            self.agent_id: bool(terminated),
            "__all__": bool(terminated),
        }
        truncateds = {
            self.agent_id: bool(truncated),
            "__all__": bool(truncated),
        }
        infos = {self.agent_id: info}

        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index if self.m_ctx else None,
                observation=observations,
                observation_map=self.obs_map,
                reward=rewards,
                action={self.agent_id: action},
                info=infos,
            )
        )

        self._t += 1
        return observations, rewards, terminateds, truncateds, infos

    def _is_truncated(self) -> bool:
        """Return whether the episode should be truncated by the wrapper.

        CartPole-v1 manages its own truncation internally; this hook always
        returns ``False`` so the base-class logic does not interfere.

        Returns
        -------
        bool
            Always ``False``.
        """
        # Gym Cartpole truncation is handled by wrapped env
        return False

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Return the agent's intrinsic utility for the current step.

        CartPole rewards are provided directly by the wrapped environment, so
        this stub always returns ``0.0``.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the acting agent (unused).
        action : ActType
            Action taken by the agent (unused).
        S_t : dict[str, MultiAgentDict]
            Current environment state (unused).

        Returns
        -------
        SupportsFloat
            Always ``0.0``.
        """
        del agent_id, action, S_t
        return 0.0

    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Return the regulatory violation signal for the current step.

        CartPole has no regulatory mechanism, so this stub always returns
        ``0.0``.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the acting agent (unused).
        u_i : SupportsFloat
            Intrinsic utility of the agent (unused).
        S_t : dict[str, MultiAgentDict]
            Current environment state (unused).

        Returns
        -------
        SupportsFloat
            Always ``0.0``.
        """
        del agent_id, u_i, S_t
        return 0.0

    def penalty(self) -> SupportsFloat:
        """Return the regulatory penalty for the current step.

        CartPole has no regulatory penalty, so this stub always returns
        ``0.0``.

        Returns
        -------
        SupportsFloat
            Always ``0.0`` (``np.float32``).
        """
        return np.array(0.0, dtype=np.float32)

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        """Compute the next environment state given actions and current state.

        Not implemented for CartPole because the state transition is fully
        handled by the wrapped Gymnasium environment.

        Parameters
        ----------
        A_t : MultiAgentEnv
            Joint action (unused — handled by wrapped env).
        S_t : dict[str, float]
            Current state (unused — handled by wrapped env).

        Returns
        -------
        dict[str, float]
            Next state (not yet implemented; returns ``None``).
        """
        pass  # TODO

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        """Aggregate rewards across agents without scaling or clipping."""
        return rewards

    # TODO canonical observation in base multiagent env
    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        """We assume complete transparency. Observations normalized to [0, 1]."""
        del agent_id
        return np.asarray(S_t, dtype=np.float32)

    def close(self) -> None:
        """Close the underlying Gymnasium CartPole environment and free resources."""
        self.env.close()
