import logging
from abc import abstractmethod
from typing import Any, SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.types import OptimizerID
from core.world.base import World
from core.world.context import EnvStepContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type

# TODO remove inheritance from regulatedEnv completley
# class SingleAgentBaseEnv(gym.Env): ...
# class MultiAgentBaseEnv(MultiAgentEnv): ...

# class RegulatedEnv(SingleAgentBaseEnv): ...
# class MultiAgentRegulatedEnv(MultiAgentBaseEnv): ...


class MultiAgentRegulatedEnv(RegulatedEnv, MultiAgentEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID,
        agents: list[AgentID],
        **kwargs,
    ):
        super().__init__(world=world, opt_id=opt_id, **kwargs)
        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.observation_spaces = kwargs.get("observation_spaces", {})

        # TODO move this to baseenv later
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

    @abstractmethod
    def _step(
        self, action_dict: MultiAgentDict = None
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> MultiAgentDict:
        raise NotImplementedError

    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self._base_reset(seed=seed)
        obs = self._reset()
        infos = {agent_id: {} for agent_id in self.agents}
        return obs, infos

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        actions = self.action(action_dict)
        obs, rewards, terminated, truncated, infos = self._step(actions)

        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index if self.m_ctx else None,
                observation=obs,
                observation_map=self.obs_map,
                reward=rewards,
                action=actions,
                info=infos,
            )
        )

        self._t += 1
        return obs, rewards, terminated, truncated, infos

    @abstractmethod
    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
        **kwargs,
    ) -> dict[str, float]:
        """S_{t+1} = T(S_t, A_t)"""
        ...

    @abstractmethod
    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """u_i = U(a_i, S_t)"""
        ...

    @abstractmethod
    @override(RegulatedEnv)
    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """v_i = V(a_i, S_t, M)"""
        ...

    @abstractmethod
    @override(RegulatedEnv)
    def penalty(self) -> np.ndarray:
        """λ = λ(M)"""
        ...

    @abstractmethod
    def _observation(
        self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]
    ) -> ObsType:
        """o_i = O_i(S_t)"""
        ...

    # TODO Restrict Any Type
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> Any:
        """o_i = O_i(S_t, theta)"""
        # TODO may wanna normalize base_obs later
        base_obs = self._observation(agent_id=agent_id, S_t=S_t)
        theta = self.m.to_vector()
        return np.concatenate([base_obs, theta], axis=0)

    @abstractmethod
    def _is_truncated(self) -> bool: ...

    @abstractmethod
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict: ...

    # @override(BaseEnv)
    # def reward(self, agent_id: AgentID, action: ActType) -> SupportsFloat:
    #     o_i = self.observation(agent_id=agent_id, S_t=self.S_t)
    #     u_i = self.intrinsic_utility(agent_id=agent_id, action=action, observation=o_i)
    #     return u_i - self.penalty() * self.violation_signal(
    #         agent_id=agent_id, reward=u_i, observation=o_i
    #     )