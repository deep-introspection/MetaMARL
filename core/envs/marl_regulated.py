from abc import abstractmethod
from typing import SupportsFloat, Tuple

import numpy as np
import ray
from gymnasium.core import ActType, ObsType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.mechanism.base import Mechanism
from core.types import OptimizerID
from core.world.base import World

# TODO create a reward type


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

    @abstractmethod
    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
        **kwargs,
    ) -> MultiAgentDict:
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
    def penalty(self) -> SupportsFloat:
        """λ = λ(M)"""
        ...

    @abstractmethod
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> ObsType:
        """o_i = O_i(S_t)"""
        ...

    @abstractmethod
    def _is_terminated(self) -> bool: ...

    @abstractmethod
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict: ...

    @override(BaseEnv)
    def reward(self, agent_id: AgentID, action: ActType) -> SupportsFloat:
        o_i = self.observation(agent_id=agent_id, S_t=self.S_t)
        u_i = self.intrinsic_utility(agent_id=agent_id, action=action, observation=o_i)
        return u_i - self.penalty() * self.violation_signal(
            agent_id=agent_id, reward=u_i, observation=o_i
        )

    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        # update institution mechanism from world
        m_params: np.ndarray = ray.get(self.world.get_mechanism.remote(self.env_id))
        self.m: Mechanism = self.m_space.decode(m_params)

        rewards = {}
        for agent_id in self.agents:
            u = self.intrinsic_utility(agent_id, action_dict[agent_id], self.S_t)
            v = self.violation_signal(agent_id, u, self.S_t)
            rewards[agent_id] = u - self.penalty() * v

        rewards = self.aggregate_rewards(rewards)

        # update obsevations
        self.S_t = self.transition_kernel(A_t=action_dict, S_t=self.S_t)

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

        # check terminated and truncated conditions
        terminated = {"__all__": self.is_terminated()}
        truncated = {"__all__": False}

        return obs, rewards, terminated, truncated, {}
