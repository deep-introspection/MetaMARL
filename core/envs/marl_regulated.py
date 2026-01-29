from abc import abstractmethod
from typing import SupportsFloat, Tuple

import numpy as np
import ray
from gymnasium.core import ActType, Env, ObsType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.regulated import RegulatedEnv
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.world.base import World

# TODO create a reward type


class MultiAgentRegulatedEnv(RegulatedEnv, MultiAgentEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID,
        mechanism_space: MechanismSpace,
        **kwargs,
    ):
        super().__init__(world=world, opt_id=opt_id, **kwargs)
        self.m_space: MechanismSpace = mechanism_space
        self.m: Mechanism = None

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
    def penalty(self, **kwargs) -> SupportsFloat:
        """λ = λ(M)"""
        ...

    @abstractmethod
    @override(Env)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> ObsType:
        """o_i = O_i(S_t)"""
        ...

    @abstractmethod
    def is_terminated(self, S_t: dict[str, MultiAgentDict]) -> bool: ...

    @abstractmethod
    def aggreagate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict: ...

    @override(Env)
    def reward(self, agent_id: AgentID, action: ActType) -> SupportsFloat:
        o_i = self.observation(agent_id=agent_id)
        u_i = self.intrinsic_utility(agent_id=agent_id, action=action, observation=o_i)
        return u_i - self.penalty(agent_id, o_i, u_i) * self.violation_signal(
            agent_id=agent_id, reward=u_i, observation=o_i
        )

    @abstractmethod
    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        # update institution mechanism from world
        m_params: np.ndarray = ray.get(self.world.get_latest_mechanism.remote())
        self.m: Mechanism = self.m_space.decode(m_params)

        # get current observations
        # TODO ???

        # get intrinsic reward dict (instrinsic reward may implement different intrinsic rewards per agent types)
        rewards = {
            agent_id: self.reward(agent_id=agent_id, action=action_dict[agent_id])
            for agent_id in self.agents
        }

        # apply regulatory mechanism for each agent and aggregate rewards
        rewards = self.aggreagate_rewards(rewards)

        # update obsevations
        observations = self.transition_kernel(
            action=action_dict, rewards=rewards, observation=observations
        )

        # check terminated and truncated conditions
        terminated = {fisher_id: False for fisher_id in self.agents}
        truncated = {fisher_id: self._t >= self.horizon for fisher_id in self.agents}
        terminated["__all__"] = False
        truncated["__all__"] = any(truncated.values())

        self._t = 1

        return observations, rewards, terminated, truncated, {}
