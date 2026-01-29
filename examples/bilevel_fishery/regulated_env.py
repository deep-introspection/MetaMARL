from typing import SupportsFloat

import numpy as np
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.marl_regulated import MultiAgentRegulatedEnv

# TODO add multiagent state in types


# TODO number of agents spawned dynamically as a byproduct of config stating number of agents
class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Ecology params
        # TODO env_cfg to pass ecology params to env
        self.algae_init = ecology_cfg["algae_init"]
        self.fish_init = ecology_cfg["fish_init"]
        self.alpha = ecology_cfg["alpha"]
        self.beta = ecology_cfg["beta"]
        self.delta = ecology_cfg["delta"]
        self.gamma = ecology_cfg["gamma"]
        self.dt = ecology_cfg["dt"]

    def _reset(self):
        self._t = 0
        self.S_t = {
            "fish": self.fish_init,
            "algae": self.algae_init,
        }
        return {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

    def intrinsic_utility(
        self, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        return float(action) * S_t["fish"]

    # TODO this returns a float
    # TODO observation must be a param here not self
    def violation_signal(
        self, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        quota = (0.0, u_i - min(self.m.fixed_quota, self.m.prop_quota * S_t["fish"]))
        ban = float(S_t["fish"] < self.m.min_stock) * u_i
        return quota + ban

    def penalty(self) -> SupportsFloat:
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, MultiAgentDict]
    ) -> MultiAgentEnv:
        fish = self.S_t["fish"]
        algae = self.S_t["algae"]

        # Total Harvest
        H = sum(
            self.intrinsic_utility(action=A_t[agent_id], S_t=S_t[agent_id])
            for agent_id in self.agents
        )

        # Lotka-volterra
        fish_next = max(
            0,
            fish + self.dt * (self.delta * algae * fish - self.gamma * fish - H),
        )
        algae_next = max(
            0,
            fish + self.dt * (self.alpha * algae - self.beta * algae * fish),
        )
        return {"fish": fish_next, "algae": algae_next}

    # TODO abstract this to multiagentenv
    def aggregate_rewards(self, rewards: list[SupportsFloat]) -> SupportsFloat:
        return np.sum(rewards) / len(self.agents)

    # TODO canonical observation in base multiagent env
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        """We assume complete transparency"""
        return np.array([S_t["fish"], S_t["algae"]], dtype=np.float32)

        # # TODO separate state space from observation space
        # self.observation_spaces: dict[AgentID, spaces.Dict] = {
        #     fisher_id: spaces.Dict[
        #         "fish" : spaces.Box(
        #             low=0.0,
        #             high=np.finfo(np.float32).max,
        #             shape=(1, 0),
        #             dtype=np.float32,
        #         ),
        #         "algae" : spaces.Box(
        #             low=0.0,
        #             high=np.finfo(np.float32).max,
        #             shape=(1, 0),
        #             dtype=np.float32,
        #         ),
        #     ]
        #     for fisher_id in fishermen
        # }

        # self.action_space: dict[AgentID, spaces.Dict] = {
        #     fisher_id: spaces.Dict[
        #         "fish_harvest" : spaces.Box(
        #             low=0.0,
        #             high=np.finfo(np.float32).max,
        #             shape=(1, 0),
        #             dtype=np.float32,
        #         )
        #     ]
        #     for fisher_id in fishermen
        # }

        # # Current state
        # self._t = 0
        # self.S_t: dict[str, np.float32] = {
        #     "fish": self.fish_init,
        #     "algae": self.algae_init,
        # }
