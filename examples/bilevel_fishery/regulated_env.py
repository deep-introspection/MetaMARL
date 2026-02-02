import logging
from typing import SupportsFloat

import numpy as np
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

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
class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Ban tracking per agent
        self._agent_bans: dict[str, int] = {}

        # Ecology params
        # TODO env_cfg to pass ecology params to env
        # TODO default values
        self.algae_init = ecology_cfg["algae_init"]
        self.fish_init = ecology_cfg["fish_init"]
        self.max_fish = ecology_cfg["max_fish"]
        self.max_algae = ecology_cfg["max_algae"]
        self.alpha = ecology_cfg["alpha"]
        self.beta = ecology_cfg["beta"]
        self.delta = ecology_cfg["delta"]
        self.gamma = ecology_cfg["gamma"]
        self.dt = ecology_cfg["dt"]
        self.agent_bans: dict[AgentID, int] = {}


    def _reset(self):
        # Reset ban counters for all agents
        self._agent_bans = {agent_id: 0 for agent_id in self.agents}

        self.S_t = {
            "fish": max(EPS, self.rng.lognormal(np.log(self.fish_init), 0.05)),
            "algae": max(EPS, self.rng.lognormal(np.log(self.algae_init), 0.05)),
        }
        self.agent_bans = {agent_id: 0 for agent_id in self.agents}
        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        return obs

    def _is_terminated(self) -> bool:
        return self._t >= self.horizon

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        if self.agent_bans.get(agent_id, 0) > 0:
            return 0.0
        # return action * S_t["fish"]
        action = float(np.asarray(action).item())  # cast to scalar
        fish_norm = S_t["fish"] / self.max_fish
        u = action * fish_norm
        return u

    # TODO this returns a float
    # TODO observation must be a param here not self
    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:

        # fish = float(S_t["fish"])
        # fish_norm = fish / self.max_fish
        # quota_norm = min(
        #     self.m.fixed_quota / self.max_fish,
        #     self.m.prop_quota * fish_norm,
        # )
        # quota_violation = max(0.0, u_i - quota_norm)
        # if fish < self.m.min_stock:
        #     depth = (self.m.min_stock - fish) / max(EPS, self.m.min_stock)
        # else:
        #     depth = 0.0
        # # TODO add stock weight to mechanism
        # # violation = quota_violation + self.m.stock_weight * depth
        # violation = quota_violation + 1.0 * depth

        # if violation > EPS:
        #     self.agent_bans[agent_id] = max(self.agent_bans.get(agent_id, 0), self.m.ban_period)

        # return float(violation)

        quota = max(0.0, u_i - min(self.m.fixed_quota, self.m.prop_quota * S_t["fish"] / self.max_fish))
        ban = float(S_t["fish"] / self.max_fish < self.m.min_stock) * u_i
        v = float(quota + ban)
        # if v > 0.0:
        # logger.info(
        #     "[VIOLATION] %s u=%.6f quota=%.6f ban=%.6f total=%.6f",
        #     agent_id,
        #     u_i,
        #     quota,
        #     ban,
        #     v,
        # )
        return v


    def penalty(self) -> SupportsFloat:
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        fish = self.S_t["fish"]
        algae = self.S_t["algae"]

        # Total Absolute harvest
        fish_norm = fish / self.max_fish
        desired = {
            agent_id: self.intrinsic_utility(
                agent_id=agent_id, action=A_t[agent_id], S_t=S_t
            )
            for agent_id in self.agents
        }
        total_desired = sum(desired.values())
        scale = min(1.0, fish_norm / max(EPS, total_desired))
        H = self.max_fish * sum(desired[agent_id] * scale for agent_id in self.agents)


        # Lotka-volterra
        fish_next = fish + self.dt * (self.delta * algae * fish - self.gamma * fish - H)
        algae_next = algae + self.dt * (self.alpha * algae - self.beta * algae * fish)

        # clamp transitions for numerical stability:
        fish_next = np.clip(fish_next, 0.0, self.max_fish)
        algae_next = np.clip(algae_next, 0.0, self.max_algae)

        for agent_id in self.agent_bans:
            if self.agent_bans[agent_id] > 0:
                self.agent_bans[agent_id] -= 1

        return {"fish": fish_next, "algae": algae_next}

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        """Aggregate rewards across agents without scaling or clipping."""
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}

    # TODO canonical observation in base multiagent env
    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        """We assume complete transparency. Observations normalized to [0, 1]."""
        fish_norm = S_t["fish"] / self.max_fish
        algae_norm = S_t["algae"] / self.max_algae

        # Ban status: normalized remaining ban steps (0 = not banned, 1 = just banned)
        ban_remaining = 0.0
        if self.m.ban_period > 0:
            ban_remaining = self._agent_bans.get(agent_id, 0) / self.m.ban_period

        # Computed signals to help learning
        effective_quota = min(self.m.fixed_quota, self.m.prop_quota * fish_norm)
        no_fish_zone = float(fish_norm < self.m.min_stock)

        return np.array([
            fish_norm, algae_norm, ban_remaining,
            effective_quota, no_fish_zone,
        ], dtype=np.float32)

    def _is_banned(self, agent_id: AgentID) -> bool:
        """Check if agent is currently banned."""
        return self._agent_bans.get(agent_id, 0) > 0

    def _decrement_ban(self, agent_id: AgentID) -> None:
        """Decrement ban counter for agent."""
        if self._agent_bans.get(agent_id, 0) > 0:
            self._agent_bans[agent_id] -= 1

    def _apply_ban(self, agent_id: AgentID) -> None:
        """Apply ban to agent based on mechanism's ban_period."""
        if self.m.ban_period > 0:
            self._agent_bans[agent_id] = self.m.ban_period
