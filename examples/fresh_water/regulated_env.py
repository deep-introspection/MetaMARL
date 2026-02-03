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
class WaterRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        water_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # TODO env_cfg to pass ecology params to env
        # TODO default values

        # Ban water-consumption per agent (data center)
        self._agent_bans: dict[str, int] = {}

        # Initial
        self.water_init = water_cfg["water_init"]
        self.max_water = water_cfg["max_water"]
        self.min_water = water_cfg["min_water"]

    def _reset(self):
        # Reset ban counters for all agents
        self._agent_bans = {agent_id: 0 for agent_id in self.agents}

        initial_water = max(
            EPS,
            self.rng.normal(self.water_init, 0.05)
        )

        # TODO initial water amount
        self.S_t = {
            "water": float(np.clip(initial_water, 0.0, self.max_water)),
        }
        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        return obs

    def _is_terminated(self) -> bool:
        # does the episode terminate when the water is depleted ?
        return self._t >= self.horizon

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        # return action * S_t["fish"]
        action = float(np.asarray(action).item())  # cast to scalar - this would be the amound of water consumed
        action = np.clip(action, 0.0, 1.0)
        water_norm = S_t["water"] / self.max_water
        effective_consumption = action * min(1.0, water_norm)
        u = effective_consumption * np.sqrt(max(EPS, water_norm))
        return float(u)

    # TODO this returns a float
    # TODO observation must be a param here not self
    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        quota = max(0.0, u_i - min(self.m.fixed_quota, self.m.prop_quota * S_t["fish"] / self.max_fish))
        ban = float(S_t["fish"] / self.max_fish < self.m.min_stock) * u_i
        v = float(quota + ban)
        return v

    def penalty(self) -> SupportsFloat:
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        water_norm = S_t["water"] / self.max_water

        # Quota violation: consumption beyond allowed limit
        effective_quota = min(
            self.m.fixed_quota,
            self.m.prop_quota * water_norm
        )
        quota_violation = max(0.0, float(u_i) - effective_quota)

        # Critical level violation: any consumption when water is scarce
        critical_violation = 0.0
        if water_norm < self.m.min_stock:
            critical_violation = float(u_i)


        quota_violation = max(0.0, float(u_i) - effective_quota)
        total_desired = sum(desired.values())
        scale = min(1.0, fish_norm / max(EPS, total_desired))
        H = self.max_fish * sum(desired[agent_id] * scale for agent_id in self.agents)

        # logger.debug(
        #     "[TRANSITION] fish=%.4f algae=%.4f fish_norm=%.4f "
        #     "total_desi red=%.6f scale=%.4f H=%.6f",
        #     fish,
        #     algae,
        #     fish_norm,
        #     total_desired,
        #     scale,
        #     H,
        # )

        # Lotka-volterra
        fish_next = fish + self.dt * (self.delta * algae * fish - self.gamma * fish - H)
        algae_next = algae + self.dt * (self.alpha * algae - self.beta * algae * fish)

        # clamp transitions for numerical stability:
        fish_next = np.clip(fish_next, 0.0, self.max_fish)
        algae_next = np.clip(algae_next, 0.0, self.max_algae)

        # logger.debug(
        #     "[NEXT_STATE] fish_next=%.4f algae_next=%.4f",
        #     fish_next,
        #     algae_next,
        # )

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
