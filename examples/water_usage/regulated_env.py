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


class WaterRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        env_cfg = ecology_cfg or {}
        # Ecology params
        self.water_init = env_cfg.get("water_init", 80.0)
        self.max_water = env_cfg.get("max_water", 100.0)
        self.inflow_rate = env_cfg.get("inflow_rate", 1.0)
        # time-step used in transition kernel (mirror fishery `dt` semantics)
        self.dt = env_cfg.get("dt", 1.0)

        # Restriction tracking per agent
        self._agent_restrictions = {}

    def _reset(self):
        # Reset restriction counters for all agents
        self._agent_restrictions = {agent_id: 0 for agent_id in self.agents}

        self.S_t = {
            "water": max(EPS, self.rng.lognormal(np.log(self.water_init), 0.05)),
        }

        obs = {agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents}
        return obs

    def _is_truncated(self) -> bool:
        return self._t >= self.horizon

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        action = float(np.asarray(action).item())
        water_norm = S_t["water"] / self.max_water
        u = action * water_norm
        return u

    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        # quota-like violation and resource-level ban
        quota = max(0.0, u_i - min(self.m.fixed_quota, self.m.prop_quota * S_t["water"] / self.max_water))
        restriction = float(S_t["water"] / self.max_water < self.m.min_stock) * u_i
        v = float(quota + restriction)
        return v

    def penalty(self) -> SupportsFloat:
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentDict, S_t: dict[str, float]
    ) -> dict[str, float]:
        water = self.S_t["water"]

        desired = {
            agent_id: self.intrinsic_utility(agent_id=agent_id, action=A_t[agent_id], S_t=S_t)
            for agent_id in self.agents
        }
        total_desired = sum(desired.values())
        scale = min(1.0, water / max(EPS, total_desired))
        usage = self.max_water * sum(desired[agent_id] * scale for agent_id in self.agents)

        # simple replenishment
        water_next = water + self.dt * (self.inflow_rate) - usage
        water_next = np.clip(water_next, 0.0, self.max_water)

        return {"water": water_next}

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}

    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        water_norm = S_t["water"] / self.max_water

        # Restriction status normalized to [0,1]
        restriction_remaining = 0.0
        if self.m.ban_period > 0:
            restriction_remaining = self._agent_restrictions.get(agent_id, 0) / self.m.ban_period

        effective_quota = min(self.m.fixed_quota, self.m.prop_quota * water_norm)
        no_water_zone = float(water_norm < self.m.min_stock)

        return np.array([
            water_norm, 0.0, restriction_remaining, effective_quota, no_water_zone
        ], dtype=np.float32)

    def _is_restricted(self, agent_id: AgentID) -> bool:
        return self._agent_restrictions.get(agent_id, 0) > 0

    def _decrement_restriction(self, agent_id: AgentID) -> None:
        if self._agent_restrictions.get(agent_id, 0) > 0:
            self._agent_restrictions[agent_id] -= 1

    def _apply_restriction(self, agent_id: AgentID) -> None:
        if self.m.ban_period > 0:
            self._agent_restrictions[agent_id] = self.m.ban_period
