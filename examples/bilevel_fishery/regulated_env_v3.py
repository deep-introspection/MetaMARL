import logging
from typing import SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

EPS = 1e-8


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.algae_init = ecology_cfg["algae_init"]
        self.fish_init = ecology_cfg["fish_init"]
        self.max_fish = ecology_cfg["max_fish"]
        self.max_algae = ecology_cfg["max_algae"]
        self.alpha = ecology_cfg["alpha"]
        self.beta = ecology_cfg["beta"]
        self.delta = ecology_cfg["delta"]
        self.gamma = ecology_cfg["gamma"]
        self.dt = ecology_cfg["dt"]

        self.obs_map = [
            "fish_norm",
            "algae_norm",
            "effective_quota",
            "no_fish_zone",
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
        ]

    @override(MultiAgentRegulatedEnv)
    def _reset(self):
        self.S_t = {
            "fish": max(EPS, self.rng.lognormal(np.log(self.fish_init), 0.05)),
            "algae": max(EPS, self.rng.lognormal(np.log(self.algae_init), 0.05)),
        }

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        return obs

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        # TODO move this to parent class
        return self.horizon is not None and (self._t + 1) >= self.horizon

    @override(MultiAgentRegulatedEnv)
    def intrinsic_utility(
            self, 
            A_t: dict[AgentID, ActType],
        ) -> MultiAgentDict:
        return {
            agent_id: (action.item() * self.S_t["fish"]) / self.max_fish
            for agent_id, action in A_t.items()
        }

    @override(MultiAgentRegulatedEnv)
    def violation_signal(
        self,
        u_i: SupportsFloat,
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:
        fish_norm = self.S_t["fish"] / self.max_fish
        
        # TODO move this calculation to mechanism ?
        effective_quota = min(
            self.mechanism.fixed_quota, 
            self.mechanism.prop_quota * fish_norm,
        )
        quota_violation = max(0.0, u_i - effective_quota)

        quota_penalty = min(
            1.0,
            self.mechanism.fine_amount * quota_violation,
        )

        # TODO add stock penalty !
        shortage_severity = max(0.0, self.mechanism.min_stock - fish_norm)
        stock_penalty = min(
            1.0,
            self.mechanism.risk_penalty_scale
            * (shortage_severity ** self.mechanism.risk_penalty_power)
            * float(u_i > 0.0),
        )

        total_penalty = min(1.0,quota_penalty + stock_penalty)

        # For debugging
        self._update_infos(
            key="quota_violation",
            values={agent_id: quota_violation},
        )
        self._update_infos(
            key="quota_penalty",
            values={agent_id: quota_penalty},
        )
        self._update_infos(
            key="stock_penalty",
            values={agent_id: stock_penalty},
        )

        return total_penalty

    @override(MultiAgentRegulatedEnv)
    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
    ) -> dict[str, float]:
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish
        algae = S_t["algae"]

        self._update_infos(
            key="below_target_zone",
            values=float(fish_norm < self.mechanism.target_stock),
        )
        self._update_infos(
            key="target_shortfall",
            values=max(0.0, self.mechanism.target_stock - fish_norm),
        )

        realized_harvest, H, harvest_scale = self._compute_harvest_metrics(A_t=A_t)

        self._update_infos(
            key="harvest",
            values=realized_harvest,
        )
        self._update_infos(
            key="H_total",
            values=H,
        )
        self._update_infos(
            key="harvest_scale",
            values=harvest_scale,
        )
        fish_next = fish + self.dt * (
            self.delta * algae * fish * (1 - fish_norm)
            - self.gamma * fish
            - H
        )
        algae_next = algae + self.dt * (
            self.alpha * algae * (1 - algae / self.max_algae) - self.beta * algae * fish
        )

        fish_next = np.clip(fish_next, 0.0, self.max_fish)
        algae_next = np.clip(algae_next, 0.0, self.max_algae)

        return {"fish": fish_next, "algae": algae_next}

    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        fish_norm = S_t["fish"] / self.max_fish
        algae_norm = S_t["algae"] / self.max_algae
        effective_quota = min(self.mechanism.fixed_quota, self.mechanism.prop_quota * fish_norm)
        no_fish_zone = float(fish_norm < self.mechanism.min_stock)

        return np.array(
            [
                fish_norm,
                algae_norm,
                effective_quota,
                no_fish_zone,
            ],
            dtype=np.float32,
        )

    def _compute_harvest_metrics(
        self, A_t: dict[AgentID, ActType]
    ) -> tuple[dict[AgentID, float], float, float]:
        fish = self.S_t["fish"]
        fish_norm = fish / self.max_fish
        # TODO what if not all fishermen have fish?
        desired = self.intrinsic_utility(A_t=A_t)
        total_desired = float(sum(desired.values()))
        
        # N.B. this normalizes the action such that total fish in the pond is split evently
        # However this is unrealistic, if two fishermen cast nets into a lake, their nets don't 
        # shrink just because another fisherman showed up.
        # TODO independent sequential sampling
        scale = min(1.0, fish_norm / max(EPS, total_desired))
        realized_harvest = {
            agent_id: desired[agent_id] * scale
            for agent_id in self.agents
        }
        H_total = float(sum(realized_harvest.values()))
        return realized_harvest, H_total, scale
