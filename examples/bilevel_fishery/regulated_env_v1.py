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

    def _reset(self):
        self.S_t = {
            "fish": max(EPS, self.rng.lognormal(np.log(self.fish_init), 0.05)),
            "algae": max(EPS, self.rng.lognormal(np.log(self.algae_init), 0.05)),
        }

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        return obs

    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        rewards = {}
        fines = {}
        utilities = {}
        violations = {}
        quota_violations = {}
        preventive_penalties = {}

        effective_actions = dict(action_dict)

        for agent_id in self.agents:
            u = float(
                self.intrinsic_utility(agent_id, effective_actions[agent_id], self.S_t)
            )

            v_dict = self.violation_signal(
                agent_id=agent_id,
                u_i=u,
                S_t=self.S_t,
                A_t=effective_actions,
            )

            quota_violation = float(v_dict["quota"])
            preventive_penalty = float(v_dict["preventive"])
            v = float(v_dict["total"])

            utilities[agent_id] = u
            quota_violations[agent_id] = quota_violation
            preventive_penalties[agent_id] = preventive_penalty
            violations[agent_id] = v
            rewards[agent_id] = u - v
            fines[agent_id] = preventive_penalty

        rewards = self.aggregate_rewards(rewards)

        realized_harvest, H_total, harvest_scale = self._compute_harvest_metrics(
            effective_actions, self.S_t
        )
        S_t = self.S_t.copy()

        self.S_t = self.transition_kernel(A_t=effective_actions, S_t=S_t)

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

        time_limit = self._is_truncated()

        terminated = {aid: False for aid in self.agents}
        terminated["__all__"] = False

        truncated = {aid: time_limit for aid in self.agents}
        truncated["__all__"] = time_limit

        infos = {
            agent_id: {
                "harvest": realized_harvest.get(agent_id, 0.0),
                "intrinsic_utility": utilities.get(agent_id, 0.0),
                "violation_signal": violations.get(agent_id, 0.0),
                "quota_violation": quota_violations.get(agent_id, 0.0),
                "preventive_penalty": preventive_penalties.get(agent_id, 0.0),
                "fine": fines.get(agent_id, 0.0),
                "harvest_scale": harvest_scale,
                "H_total": H_total,
                "below_target_zone": float(
                    S_t["fish"] / self.max_fish < self.mechanism.target_stock
                ),
                "target_shortfall": float(
                    max(0.0, self.mechanism.target_stock - (S_t["fish"] / self.max_fish))
                ),
            }
            for agent_id in self.agents
        }
        return obs, rewards, terminated, truncated, infos

    def _is_truncated(self) -> bool:
        return self._t >= self.horizon

    def desired_harvest_signal(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        action = float(np.asarray(action).item())
        fish_norm = S_t["fish"] / self.max_fish
        return action * fish_norm

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        action = float(np.asarray(action).item())
        fish_norm = S_t["fish"] / self.max_fish
        mechanism : FisheryMechanism = self.m or self.m_space.default()
        target_stock = max(EPS, mechanism.target_stock)

        sustainability_factor = min(1.0, fish_norm / target_stock)
        return action * fish_norm * sustainability_factor

    def violation_signal(
        self,
        agent_id: AgentID,
        u_i: SupportsFloat,
        S_t: dict[str, MultiAgentDict],
        A_t: dict[AgentID, ActType],
    ) -> dict[str, float]:
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish

        raw_harvest_signal = float(
            self.desired_harvest_signal(
                agent_id=agent_id,
                action=A_t[agent_id],
                S_t=S_t,
            )
        )

        quota = max(
            0.0,
            raw_harvest_signal - min(self.mechanism.fixed_quota, self.mechanism.prop_quota * fish_norm),
        )
        preventive = self._predictive_collapse_penalty(A_t=A_t, S_t=S_t)

        total = float(quota + preventive)
        return {
            "quota": float(quota),
            "preventive": float(preventive),
            "total": total,
        }

    def penalty(self) -> SupportsFloat:
        return self.mechanism.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        fish = S_t["fish"]
        algae = S_t["algae"]

        _, H, _ = self._compute_harvest_metrics(A_t, S_t)

        fish_next = fish + self.dt * (
            self.delta * algae * fish * (1 - fish / self.max_fish)
            - self.gamma * fish
            - H
        )
        algae_next = algae + self.dt * (
            self.alpha * algae * (1 - algae / self.max_algae) - self.beta * algae * fish
        )

        fish_next = np.clip(fish_next, 0.0, self.max_fish)
        algae_next = np.clip(algae_next, 0.0, self.max_algae)

        return {"fish": fish_next, "algae": algae_next}

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}

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
        self, A_t: dict[AgentID, ActType], S_t: dict[str, float]
    ) -> tuple[dict[AgentID, float], float, float]:
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish

        desired = {
            agent_id: float(
                self.desired_harvest_signal(
                    agent_id=agent_id,
                    action=A_t[agent_id],
                    S_t=S_t,
                )
            )
            for agent_id in self.agents
        }

        total_desired = float(sum(desired.values()))
        scale = min(1.0, fish_norm / max(EPS, total_desired))

        realized_harvest = {
            agent_id: self.max_fish * desired[agent_id] * scale
            for agent_id in self.agents
        }
        H_total = float(sum(realized_harvest.values()))

        return realized_harvest, H_total, scale

    def _predictive_collapse_penalty(
        self, A_t: dict[AgentID, ActType], S_t: dict[str, float]
    ) -> float:
        fish = S_t["fish"]
        algae = S_t["algae"]

        _, H_total, _ = self._compute_harvest_metrics(A_t, S_t)

        fish_next_pred = fish + self.dt * (
            self.delta * algae * fish - self.gamma * fish - H_total
        )

        safe_fish = self.mechanism.target_stock * self.max_fish

        current_shortage = max(0.0, safe_fish - fish)
        predicted_shortage = max(0.0, safe_fish - fish_next_pred)

        current_shortage_norm = current_shortage / self.max_fish
        predicted_shortage_norm = predicted_shortage / self.max_fish

        penalty_scale = self.mechanism.risk_penalty_scale
        penalty_power = self.mechanism.risk_penalty_power

        return float(
            penalty_scale
            * (
                current_shortage_norm**penalty_power
                + predicted_shortage_norm**penalty_power
            )
        )
