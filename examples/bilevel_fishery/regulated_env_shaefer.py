import logging
from typing import SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

EPS = 1e-8


def pella_tomlinson_step(
    B: float,
    H: float,
    r: float,
    K: float,
    p: float = 1.0,
    noise: float = 0.0,
) -> tuple[float, float, float]:
    """
    Single-stock surplus production step.

    Returns:
        B_next
        H_realized
        growth
    """

    B = max(float(B), EPS)
    K = max(float(K), EPS)
    p = max(float(p), EPS)

    growth = (r / p) * B * (1.0 - (B / K) ** p) + noise
    available = max(B + growth, 0.0)

    H_realized = min(float(H), available)
    B_next = available - H_realized
    B_next = float(np.clip(B_next, 0.0, K))

    return B_next, H_realized, growth


def reference_points(r: float, K: float, p: float = 1.0) -> dict[str, float]:
    p = max(float(p), EPS)

    B_msy = K * (1.0 / (p + 1.0)) ** (1.0 / p)
    MSY = r * K / (p + 1.0) ** ((p + 1.0) / p)
    F_msy = MSY / max(B_msy, EPS)

    return {
        "B_msy": float(B_msy),
        "MSY": float(MSY),
        "F_msy": float(F_msy),
    }


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.r = ecology_cfg.get("r", 0.3)
        self.K = ecology_cfg.get("K", ecology_cfg.get("max_fish", 1000.0))
        self.p = ecology_cfg.get("p", 1.0)
        self.sigma = ecology_cfg.get("sigma", 0.05)

        self.max_fish = self.K
        self.fish_init = ecology_cfg.get("fish_init", ecology_cfg.get("B0", self.K))

        rp = reference_points(self.r, self.K, self.p)
        self.B_msy = rp["B_msy"]
        self.MSY = rp["MSY"]
        self.F_msy = rp["F_msy"]

        self.full_required_harvest = 0.0

        self.obs_map = [
            "fish_norm",
            "effective_quota",
            "total_usage_norm",
        ]

    @override(MultiAgentRegulatedEnv)
    def _reset(self):
        self.S_t = {
            "fish": max(
                EPS,
                self.rng.lognormal(np.log(max(self.fish_init, EPS)), 0.05),
            ),
            "last_usage": 0.0,
        }

        return {
            agent_id: self.observation(agent_id, self.S_t)
            for agent_id in self.agents
        }

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        return self.horizon is not None and (self._t + 1) >= self.horizon
    
    def _action_to_float(self, action: ActType) -> float:
        if hasattr(action, "item"):
            return float(action.item())
        return float(action)
    
    def _quota_stress(self, fish_norm: float) -> float:
        return float(
            np.clip(
                (fish_norm - self.mechanism.fixed_quota)
                / max(EPS, 1.0 - self.mechanism.fixed_quota),
                0.0,
                1.0,
            )
        )

    def _allowed_frac(self, fish_norm: float) -> float:
        stress = self._quota_stress(fish_norm)

        return float(
            self.mechanism.min_demand_frac
            + stress
            * (
                self.mechanism.max_demand_frac
                - self.mechanism.min_demand_frac
            )
        )

    def _allowed_harvest(self, fish_norm: float) -> float:
        return self._allowed_frac(fish_norm) * self.full_required_harvest
    
    def intrinsic_utility(
        self,
        A_t: dict[AgentID, ActType],
    ) -> MultiAgentDict:
        fish = float(self.S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        self.full_required_harvest = fish / len(self.agents)

        utilities = {}

        for agent_id, action in A_t.items():
            harvest_frac = np.clip(self._action_to_float(action), 0.0, 1.0)

            requested_harvest = harvest_frac * self.full_required_harvest
            allowed_harvest = self._allowed_harvest(fish_norm)
            delivered_harvest = min(requested_harvest, allowed_harvest)

            utilities[agent_id] = float(
                delivered_harvest / max(EPS, self.full_required_harvest)
            )

        self._update_infos(key="fish_norm", values=fish_norm)
        self._update_infos(key="full_required_harvest", values=self.full_required_harvest)

        return utilities
    
    def violation_signal(
        self,
        u_i: SupportsFloat,
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:
        fish = float(self.S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        requested_harvest = (
            np.clip(self._action_to_float(A_t[agent_id]), 0.0, 1.0)
            * self.full_required_harvest
        )

        allowed_frac = self._allowed_frac(fish_norm)
        allowed_harvest = allowed_frac * self.full_required_harvest

        delivered_harvest = min(requested_harvest, allowed_harvest)

        quota_violation = max(0.0, requested_harvest - allowed_harvest)

        quota_penalty = min(
            1.0,
            self.mechanism.fine_amount
            * quota_violation
            / max(EPS, self.full_required_harvest),
        )

        requested_frac_norm = requested_harvest / max(EPS, self.full_required_harvest)

        stock_pressure = max(0.0, 1.0 - fish_norm)

        risk_penalty = (
            self.mechanism.risk_penalty_scale
            * stock_pressure
            * (requested_frac_norm ** self.mechanism.risk_penalty_power)
        )

        total_penalty = min(1.0, quota_penalty + risk_penalty)

        self._update_infos(key="requested_harvest", values={agent_id: requested_harvest})
        self._update_infos(key="allowed_harvest", values={agent_id: allowed_harvest})
        self._update_infos(key="delivered_harvest", values={agent_id: delivered_harvest})
        self._update_infos(key="requested_frac", values={agent_id: requested_frac_norm})
        self._update_infos(key="quota_violation", values={agent_id: quota_violation})
        self._update_infos(key="quota_penalty", values={agent_id: quota_penalty})
        self._update_infos(key="risk_penalty", values={agent_id: risk_penalty})
        self._update_infos(key="quota_stress", values={agent_id: self._quota_stress(fish_norm)})
        self._update_infos(key="min_demand_frac", values={agent_id: self.mechanism.min_demand_frac})
        self._update_infos(key="max_demand_frac", values={agent_id: self.mechanism.max_demand_frac})

        return total_penalty

    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict,
    ) -> dict[str, float]:
        fish = float(S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        self.full_required_harvest = fish / len(self.agents)

        delivered_harvest = {}

        for agent_id, action in A_t.items():
            harvest_frac = np.clip(self._action_to_float(action), 0.0, 1.0)

            requested_harvest = harvest_frac * self.full_required_harvest
            allowed_harvest = self._allowed_harvest(fish_norm)

            delivered_harvest[agent_id] = min(
                requested_harvest,
                allowed_harvest,
            )

        H_attempted = float(sum(delivered_harvest.values()))

        noise = self.sigma * self.rng.normal() * fish

        fish_next, H_realized, growth = pella_tomlinson_step(
            B=fish,
            H=H_attempted,
            r=self.r,
            K=self.K,
            p=self.p,
            noise=noise,
        )

        new_state = {
            "fish": fish_next,
            "last_usage": H_realized,
        }

        self._update_infos(key="fish", values=fish)
        self._update_infos(key="fish_next", values=fish_next)
        self._update_infos(key="fish_norm", values=fish_norm)
        self._update_infos(key="fish_norm_next", values=fish_next / max(self.max_fish, EPS))
        self._update_infos(key="growth", values=growth)
        self._update_infos(key="growth_noise", values=noise)
        self._update_infos(key="H_attempted", values=H_attempted)
        self._update_infos(key="H_realized", values=H_realized)
        self._update_infos(key="total_usage_norm", values=H_realized / max(EPS, self.max_fish))
        self._update_infos(key="B_msy", values=self.B_msy)
        self._update_infos(key="MSY", values=self.MSY)
        self._update_infos(key="F_msy", values=self.F_msy)

        self.S_t = new_state
        return self.S_t
    
    def _observation(
        self,
        agent_id: AgentID,
        S_t: dict,
    ):
        fish_norm = float(S_t["fish"] / max(self.max_fish, EPS))
        effective_quota = self._allowed_frac(fish_norm)
        total_usage_norm = float(S_t.get("last_usage", 0.0) / max(EPS, self.max_fish))

        return np.array(
            [
                fish_norm,
                effective_quota,
                total_usage_norm,
            ],
            dtype=np.float32,
        )

    def _step(
        self,
        action_dict: dict[AgentID, ActType],
    ) -> Tuple[
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
    ]:
        S_t = self.S_t.copy()
        fish = float(S_t["fish"])
        fish_norm = fish / max(self.max_fish, EPS)

        # Same parent-style logic as water:
        # utility first, violation second, then transition
        utilities = self.intrinsic_utility(action_dict)

        violations = {
            agent_id: self.violation_signal(
                utilities[agent_id],
                agent_id,
                A_t=action_dict,
            )
            for agent_id in self.agents
        }

        rewards = {
            agent_id: float(utilities[agent_id] - violations[agent_id])
            for agent_id in self.agents
        }

        rewards = self._aggregate_rewards(rewards)

        S_next = self.transition_kernel(
            A_t=action_dict,
            S_t=S_t,
        )

        obs = {
            agent_id: self.observation(agent_id, S_next)
            for agent_id in self.agents
        }

        time_limit = self._is_truncated()

        terminated = {agent_id: False for agent_id in self.agents}
        terminated["__all__"] = False

        truncated = {agent_id: time_limit for agent_id in self.agents}
        truncated["__all__"] = time_limit

        infos = {
            agent_id: {
                "fish": fish,
                "fish_next": S_next["fish"],
                "fish_norm": fish_norm,
                "fish_norm_next": S_next["fish"] / max(self.max_fish, EPS),
                "intrinsic_utility": utilities[agent_id],
                "violation_signal": violations[agent_id],
                "B_msy": self.B_msy,
                "MSY": self.MSY,
                "F_msy": self.F_msy,
            }
            for agent_id in self.agents
        }

        return obs, rewards, terminated, truncated, infos