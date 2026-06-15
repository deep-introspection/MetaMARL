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

        self.obs_map = [
            "fish_norm",
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
            "fish": max(
                EPS,
                self.rng.lognormal(np.log(max(self.fish_init, EPS)), 0.05),
            )
        }

        return {
            agent_id: self.observation(agent_id, self.S_t)
            for agent_id in self.agents
        }

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        return self.horizon is not None and (self._t + 1) >= self.horizon

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

        desired_harvest, H_attempted = self._compute_attempted_harvest(
            A_t=action_dict,
            S_t=S_t,
        )

        noise = self.sigma * self.rng.normal() * fish

        fish_next, H_realized, growth = pella_tomlinson_step(
            B=fish,
            H=H_attempted,
            r=self.r,
            K=self.K,
            p=self.p,
            noise=noise,
        )

        harvest_scale = H_realized / max(EPS, H_attempted)

        realized_harvest = {
            agent_id: desired_harvest.get(agent_id, 0.0) * harvest_scale
            for agent_id in self.agents
        }

        utilities = {
            agent_id: realized_harvest[agent_id] / max(self.max_fish, EPS)
            for agent_id in self.agents
        }

        violations = {}
        quota_violations = {}
        quota_penalties = {}
        stock_penalties = {}

        for agent_id in self.agents:
            u_i = self._desired_harvest_signal(
                agent_id=agent_id,
                A_t=action_dict,
                S_t=S_t,
            )

            fish_norm_current = self.S_t["fish"] / max(self.max_fish, EPS)

            effective_quota = min(
                self.mechanism.fixed_quota,
                self.mechanism.prop_quota * fish_norm_current,
            )

            quota_violation = max(0.0, float(u_i) - effective_quota)

            quota_penalty = min(
                1.0,
                self.mechanism.fine_amount * quota_violation,
            )

            shortage_severity = max(0.0, self.mechanism.min_stock - fish_norm_current)

            stock_penalty = min(
                1.0,
                self.mechanism.risk_penalty_scale
                * (shortage_severity ** self.mechanism.risk_penalty_power)
                * float(u_i > 0.0),
            )

            total_penalty = min(1.0, quota_penalty + stock_penalty)

            quota_violations[agent_id] = quota_violation
            quota_penalties[agent_id] = quota_penalty
            stock_penalties[agent_id] = stock_penalty
            violations[agent_id] = total_penalty

        rewards = {
            agent_id: utilities[agent_id] - violations[agent_id]
            for agent_id in self.agents
        }

        rewards = self._aggregate_rewards(rewards)

        self.S_t = {"fish": fish_next}

        obs = {
            agent_id: self.observation(agent_id, self.S_t)
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
                "fish_next": fish_next,
                "fish_norm": fish_norm,
                "growth": growth,
                "growth_noise": noise,
                "harvest": realized_harvest[agent_id],
                "desired_harvest": desired_harvest[agent_id],
                "intrinsic_utility": utilities[agent_id],
                "violation_signal": violations[agent_id],
                "H_attempted": H_attempted,
                "H_realized": H_realized,
                "harvest_scale": harvest_scale,
                "below_target_zone": float(fish_norm < self.mechanism.target_stock),
                "target_shortfall": max(
                    0.0,
                    self.mechanism.target_stock - fish_norm,
                ),
                "B_msy": self.B_msy,
                "MSY": self.MSY,
                "F_msy": self.F_msy,
                "quota_violation": quota_violations[agent_id],
                "quota_penalty": quota_penalties[agent_id],
                "stock_penalty": stock_penalties[agent_id],
            }
            for agent_id in self.agents
        }

        return obs, rewards, terminated, truncated, infos

    def intrinsic_utility(
        self,
        A_t: dict[AgentID, ActType],
    ) -> MultiAgentDict:
        desired_harvest, H_attempted = self._compute_attempted_harvest(
            A_t=A_t,
            S_t=self.S_t,
        )

        scale = min(1.0, float(self.S_t["fish"]) / max(EPS, H_attempted))

        return {
            agent_id: desired_harvest[agent_id] * scale / max(self.max_fish, EPS)
            for agent_id in self.agents
        }

    def violation_signal(
        self,
        u_i: SupportsFloat,
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:
        fish_norm = self.S_t["fish"] / max(self.max_fish, EPS)

        effective_quota = min(
            self.mechanism.fixed_quota,
            self.mechanism.prop_quota * fish_norm,
        )

        quota_violation = max(0.0, float(u_i) - effective_quota)

        quota_penalty = min(
            1.0,
            self.mechanism.fine_amount * quota_violation,
        )

        shortage_severity = max(0.0, self.mechanism.min_stock - fish_norm)

        stock_penalty = min(
            1.0,
            self.mechanism.risk_penalty_scale
            * (shortage_severity ** self.mechanism.risk_penalty_power)
            * float(u_i > 0.0),
        )

        total_penalty = min(1.0, quota_penalty + stock_penalty)

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

    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict,
    ) -> dict[str, float]:
        desired_harvest, H_attempted = self._compute_attempted_harvest(
            A_t=A_t,
            S_t=S_t,
        )

        fish = float(S_t["fish"])
        noise = self.sigma * self.rng.normal() * fish

        fish_next, _, _ = pella_tomlinson_step(
            B=fish,
            H=H_attempted,
            r=self.r,
            K=self.K,
            p=self.p,
            noise=noise,
        )

        return {"fish": fish_next}

    def _observation(
        self,
        agent_id: AgentID,
        S_t: dict,
    ):
        fish_norm = S_t["fish"] / max(self.max_fish, EPS)

        effective_quota = min(
            self.mechanism.fixed_quota,
            self.mechanism.prop_quota * fish_norm,
        )

        no_fish_zone = float(fish_norm < self.mechanism.min_stock)

        return np.array(
            [
                fish_norm,
                effective_quota,
                no_fish_zone,
            ],
            dtype=np.float32,
        )

    def _desired_harvest_signal(
        self,
        agent_id: AgentID,
        A_t: dict[AgentID, ActType],
        S_t: dict,
    ) -> float:
        fish = float(S_t["fish"])
        action = max(0.0, float(np.asarray(A_t[agent_id]).item()))

        desired_biomass = action * fish

        return desired_biomass / max(self.max_fish, EPS)

    def _compute_attempted_harvest(
        self,
        A_t: dict[AgentID, ActType],
        S_t: dict,
    ) -> tuple[dict[AgentID, float], float]:
        fish = float(S_t["fish"])

        desired = {
            agent_id: max(0.0, float(np.asarray(A_t[agent_id]).item())) * fish
            for agent_id in self.agents
        }

        H_attempted = float(sum(desired.values()))

        return desired, H_attempted

    def _compute_harvest_metrics(
        self,
        A_t: dict[AgentID, ActType],
    ) -> tuple[dict[AgentID, float], float, float]:
        desired_harvest, H_attempted = self._compute_attempted_harvest(
            A_t=A_t,
            S_t=self.S_t,
        )

        fish = float(self.S_t["fish"])
        available = max(fish, 0.0)

        H_realized = min(H_attempted, available)
        harvest_scale = H_realized / max(EPS, H_attempted)

        realized_harvest = {
            agent_id: desired_harvest.get(agent_id, 0.0) * harvest_scale
            for agent_id in self.agents
        }

        return realized_harvest, H_attempted, harvest_scale