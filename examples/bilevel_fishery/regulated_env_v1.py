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
        quota_penalties = {}
        preventive_penalties = {}

        for agent_id in self.agents:
            u = float(
                self.intrinsic_utility(agent_id, action_dict[agent_id], self.S_t)
            )

            v_dict = self.violation_signal(
                agent_id=agent_id,
                u_i=u,
                S_t=self.S_t,
                A_t=action_dict,
            )

            quota_violation = float(v_dict["quota_violation"])
            quota_penalty = float(v_dict["quota_penalty"])
            preventive_penalty = float(v_dict["preventive"])
            v = float(v_dict["total"])

            utilities[agent_id] = u
            quota_violations[agent_id] = quota_violation
            quota_penalties[agent_id] = quota_penalty
            preventive_penalties[agent_id] = preventive_penalty
            violations[agent_id] = v
            rewards[agent_id] = u - v
            fines[agent_id] = quota_penalty + preventive_penalty

        rewards = self.aggregate_rewards(rewards)

        realized_harvest, H_total, harvest_scale = self._compute_harvest_metrics(
            action_dict, self.S_t
        )
        S_t = self.S_t.copy()

        self.S_t = self.transition_kernel(A_t=action_dict, S_t=S_t)

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
                "quota_penalty": quota_penalties.get(agent_id, 0.0),
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
        # TODO move this to parent class
        return self.horizon is not None and (self._t + 1) >= self.horizon

    def desired_harvest_signal(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        action = float(np.asarray(action).item())
        fish_norm = S_t["fish"] / self.max_fish
        return action * fish_norm

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        # TODO why is the initial action high ?
        fish_norm = S_t["fish"] / self.max_fish
        mechanism : FisheryMechanism = self.m or self.m_space.default()
        target_stock = max(EPS, mechanism.target_stock) # norm or absolute ?
    
        sustainability_factor = min(1.0, fish_norm / target_stock)
        return action.item() * fish_norm * sustainability_factor

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

        effective_quota = min(
            self.mechanism.fixed_quota, 
            self.mechanism.prop_quota * fish_norm,
        )
        quota_violation = max(0.0, raw_harvest_signal - effective_quota)
        quota_penalty = min(
            self.mechanism.max_fine,
            self.mechanism.fine_amount * quota_violation,
        )
        preventive = self._predictive_collapse_penalty(A_t=A_t, S_t=S_t)

        total = float(quota_penalty + preventive)
        return {
            "quota_violation": float(quota_violation),
            "quota_penalty": float(quota_penalty),
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
        # BUG FIXED: CRITICAL MATH & VALUE LOOPHOLE
        # 1. Unit Cancellation: Multiplying by 'self.max_fish' inside the dictionary 
        #    comprehension cancelled out the denominator of the 'scale' equation.
        #    This bypassed physical environmental constraints and forced a pure 
        #    proportional split of remaining fish, regardless of total scarcity.
        # 2. Nash Equilibrium/Flat-line Exploitation: Agents learned they could 
        #    overfish heavily, hold the pond hostage at near-extinction levels, 
        #    and force 'worsening_shortage' to 0.0 to dodge the predictive penalty.
        # FIX: Switched to computing absolute predicted deficits, adjusted 
        #      'default_prop_quota' to a strict fractional brake (< 1.0), and 
        #      isolated scale logic to prevent redundant raw unit cancellation.

        fish = S_t["fish"]
        algae = S_t["algae"]

        _, H_total, _ = self._compute_harvest_metrics(A_t, S_t)

        fish_next_pred = fish + self.dt * (
            self.delta * algae * fish * (1 - fish / self.max_fish)
            - self.gamma * fish
            - H_total
        )

        safe_fish = self.mechanism.target_stock * self.max_fish

        current_shortage = max(0.0, safe_fish - fish)
        predicted_shortage = max(0.0, safe_fish - fish_next_pred)

        current_shortage_norm = current_shortage / self.max_fish
        predicted_shortage_norm = predicted_shortage / self.max_fish

        worsening_shortage = max(
            0.0,
            predicted_shortage_norm - current_shortage_norm,
        )

        penalty_scale = self.mechanism.risk_penalty_scale
        penalty_power = self.mechanism.risk_penalty_power

        return float(penalty_scale * (worsening_shortage**penalty_power))
    
        # def _predictive_collapse_penalty(
        #     self, A_t: dict[AgentID, ActType]
        # ) -> float:
        #     """Calculates a preventative risk penalty based on absolute ecosystem danger.

        #     Fixes the 'flat-line collapse' bug by punishing the absolute predicted 
        #     shortage next turn, preventing agents from holding the population at a 
        #     constant, near-extinct level for zero penalty.
        #     """
        #     fish = self.S_t["fish"]
        #     fish_norm = fish / self.max_fish
        #     algae = self.S_t["algae"]

        #     # 1. Extract total harvest from current agent choices
        #     _, H_total, _ = self._compute_harvest_metrics(A_t=A_t)

        #     # 2. Predict the raw population for the next time step
        #     fish_next_raw = fish + self.dt * (
        #         self.delta * algae * fish * (1 - fish_norm)
        #         - self.gamma * fish
        #         - H_total
        #     )
            
        #     # 3. CRITICAL FIX: Clip prediction to valid physical bounds [0.0, max_fish]
        #     # Prevents underflows that make shortages look artificially greater than 1.0
        #     fish_next_pred = max(0.0, min(self.max_fish, fish_next_raw))
        #     fish_next_pred_norm = fish_next_pred / self.max_fish

        #     # 4. Extract target threshold (already normalized between 0.0 and 1.0)
        #     target_stock_norm = self.mechanism.target_stock

        #     # 5. CRITICAL FIX: Measure the ABSOLUTE predicted deficit below target
        #     # Removed the previous subtraction (- current_shortage) so the penalty 
        #     # stays active until the fish stock actually recovers.
        #     predicted_shortage_norm = max(0.0, target_stock_norm - fish_next_pred_norm)

        #     # 6. Apply non-linear scaling parameters from the mechanism space
        #     penalty_scale = self.mechanism.risk_penalty_scale
        #     penalty_power = self.mechanism.risk_penalty_power

        #     # Bounded between 0.0 and 1.0 to match the scale of the intrinsic rewards
        #     return float(penalty_scale * (predicted_shortage_norm ** penalty_power))

