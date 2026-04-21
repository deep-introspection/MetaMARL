import logging
from typing import SupportsFloat, Tuple

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

        # observation map
        self.obs_map = [
            "fish_norm",
            "algae_norm",
            "ban_remaining",
            "effective_quota",
            "no_fish_zone",
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "ban_period",
            "catch_prob",
        ]

    def _reset(self):
        # Reset ban counters for all agents
        self._agent_bans = {agent_id: 0 for agent_id in self.agents}

        self.S_t = {
            "fish": max(EPS, self.rng.lognormal(np.log(self.fish_init), 0.05)),
            "algae": max(EPS, self.rng.lognormal(np.log(self.algae_init), 0.05)),
        }

        # logger.debug(
        #     "[RESET] fish=%.4f algae=%.4f",
        #     self.S_t["fish"],
        #     self.S_t["algae"],
        # )
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
        

        # DEBUG: force all actions to zero
        # effective_actions = {
        #     agent_id: np.array([0.0], dtype=np.float32)
        #     for agent_id in self.agents
        # }
        effective_actions = dict(action_dict)


        for agent_id in self.agents:
            # Check if agent is banned - zero out their action and apply penalty
            if hasattr(self, "_is_banned") and self._is_banned(agent_id):
                self._decrement_ban(agent_id)
                # DEBUG force actions to zero
                # effective_actions[agent_id] = np.array([0.0], dtype=np.float32)
                effective_actions[agent_id] = action_dict[agent_id] * 0
                # Mild ban penalty: just "time out", not heavy punishment
                rewards[agent_id] = -0.01
                fines[agent_id] = 0.0
                utilities[agent_id] = 0.0
                violations[agent_id] = 0.0
                continue

            # DEBUG: force actions to zero
            # u = float(self.intrinsic_utility(agent_id, effective_actions[agent_id], self.S_t))
            u = float(self.intrinsic_utility(agent_id, action_dict[agent_id], self.S_t))
            v = float(self.violation_signal(agent_id, u, self.S_t))

            utilities[agent_id] = u
            violations[agent_id] = v

            # Stochastic enforcement: only penalize if violation is detected
            catch_prob = getattr(self.m, "catch_prob", 1.0)
            if v > 0 and self.rng.random() < catch_prob:
                fine = float(self.penalty() * v)
                rewards[agent_id] = u - fine
                fines[agent_id] = fine
                # Apply ban if violation detected
                if hasattr(self, "_apply_ban"):
                    self._apply_ban(agent_id)
            else:
                rewards[agent_id] = u
                fines[agent_id] = 0.0

        rewards = self.aggregate_rewards(rewards)

        realized_harvest, H_total, harvest_scale = self._compute_harvest_metrics(
            effective_actions, self.S_t
        )

        # update obsevations
        prev_state = dict(self.S_t)
        self.S_t = self.transition_kernel(A_t=effective_actions, S_t=self.S_t)

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

        # check terminated and truncated conditions
        # terminated = natural end (e.g., goal reached or failure)
        # truncated = artificial time limit (horizon reached)
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
                "fine": fines.get(agent_id, 0.0),
                "harvest_scale": harvest_scale,
                "H_total": H_total, 
            } for agent_id in self.agents
        }
        return obs, rewards, terminated, truncated, infos


    def _is_truncated(self) -> bool:
        return self._t >= self.horizon

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        # return action * S_t["fish"]
        action = float(np.asarray(action).item())  # cast to scalar
        fish_norm = S_t["fish"] / self.max_fish
        u = action * fish_norm
        # logger.debug(
        #     "[UTILITY] %s action=%.4f fish_norm=%.4f u=%.6f",
        #     agent_id,
        #     action,
        #     fish_norm,
        #     u,
        # )
        return u

    # TODO this returns a float
    # TODO observation must be a param here not self
    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        quota = max(
            0.0,
            u_i
            - min(self.m.fixed_quota, self.m.prop_quota * S_t["fish"] / self.max_fish),
        )
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

        # TODO scale is not used
        _, H, _ = self._compute_harvest_metrics(A_t, S_t)

        # logger.debug(
        #     "[TRANSITION] fish=%.4f algae=%.4f fish_norm=%.4f "
        #     "total_desired=%.6f scale=%.4f H=%.6f",
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

        return np.array(
            [
                fish_norm,
                algae_norm,
                ban_remaining,
                effective_quota,
                no_fish_zone,
            ],
            dtype=np.float32,
        )

    # FISHERY SPECIFIC HELPERS
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

    def _compute_harvest_metrics(
        self, A_t: dict[AgentID, ActType], S_t: dict[str, float]
    ) -> tuple[dict[AgentID, float], float, float]:
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish

        desired = {
            agent_id: float(
                self.intrinsic_utility(
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
