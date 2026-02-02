"""Multi-agent fishery environment with Lotka-Volterra ecosystem dynamics.

This module contains the fishery environment implementation for the bilevel
optimization experiment. The environment simulates shared fishery resources
with regulatory mechanisms including quotas, fines, and temporary bans.

Optional social extensions (enabled via config):
- Multiple commons (resource pools) with per-commons governance groups.
- Reputation as an observable state, updated by peer votes and behavior.
- Semi-private commons with access controlled by group vote.
- Gaussian noise for actions (policy mean -> sampled action) and parameters.
"""

from typing import Dict, Optional, Tuple, List, Iterable

import numpy as np
from config import (
    ACTION_BOUNDS,
    DEFAULT_ECOLOGY_CONFIG,
    DEFAULT_MECHANISM_CONFIG,
    EPS,
    OBSERVATION_BOUNDS,
)
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

GOVERNANCE_PARAM_NAMES = (
    "fixed_quota",
    "prop_quota",
    "min_stock",
    "fine_amount",
    "ban_period",
    "ban_success_prob",
    "fine_success_prob",
    "access_threshold",
)

MECHANISM_RANGES = {
    "fixed_quota": (0.0, 1.0),
    "prop_quota": (0.0, 1.0),
    "min_stock": (0.0, 1.0),
    "fine_amount": (0.0, 2.0),
    "ban_period": (0.0, 10.0),
    "ban_success_prob": (0.0, 1.0),
    "fine_success_prob": (0.0, 1.0),
}

ACCESS_MODES = {"public", "members", "voted"}
ACCESS_SCOPES = {"all", "members"}


class FisheryEnvFixed(MultiAgentEnv):
    """Multi-agent fishery environment with Lotka-Volterra ecosystem dynamics.

    This environment simulates a fishery where multiple fishermen harvest from
    shared resources governed by algae-fish population dynamics. The regulatory
    mechanism includes quotas, fines, and temporary bans. Optional social
    features add reputation, governance, and semi-private commons.

    Base State Space (per agent):
        - Algae population (continuous)
        - Fish population (continuous)

    Base Action Space (per agent):
        - Desired harvest fraction [0,1]

    Mechanism Parameters (per commons):
        - fixed_quota: Absolute harvest limit
        - prop_quota: Proportional quota factor
        - min_stock: Minimum fish stock threshold
        - fine_amount: Penalty per unit over-harvest
        - ban_period: Duration of ban after violation
    """

    def __init__(self, env_config: Optional[Dict] = None) -> None:
        super().__init__()

        config = env_config or {}

        # Ecological parameters (Lotka-Volterra dynamics)
        ecology_config = {**DEFAULT_ECOLOGY_CONFIG, **config}
        self.alpha_mean = float(ecology_config["alpha"])  # Algae growth rate
        self.beta_mean = float(ecology_config["beta"])  # Predation rate
        self.delta_mean = float(ecology_config["delta"])  # Fish growth efficiency
        self.gamma_mean = float(ecology_config["gamma"])  # Fish death rate
        self.dt = float(ecology_config["dt"])  # Integration time step
        self.horizon = int(ecology_config["horizon"])  # Episode length
        self.algae_init = float(ecology_config["algae_init"])
        self.fish_init = float(ecology_config["fish_init"])

        # Regulatory mechanism parameters (defaults)
        mechanism_config = {**DEFAULT_MECHANISM_CONFIG, **config}
        self.fixed_quota = float(mechanism_config["fixed_quota"])
        self.prop_quota = float(mechanism_config["prop_quota"])
        self.min_stock = float(mechanism_config["min_stock"])
        self.fine_amount = float(mechanism_config["fine_amount"])
        self.ban_period = int(mechanism_config["ban_period"])
        self.ban_success_prob = float(mechanism_config["ban_success_prob"])
        self.fine_success_prob = float(mechanism_config["fine_success_prob"])
        self._ban_period_max = int(config.get("ban_period_max", 10))

        self._base_mechanism_defaults = {
            "fixed_quota": self.fixed_quota,
            "prop_quota": self.prop_quota,
            "min_stock": self.min_stock,
            "fine_amount": self.fine_amount,
            "ban_period": float(self.ban_period),
            "ban_success_prob": self.ban_success_prob,
            "fine_success_prob": self.fine_success_prob,
        }

        # Agent setup
        self.num_fishermen = int(config.get("num_fishermen", 3))
        self.fishermen = [f"fisherman_{i}" for i in range(self.num_fishermen)]
        self._agent_index = {aid: idx for idx, aid in enumerate(self.fishermen)}
        self.possible_agents = list(self.fishermen)

        # Social/governance config
        self.num_commons = int(config.get("num_commons", 1))
        self.enable_reputation = bool(config.get("enable_reputation", False))
        self.enable_governance = bool(config.get("enable_governance", False))
        self.governance_strength = float(config.get("governance_strength", 1.0))
        self.reputation_init = float(config.get("reputation_init", 0.5))
        self.reputation_min = float(config.get("reputation_min", 0.0))
        self.reputation_max = float(config.get("reputation_max", 1.0))
        self.reputation_vote_weight = float(config.get("reputation_vote_weight", 0.05))
        self.reputation_vote_decay = float(config.get("reputation_vote_decay", 0.1))
        self.reputation_vote_weight_by_reputation = bool(
            config.get("reputation_vote_weight_by_reputation", False)
        )
        self.reputation_behavior_weight = float(
            config.get("reputation_behavior_weight", 0.1)
        )
        self.access_threshold_init = float(config.get("access_threshold_init", 0.5))
        self.access_mode_default = str(config.get("access_mode", "public"))
        self.access_scope_default = str(config.get("access_scope", "all"))

        # Observation composition
        self.include_quota_in_obs = bool(
            config.get(
                "include_quota_in_obs", self.enable_governance or self.num_commons > 1
            )
        )
        self.include_access_in_obs = bool(
            config.get(
                "include_access_in_obs",
                self.enable_governance or self.num_commons > 1,
            )
        )
        self.include_reputation_in_obs = bool(
            config.get("include_reputation_in_obs", self.enable_reputation)
        )

        # Private fishery configuration (reputation-allocated resource)
        self.enable_private_fishery = bool(config.get("enable_private_fishery", False))
        self.private_availability = float(config.get("private_availability", 0.05))
        self.private_min_stock = float(config.get("private_min_stock", self.min_stock))
        self.private_allocation_mode = str(
            config.get("private_allocation_mode", "reputation")
        )
        self.private_algae_init = float(
            config.get("private_algae_init", self.algae_init)
        )
        self.private_fish_init = float(config.get("private_fish_init", self.fish_init))
        self.include_private_in_obs = bool(
            config.get("include_private_in_obs", False)
        )

        # Noise configuration
        self.action_noise_std_config = config.get("action_noise_std", 0.0)
        self.action_noise_clip = float(config.get("action_noise_clip", 3.0))
        self.mechanism_noise_std = self._parse_noise_config(
            config.get("mechanism_noise_std", 0.0),
            list(MECHANISM_RANGES.keys()),
            default=0.0,
        )
        self.mechanism_noise_mode = str(
            config.get("mechanism_noise_mode", "episode")
        ).lower()
        self.ecology_noise_std = self._parse_noise_config(
            config.get("ecology_noise_std", 0.0),
            ["alpha", "beta", "delta", "gamma"],
            default=0.0,
        )
        self.observation_noise_std = float(config.get("observation_noise_std", 0.0))

        # Commons configuration
        self._commons = self._normalize_commons_config(config.get("commons"))
        self._base_mechanism_params = self._build_base_mechanism_params()

        # Define observation and action spaces
        self._build_spaces()
        self._action_noise_std = self._parse_action_noise(self.action_noise_std_config)

        # Environment state
        self._time_step = 0
        self._rng = np.random.default_rng()
        self._algae_population = np.zeros(self.num_commons, dtype=np.float32)
        self._fish_population = np.zeros(self.num_commons, dtype=np.float32)
        self._alpha = np.full(self.num_commons, self.alpha_mean, dtype=np.float32)
        self._beta = np.full(self.num_commons, self.beta_mean, dtype=np.float32)
        self._delta = np.full(self.num_commons, self.delta_mean, dtype=np.float32)
        self._gamma = np.full(self.num_commons, self.gamma_mean, dtype=np.float32)
        self._private_algae_population = float(self.private_algae_init)
        self._private_fish_population = float(self.private_fish_init)
        self._private_alpha = float(self.alpha_mean)
        self._private_beta = float(self.beta_mean)
        self._private_delta = float(self.delta_mean)
        self._private_gamma = float(self.gamma_mean)
        self._agent_bans: Dict[str, np.ndarray] = {
            aid: np.zeros(self.num_commons, dtype=np.int32) for aid in self.fishermen
        }
        self._reputation = np.full(
            self.num_fishermen, self.reputation_init, dtype=np.float32
        )
        self._access_thresholds_base = np.full(
            self.num_commons, self.access_threshold_init, dtype=np.float32
        )
        self._access_thresholds = self._access_thresholds_base.copy()
        self._access_mask = np.ones((self.num_fishermen, self.num_commons), dtype=bool)
        self._episode_mechanism_base = {
            name: self._base_mechanism_params[name].copy()
            for name in MECHANISM_RANGES
        }
        self._mechanism_params = {
            name: self._base_mechanism_params[name].copy()
            for name in MECHANISM_RANGES
        }
        self._last_quota_limits = np.zeros(self.num_commons, dtype=np.float32)

    def observation_space(self, agent_id: str) -> spaces.Box:
        """Get observation space for a specific agent."""
        return self._obs_space

    def action_space(self, agent_id: str) -> spaces.Box:
        """Get action space for a specific agent."""
        return self._act_space

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            Tuple of (observations, infos) dictionaries
        """
        self._time_step = 0
        self._rng = np.random.default_rng(seed)

        self._sample_ecology_parameters()

        # Initialize populations with small random variation
        algae_mean = np.log(max(self.algae_init, 1e-3))
        fish_mean = np.log(max(self.fish_init, 1e-3))

        self._algae_population = np.maximum(
            EPS,
            self._rng.lognormal(mean=algae_mean, sigma=0.05, size=self.num_commons),
        ).astype(np.float32)
        self._fish_population = np.maximum(
            EPS,
            self._rng.lognormal(mean=fish_mean, sigma=0.05, size=self.num_commons),
        ).astype(np.float32)

        private_algae_mean = np.log(max(self.private_algae_init, 1e-3))
        private_fish_mean = np.log(max(self.private_fish_init, 1e-3))
        self._private_algae_population = float(
            max(EPS, self._rng.lognormal(mean=private_algae_mean, sigma=0.05))
        )
        self._private_fish_population = float(
            max(EPS, self._rng.lognormal(mean=private_fish_mean, sigma=0.05))
        )

        # Reset all agent bans
        for agent_id in self.fishermen:
            self._agent_bans[agent_id].fill(0)

        # Reset reputations
        self._reputation = np.full(
            self.num_fishermen, self.reputation_init, dtype=np.float32
        )

        # Sample mechanism parameters for this episode
        self._episode_mechanism_base = {
            name: self._base_mechanism_params[name].copy()
            for name in MECHANISM_RANGES
        }
        if self.mechanism_noise_mode == "episode":
            self._apply_mechanism_noise(self._episode_mechanism_base)

        self._mechanism_params = {
            name: self._episode_mechanism_base[name].copy()
            for name in MECHANISM_RANGES
        }

        # Reset access thresholds
        self._access_thresholds_base = np.array(
            [commons["access_threshold_init"] for commons in self._commons],
            dtype=np.float32,
        )
        self._access_thresholds = self._access_thresholds_base.copy()

        # Initialize access and quotas for first observations
        self._access_mask = self._compute_access_mask()
        self._last_quota_limits = self._compute_quota_limits()

        observations = {
            agent_id: self._build_observation(self._agent_index[agent_id])
            for agent_id in self.fishermen
        }
        infos = {agent_id: {} for agent_id in self.fishermen}

        return observations, infos

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminated
        Dict[str, bool],  # truncated
        Dict[str, Dict],  # infos
    ]:
        """Execute one environment step.

        Args:
            action_dict: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        agent_actions = self._parse_actions(action_dict)

        # Update mechanism parameters based on governance votes
        self._apply_governance_votes(agent_actions)

        # Compute access mask for this step (based on current reputations)
        access_mask_current = self._compute_access_mask()

        # Compute quota limits for this step
        quota_limits = self._compute_quota_limits()

        agent_catches: Dict[str, np.ndarray] = {
            aid: np.zeros(self.num_commons, dtype=np.float32) for aid in self.fishermen
        }
        agent_rewards: Dict[str, float] = {aid: 0.0 for aid in self.fishermen}
        overharvest_totals = np.zeros(self.num_fishermen, dtype=np.float32)
        quota_totals = np.zeros(self.num_fishermen, dtype=np.float32)

        # Process each commons separately
        for commons_idx in range(self.num_commons):
            fishing_allowed = (
                self._fish_population[commons_idx]
                >= self._mechanism_params["min_stock"][commons_idx]
            )

            if fishing_allowed:
                quota_limit = float(quota_limits[commons_idx])
                for agent_id in self.fishermen:
                    agent_idx = self._agent_index[agent_id]
                    desired_fraction = agent_actions[agent_id]["harvest"][commons_idx]
                    access_allowed = bool(access_mask_current[agent_idx, commons_idx])
                    catch, reward, over_harvest = self._process_agent_action(
                        agent_id,
                        commons_idx,
                        desired_fraction,
                        quota_limit,
                        access_allowed,
                    )
                    agent_catches[agent_id][commons_idx] = catch
                    agent_rewards[agent_id] += reward
                    overharvest_totals[agent_idx] += over_harvest
                    if access_allowed:
                        quota_totals[agent_idx] += quota_limit
            else:
                for agent_id in self.fishermen:
                    agent_catches[agent_id][commons_idx] = 0.0

        # Calculate total harvest per commons and update ecosystem
        total_harvest = np.zeros(self.num_commons, dtype=np.float32)
        for commons_idx in range(self.num_commons):
            total_harvest[commons_idx] = float(
                sum(agent_catches[aid][commons_idx] for aid in self.fishermen)
            )

        private_catches = {aid: 0.0 for aid in self.fishermen}
        private_total_harvest = 0.0
        if self.enable_private_fishery:
            private_catches, private_total_harvest = self._allocate_private_fishery()
            for agent_id in self.fishermen:
                agent_rewards[agent_id] += float(private_catches[agent_id])

        self._update_ecosystem(total_harvest)
        if self.enable_private_fishery:
            self._update_private_ecosystem(private_total_harvest)
        self._time_step += 1

        # Update reputations after observing outcomes
        if self.enable_reputation:
            self._update_reputation(agent_actions, overharvest_totals, quota_totals)

        # Recompute access and quotas for next step observations
        self._access_mask = self._compute_access_mask()
        self._last_quota_limits = self._compute_quota_limits()

        observations = {
            agent_id: self._build_observation(self._agent_index[agent_id])
            for agent_id in self.fishermen
        }

        # Check episode termination conditions
        is_terminated = {agent_id: False for agent_id in self.fishermen}
        is_truncated = {
            agent_id: self._time_step >= self.horizon for agent_id in self.fishermen
        }
        is_terminated["__all__"] = any(is_terminated.values())
        is_truncated["__all__"] = any(is_truncated.values())

        # Create info dictionaries
        infos = {
            agent_id: {
                "algae": float(np.sum(self._algae_population)),
                "fish": float(np.sum(self._fish_population)),
                "catch": float(np.sum(agent_catches[agent_id])),
                "ban": int(np.max(self._agent_bans[agent_id])),
                "quota_limit": float(np.sum(self._last_quota_limits)),
                "private_algae": float(self._private_algae_population),
                "private_fish": float(self._private_fish_population),
                "private_catch": float(private_catches[agent_id]),
                "algae_by_common": self._algae_population.tolist(),
                "fish_by_common": self._fish_population.tolist(),
                "catch_by_common": agent_catches[agent_id].tolist(),
                "ban_by_common": self._agent_bans[agent_id].tolist(),
                "quota_limit_by_common": self._last_quota_limits.tolist(),
                "access_mask": self._access_mask[
                    self._agent_index[agent_id]
                ].astype(np.float32).tolist(),
                "reputation": float(self._reputation[self._agent_index[agent_id]]),
            }
            for agent_id in self.fishermen
        }

        return observations, agent_rewards, is_terminated, is_truncated, infos

    def _process_agent_action(
        self,
        agent_id: str,
        commons_idx: int,
        desired_fraction: float,
        quota_limit: float,
        access_allowed: bool,
    ) -> Tuple[float, float, float]:
        """Process a single agent's fishing action for one commons."""
        if self._agent_bans[agent_id][commons_idx] > 0:
            self._agent_bans[agent_id][commons_idx] -= 1
            return 0.0, 0.0, 0.0

        if not access_allowed:
            return 0.0, 0.0, 0.0

        desired_fraction = float(np.clip(desired_fraction, 0.0, 1.0))
        desired_harvest = desired_fraction * self._fish_population[commons_idx]
        over_harvest = max(0.0, desired_harvest - quota_limit)
        actual_catch = min(desired_harvest, self._fish_population[commons_idx])

        fine_amount = float(self._mechanism_params["fine_amount"][commons_idx])
        fine_success_prob = float(
            np.clip(self._mechanism_params["fine_success_prob"][commons_idx], 0.0, 1.0)
        )
        fine_applied = over_harvest > EPS and fine_amount > 0.0
        if fine_applied and self._rng.random() >= fine_success_prob:
            fine_applied = False
        reward = actual_catch - fine_amount * over_harvest if fine_applied else actual_catch

        ban_period = int(
            np.clip(
                np.round(self._mechanism_params["ban_period"][commons_idx]),
                0,
                self._ban_period_max,
            )
        )
        if over_harvest > EPS and ban_period > 0:
            ban_success_prob = float(
                np.clip(
                    self._mechanism_params["ban_success_prob"][commons_idx], 0.0, 1.0
                )
            )
            if self._rng.random() < ban_success_prob:
                self._agent_bans[agent_id][commons_idx] = ban_period

        return float(actual_catch), float(reward), float(over_harvest)

    def _update_ecosystem(self, total_harvest: np.ndarray) -> None:
        """Update ecosystem populations using Lotka-Volterra dynamics."""
        algae = self._algae_population
        fish = self._fish_population

        d_algae_dt = self._alpha * algae - self._beta * algae * fish
        d_fish_dt = self._delta * algae * fish - self._gamma * fish - total_harvest

        self._algae_population = np.maximum(0.0, algae + self.dt * d_algae_dt).astype(
            np.float32
        )
        self._fish_population = np.maximum(0.0, fish + self.dt * d_fish_dt).astype(
            np.float32
        )

    def _update_private_ecosystem(self, total_harvest: float) -> None:
        """Update the private fishery using Lotka-Volterra dynamics."""
        algae = self._private_algae_population
        fish = self._private_fish_population

        d_algae_dt = self._private_alpha * algae - self._private_beta * algae * fish
        d_fish_dt = self._private_delta * algae * fish - self._private_gamma * fish
        d_fish_dt -= total_harvest

        self._private_algae_population = float(max(0.0, algae + self.dt * d_algae_dt))
        self._private_fish_population = float(max(0.0, fish + self.dt * d_fish_dt))

    def _allocate_private_fishery(self) -> Tuple[Dict[str, float], float]:
        """Allocate private fishery harvest based on reputation share."""
        catches = {aid: 0.0 for aid in self.fishermen}
        if self.private_availability <= 0.0:
            return catches, 0.0

        fish = float(self._private_fish_population)
        if fish < self.private_min_stock:
            return catches, 0.0

        available_raw = self.private_availability * fish
        available_cap = max(0.0, fish - self.private_min_stock)
        available = min(available_raw, available_cap)
        if available <= 0.0:
            return catches, 0.0

        if self.private_allocation_mode == "reputation":
            weights = np.clip(self._reputation, 0.0, None)
        else:
            weights = np.ones(self.num_fishermen, dtype=np.float32)
        total_weight = float(np.sum(weights))
        if total_weight <= EPS:
            weights = np.ones(self.num_fishermen, dtype=np.float32)
            total_weight = float(self.num_fishermen)

        shares = weights / total_weight
        for agent_id in self.fishermen:
            idx = self._agent_index[agent_id]
            catches[agent_id] = float(available * shares[idx])

        return catches, float(available)

    def _compute_quota_limits(self) -> np.ndarray:
        quota_limits = np.zeros(self.num_commons, dtype=np.float32)
        for commons_idx in range(self.num_commons):
            if (
                self._fish_population[commons_idx]
                >= self._mechanism_params["min_stock"][commons_idx]
            ):
                quota_limits[commons_idx] = float(
                    min(
                        self._mechanism_params["fixed_quota"][commons_idx],
                        self._mechanism_params["prop_quota"][commons_idx]
                        * self._fish_population[commons_idx],
                    )
                )
        return quota_limits

    def _compute_access_mask(self) -> np.ndarray:
        access_mask = np.zeros((self.num_fishermen, self.num_commons), dtype=bool)
        for commons_idx, commons in enumerate(self._commons):
            mode = commons["access_mode"]
            if mode == "public":
                access_mask[:, commons_idx] = True
            elif mode == "members":
                access_mask[commons["members_idx"], commons_idx] = True
            elif mode == "voted":
                eligible = commons["eligible_idx"]
                threshold = float(self._access_thresholds[commons_idx])
                if eligible:
                    eligible_reps = self._reputation[eligible]
                    access_mask[eligible, commons_idx] = eligible_reps >= threshold
        return access_mask

    def _update_reputation(
        self,
        agent_actions: Dict[str, Dict[str, np.ndarray]],
        overharvest_totals: np.ndarray,
        quota_totals: np.ndarray,
    ) -> None:
        vote_matrix = np.zeros((self.num_fishermen, self.num_fishermen), dtype=np.float32)
        for agent_id in self.fishermen:
            agent_idx = self._agent_index[agent_id]
            votes = agent_actions[agent_id].get("reputation_votes")
            if votes is None:
                continue
            vote_matrix[agent_idx] = votes

        np.fill_diagonal(vote_matrix, 0.0)
        mask = 1.0 - np.eye(self.num_fishermen, dtype=np.float32)
        if self.reputation_vote_weight_by_reputation:
            weights = self._reputation.reshape(-1, 1) * mask
            weighted_sum = (vote_matrix * weights).sum(axis=0)
            weight_totals = weights.sum(axis=0)
            mean_votes = np.where(
                weight_totals > EPS,
                weighted_sum / weight_totals,
                (vote_matrix * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), EPS),
            )
        else:
            mean_votes = (vote_matrix * mask).sum(axis=0) / np.maximum(
                mask.sum(axis=0), EPS
            )

        mean_votes = np.clip(mean_votes, -1.0, 1.0)
        vote_target = 0.5 * (mean_votes + 1.0)
        blend = float(
            np.clip(self.reputation_vote_decay * self.reputation_vote_weight, 0.0, 1.0)
        )
        if blend > 0.0:
            self._reputation = (1.0 - blend) * self._reputation + blend * vote_target

        with np.errstate(divide="ignore", invalid="ignore"):
            over_ratio = np.where(
                quota_totals > EPS,
                overharvest_totals / quota_totals,
                (overharvest_totals > 0.0).astype(np.float32),
            )
        behavior_delta = -self.reputation_behavior_weight * over_ratio

        self._reputation = np.clip(
            self._reputation + behavior_delta,
            self.reputation_min,
            self.reputation_max,
        ).astype(np.float32)

    def _apply_governance_votes(
        self, agent_actions: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        self._mechanism_params = {
            name: self._episode_mechanism_base[name].copy()
            for name in MECHANISM_RANGES
        }
        self._access_thresholds = self._access_thresholds_base.copy()

        if not self.enable_governance:
            if self.mechanism_noise_mode == "step":
                self._apply_mechanism_noise(self._mechanism_params)
            return

        for commons_idx, commons in enumerate(self._commons):
            gov_idx = commons["governance_idx"]
            if not gov_idx:
                continue
            votes = []
            for idx in gov_idx:
                aid = self.fishermen[idx]
                gov_votes = agent_actions[aid].get("governance_votes")
                if gov_votes is None:
                    continue
                votes.append(gov_votes[commons_idx])
            if not votes:
                continue

            mean_vote = np.mean(np.stack(votes, axis=0), axis=0)
            vote_params = self._map_votes_to_params(mean_vote)

            g = float(np.clip(self.governance_strength, 0.0, 1.0))
            for param in MECHANISM_RANGES:
                base_val = self._episode_mechanism_base[param][commons_idx]
                self._mechanism_params[param][commons_idx] = (
                    (1.0 - g) * base_val + g * vote_params[param]
                )
            self._access_thresholds[commons_idx] = (
                (1.0 - g) * self._access_thresholds_base[commons_idx]
                + g * vote_params["access_threshold"]
            )

        if self.mechanism_noise_mode == "step":
            self._apply_mechanism_noise(self._mechanism_params)

    def _map_votes_to_params(self, votes: np.ndarray) -> Dict[str, float]:
        v = np.clip(np.asarray(votes, dtype=np.float32), 0.0, 1.0)
        return {
            "fixed_quota": float(v[0]),
            "prop_quota": float(v[1]),
            "min_stock": float(v[2]),
            "fine_amount": float(v[3]) * 2.0,
            "ban_period": float(np.round(v[4] * self._ban_period_max)),
            "ban_success_prob": float(v[5]),
            "fine_success_prob": float(v[6]),
            "access_threshold": float(v[7]),
        }

    def _apply_mechanism_noise(self, params: Dict[str, np.ndarray]) -> None:
        for name, (low, high) in MECHANISM_RANGES.items():
            std = self.mechanism_noise_std[name]
            if np.all(std == 0.0):
                continue
            noise = self._rng.normal(0.0, std, size=self.num_commons)
            if name == "ban_period":
                values = np.round(params[name] + noise)
                params[name] = np.clip(values, low, high).astype(np.float32)
            else:
                params[name] = np.clip(params[name] + noise, low, high).astype(
                    np.float32
                )

    def _build_observation(self, agent_idx: int) -> np.ndarray:
        eco = np.stack((self._algae_population, self._fish_population), axis=1).reshape(
            -1
        )
        parts = [eco.astype(np.float32)]
        if self.include_quota_in_obs:
            parts.append(self._last_quota_limits.astype(np.float32))
        if self.include_access_in_obs:
            parts.append(self._access_mask[agent_idx].astype(np.float32))
        if self.include_private_in_obs and self.enable_private_fishery:
            parts.append(
                np.array(
                    [self._private_algae_population, self._private_fish_population],
                    dtype=np.float32,
                )
            )
        if self.include_reputation_in_obs:
            parts.append(self._reputation.astype(np.float32))

        obs = np.concatenate(parts).astype(np.float32)

        if self.observation_noise_std > 0.0 and self._obs_noise_dim > 0:
            noise = self._rng.normal(
                0.0, self.observation_noise_std, size=self._obs_noise_dim
            )
            obs[: self._obs_noise_dim] = np.maximum(
                0.0, obs[: self._obs_noise_dim] + noise
            )
        return obs

    def _parse_actions(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        parsed: Dict[str, Dict[str, np.ndarray]] = {}
        for agent_id in self.fishermen:
            raw_action = action_dict.get(agent_id)
            action = self._normalize_action(raw_action)

            harvest = action[self._harvest_slice] if self._harvest_slice else None
            governance_votes = None
            reputation_votes = None

            if self.enable_governance and self._governance_slice is not None:
                governance_votes = action[self._governance_slice].reshape(
                    self.num_commons, self._governance_param_count
                )
            if self.enable_reputation and self._reputation_slice is not None:
                reputation_votes = action[self._reputation_slice]

            parsed[agent_id] = {
                "harvest": harvest,
                "governance_votes": governance_votes,
                "reputation_votes": reputation_votes,
            }

        return parsed

    def _normalize_action(self, action: Optional[np.ndarray]) -> np.ndarray:
        if action is None:
            action = np.zeros(self._action_dim, dtype=np.float32)
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self._action_dim:
            arr = np.pad(arr, (0, self._action_dim - arr.size))
        elif arr.size > self._action_dim:
            arr = arr[: self._action_dim]

        if np.any(self._action_noise_std > 0.0):
            noise = self._rng.normal(0.0, self._action_noise_std, size=self._action_dim)
            if self.action_noise_clip > 0.0:
                max_dev = self.action_noise_clip * self._action_noise_std
                noise = np.clip(noise, -max_dev, max_dev)
            arr = arr + noise

        return np.clip(arr, self._action_low, self._action_high)

    def _build_spaces(self) -> None:
        self._harvest_slice = slice(0, self.num_commons)
        offset = self.num_commons

        self._governance_slice = None
        self._governance_param_count = 0
        if self.enable_governance:
            self._governance_param_count = len(GOVERNANCE_PARAM_NAMES)
            gov_size = self.num_commons * self._governance_param_count
            self._governance_slice = slice(offset, offset + gov_size)
            offset += gov_size

        self._reputation_slice = None
        if self.enable_reputation:
            self._reputation_slice = slice(offset, offset + self.num_fishermen)
            offset += self.num_fishermen

        self._action_dim = offset

        act_low_parts = [
            np.full(self.num_commons, ACTION_BOUNDS["low"], dtype=np.float32)
        ]
        act_high_parts = [
            np.full(self.num_commons, ACTION_BOUNDS["high"], dtype=np.float32)
        ]
        if self.enable_governance:
            act_low_parts.append(
                np.zeros(self.num_commons * self._governance_param_count, dtype=np.float32)
            )
            act_high_parts.append(
                np.ones(self.num_commons * self._governance_param_count, dtype=np.float32)
            )
        if self.enable_reputation:
            act_low_parts.append(-np.ones(self.num_fishermen, dtype=np.float32))
            act_high_parts.append(np.ones(self.num_fishermen, dtype=np.float32))

        self._action_low = np.concatenate(act_low_parts)
        self._action_high = np.concatenate(act_high_parts)

        self._act_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

        eco_dim = self.num_commons * 2
        obs_low_parts = [
            np.full(eco_dim, OBSERVATION_BOUNDS["low"], dtype=np.float32)
        ]
        obs_high_parts = [
            np.full(eco_dim, OBSERVATION_BOUNDS["high"], dtype=np.float32)
        ]
        self._obs_noise_dim = eco_dim

        if self.include_quota_in_obs:
            obs_low_parts.append(np.zeros(self.num_commons, dtype=np.float32))
            obs_high_parts.append(
                np.full(self.num_commons, OBSERVATION_BOUNDS["high"], dtype=np.float32)
            )
            self._obs_noise_dim += self.num_commons

        if self.include_access_in_obs:
            obs_low_parts.append(np.zeros(self.num_commons, dtype=np.float32))
            obs_high_parts.append(np.ones(self.num_commons, dtype=np.float32))

        if self.include_private_in_obs and self.enable_private_fishery:
            obs_low_parts.append(np.full(2, OBSERVATION_BOUNDS["low"], dtype=np.float32))
            obs_high_parts.append(
                np.full(2, OBSERVATION_BOUNDS["high"], dtype=np.float32)
            )
            self._obs_noise_dim += 2

        if self.include_reputation_in_obs:
            obs_low_parts.append(
                np.full(self.num_fishermen, self.reputation_min, dtype=np.float32)
            )
            obs_high_parts.append(
                np.full(self.num_fishermen, self.reputation_max, dtype=np.float32)
            )

        obs_low = np.concatenate(obs_low_parts)
        obs_high = np.concatenate(obs_high_parts)
        self._obs_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_low.shape[0],),
            dtype=np.float32,
        )

        self.observation_spaces = {aid: self._obs_space for aid in self.fishermen}
        self.action_spaces = {aid: self._act_space for aid in self.fishermen}

    def _parse_action_noise(self, value) -> np.ndarray:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 1:
                return np.full(self._action_dim, float(arr[0]), dtype=np.float32)
            if arr.size < self._action_dim:
                arr = np.pad(arr, (0, self._action_dim - arr.size))
            elif arr.size > self._action_dim:
                arr = arr[: self._action_dim]
            return arr.astype(np.float32)
        return np.full(self._action_dim, float(value), dtype=np.float32)

    @staticmethod
    def _parse_noise_config(value, keys: Iterable[str], default: float = 0.0) -> Dict:
        if value is None:
            value = default
        if isinstance(value, dict):
            return {k: float(value.get(k, default)) for k in keys}
        return {k: float(value) for k in keys}

    def _sample_ecology_parameters(self) -> None:
        self._alpha = self._sample_param(
            self.alpha_mean, self.ecology_noise_std["alpha"]
        )
        self._beta = self._sample_param(self.beta_mean, self.ecology_noise_std["beta"])
        self._delta = self._sample_param(
            self.delta_mean, self.ecology_noise_std["delta"]
        )
        self._gamma = self._sample_param(
            self.gamma_mean, self.ecology_noise_std["gamma"]
        )
        self._private_alpha = self._sample_scalar_param(
            self.alpha_mean, self.ecology_noise_std["alpha"]
        )
        self._private_beta = self._sample_scalar_param(
            self.beta_mean, self.ecology_noise_std["beta"]
        )
        self._private_delta = self._sample_scalar_param(
            self.delta_mean, self.ecology_noise_std["delta"]
        )
        self._private_gamma = self._sample_scalar_param(
            self.gamma_mean, self.ecology_noise_std["gamma"]
        )

    def _sample_param(self, mean: float, std: float) -> np.ndarray:
        if std <= 0.0:
            return np.full(self.num_commons, mean, dtype=np.float32)
        values = self._rng.normal(mean, std, size=self.num_commons)
        return np.maximum(0.0, values).astype(np.float32)

    def _sample_scalar_param(self, mean: float, std: float) -> float:
        if std <= 0.0:
            return float(mean)
        value = float(self._rng.normal(mean, std))
        return float(max(0.0, value))

    def _normalize_commons_config(self, commons_config) -> List[Dict]:
        if commons_config is None:
            commons_config = [{} for _ in range(self.num_commons)]
        if len(commons_config) != self.num_commons:
            raise ValueError(
                "commons config length must match num_commons: "
                f"{len(commons_config)} != {self.num_commons}"
            )

        normalized = []
        for idx, raw in enumerate(commons_config):
            cfg = dict(raw) if raw else {}
            members = cfg.get("members", "all")
            members_ids = self._resolve_agent_list(members)

            governance = cfg.get("governance", "members")
            if governance == "members":
                governance_ids = list(members_ids)
            elif governance == "all":
                governance_ids = list(self.fishermen)
            else:
                governance_ids = self._resolve_agent_list(governance)

            access_mode = str(cfg.get("access_mode", self.access_mode_default))
            if access_mode not in ACCESS_MODES:
                raise ValueError(
                    f"access_mode must be one of {sorted(ACCESS_MODES)}, got {access_mode}"
                )

            access_scope = str(cfg.get("access_scope", self.access_scope_default))
            if access_scope not in ACCESS_SCOPES:
                raise ValueError(
                    f"access_scope must be one of {sorted(ACCESS_SCOPES)}, got {access_scope}"
                )

            access_threshold_init = float(
                cfg.get("access_threshold_init", self.access_threshold_init)
            )

            mechanism_params = {
                **self._base_mechanism_defaults,
                **cfg.get("mechanism_params", {}),
            }

            members_idx = [self._agent_index[aid] for aid in members_ids]
            governance_idx = [self._agent_index[aid] for aid in governance_ids]
            eligible_idx = (
                list(range(self.num_fishermen))
                if access_scope == "all"
                else list(members_idx)
            )

            normalized.append(
                {
                    "id": cfg.get("id", f"commons_{idx}"),
                    "members": members_ids,
                    "members_idx": members_idx,
                    "governance": governance_ids,
                    "governance_idx": governance_idx,
                    "access_mode": access_mode,
                    "access_scope": access_scope,
                    "eligible_idx": eligible_idx,
                    "access_threshold_init": access_threshold_init,
                    "mechanism_params": mechanism_params,
                }
            )

        return normalized

    def _resolve_agent_list(self, value) -> List[str]:
        if value is None or value == "all":
            return list(self.fishermen)
        if isinstance(value, str):
            if value in self.fishermen:
                return [value]
            raise ValueError(f"Unknown agent reference: {value}")
        if isinstance(value, (list, tuple, set)):
            if not value:
                return []
            if all(isinstance(v, int) for v in value):
                return [self.fishermen[int(v)] for v in value]
            return [str(v) for v in value]
        raise ValueError(f"Unsupported agent list: {value}")

    def _build_base_mechanism_params(self) -> Dict[str, np.ndarray]:
        params = {
            name: np.zeros(self.num_commons, dtype=np.float32)
            for name in MECHANISM_RANGES
        }
        for commons_idx, commons in enumerate(self._commons):
            mech = commons["mechanism_params"]
            params["fixed_quota"][commons_idx] = float(mech["fixed_quota"])
            params["prop_quota"][commons_idx] = float(mech["prop_quota"])
            params["min_stock"][commons_idx] = float(mech["min_stock"])
            params["fine_amount"][commons_idx] = float(mech["fine_amount"])
            params["ban_period"][commons_idx] = float(mech["ban_period"])
            params["ban_success_prob"][commons_idx] = float(mech["ban_success_prob"])
            params["fine_success_prob"][commons_idx] = float(mech["fine_success_prob"])
        return params
