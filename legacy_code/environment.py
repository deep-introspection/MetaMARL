"""Multi-agent fishery environment with Lotka-Volterra ecosystem dynamics.

This module contains the fishery environment implementation for the bilevel
optimization experiment. The environment simulates a shared fishery resource
with regulatory mechanisms including quotas, fines, and temporary bans.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from legacy_code.config import (
    ACTION_BOUNDS,
    DEFAULT_ECOLOGY_CONFIG,
    DEFAULT_MECHANISM_CONFIG,
    EPS,
    OBSERVATION_BOUNDS,
)


class FisheryEnvFixed(MultiAgentEnv):
    """Multi-agent fishery environment with Lotka-Volterra ecosystem dynamics.

    This environment simulates a fishery where multiple fishermen harvest from
    a shared resource governed by algae-fish population dynamics. The regulatory
    mechanism includes quotas, fines, and temporary bans.

    State Space:
        - Algae population (continuous)
        - Fish population (continuous)

    Action Space:
        - Desired harvest fraction [0,1] for each fisherman

    Mechanism Parameters (fixed per episode):
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
        self.alpha = float(ecology_config["alpha"])  # Algae growth rate
        self.beta = float(ecology_config["beta"])  # Predation rate
        self.delta = float(ecology_config["delta"])  # Fish growth efficiency
        self.gamma = float(ecology_config["gamma"])  # Fish death rate
        self.dt = float(ecology_config["dt"])  # Integration time step
        self.horizon = int(ecology_config["horizon"])  # Episode length
        self.algae_init = float(ecology_config["algae_init"])
        self.fish_init = float(ecology_config["fish_init"])

        # Regulatory mechanism parameters (fixed during episode)
        mechanism_config = {**DEFAULT_MECHANISM_CONFIG, **config}
        self.fixed_quota = float(mechanism_config["fixed_quota"])
        self.prop_quota = float(mechanism_config["prop_quota"])
        self.min_stock = float(mechanism_config["min_stock"])
        self.fine_amount = float(mechanism_config["fine_amount"])
        self.ban_period = int(mechanism_config["ban_period"])

        # Agent setup
        self.num_fishermen = int(config.get("num_fishermen", 3))
        self.fishermen = [f"fisherman_{i}" for i in range(self.num_fishermen)]
        self.possible_agents = list(self.fishermen)

        # Define observation and action spaces
        self._obs_space = spaces.Box(
            low=OBSERVATION_BOUNDS["low"],
            high=OBSERVATION_BOUNDS["high"],
            shape=OBSERVATION_BOUNDS["shape"],
            dtype=OBSERVATION_BOUNDS["dtype"],
        )
        self._act_space = spaces.Box(
            low=ACTION_BOUNDS["low"],
            high=ACTION_BOUNDS["high"],
            shape=ACTION_BOUNDS["shape"],
            dtype=ACTION_BOUNDS["dtype"],
        )
        self.observation_spaces = {aid: self._obs_space for aid in self.fishermen}
        self.action_spaces = {aid: self._act_space for aid in self.fishermen}

        # Environment state
        self._time_step = 0
        self._algae_population = self.algae_init
        self._fish_population = self.fish_init
        self._agent_bans: Dict[str, int] = {aid: 0 for aid in self.fishermen}

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
        rng = np.random.default_rng(seed)

        # Initialize populations with small random variation
        algae_mean = np.log(max(self.algae_init, 1e-3))
        fish_mean = np.log(max(self.fish_init, 1e-3))

        self._algae_population = max(
            EPS, float(rng.lognormal(mean=algae_mean, sigma=0.05))
        )
        self._fish_population = max(
            EPS, float(rng.lognormal(mean=fish_mean, sigma=0.05))
        )

        # Reset all agent bans
        for agent_id in self.fishermen:
            self._agent_bans[agent_id] = 0

        # Create initial observations
        observations = {
            agent_id: np.array(
                [self._algae_population, self._fish_population], dtype=np.float32
            )
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
            action_dict: Dictionary mapping agent IDs to actions (harvest fractions)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        agent_catches: Dict[str, float] = {}
        agent_rewards: Dict[str, float] = {}

        # Check if fishing is allowed (above minimum stock threshold)
        fishing_allowed = self._fish_population >= self.min_stock

        if fishing_allowed:
            # Calculate effective quota (minimum of fixed and proportional)
            quota_limit = min(self.fixed_quota, self.prop_quota * self._fish_population)

            # Process each fisherman's action
            for agent_id in self.fishermen:
                agent_catches[agent_id], agent_rewards[agent_id] = (
                    self._process_agent_action(agent_id, action_dict, quota_limit)
                )
        else:
            # No fishing allowed when below minimum stock
            quota_limit = 0.0
            for agent_id in self.fishermen:
                agent_catches[agent_id] = 0.0
                agent_rewards[agent_id] = 0.0

        # Calculate total harvest
        total_harvest = sum(agent_catches.values())

        # Update ecosystem using Lotka-Volterra dynamics
        self._update_ecosystem(total_harvest)

        # Increment time step
        self._time_step += 1

        # Create observations for next step
        observations = {
            agent_id: np.array(
                [self._algae_population, self._fish_population], dtype=np.float32
            )
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
                "algae": self._algae_population,
                "fish": self._fish_population,
                "catch": agent_catches[agent_id],
                "ban": self._agent_bans[agent_id],
                "quota_limit": quota_limit,
            }
            for agent_id in self.fishermen
        }

        return observations, agent_rewards, is_terminated, is_truncated, infos

    def _process_agent_action(
        self, agent_id: str, action_dict: Dict[str, np.ndarray], quota_limit: float
    ) -> Tuple[float, float]:
        """Process a single agent's fishing action.

        Args:
            agent_id: ID of the agent
            action_dict: Dictionary of all agent actions
            quota_limit: Current quota limit

        Returns:
            Tuple of (catch_amount, reward)
        """
        # Check if agent is currently banned
        if self._agent_bans[agent_id] > 0:
            self._agent_bans[agent_id] -= 1
            return 0.0, 0.0

        # Get and clip agent's desired harvest fraction
        action = action_dict.get(agent_id, np.array([0.0]))
        desired_fraction = float(np.clip(action[0], 0.0, 1.0))
        desired_harvest = desired_fraction * self._fish_population

        # Calculate over-harvest penalty
        over_harvest = max(0.0, desired_harvest - quota_limit)

        # Actual catch is limited by available fish
        actual_catch = min(desired_harvest, self._fish_population)

        # Calculate reward (catch minus fines)
        reward = actual_catch - self.fine_amount * over_harvest

        # Apply ban if over-harvesting occurred
        if over_harvest > EPS and self.ban_period > 0:
            self._agent_bans[agent_id] = self.ban_period

        return actual_catch, reward

    def _update_ecosystem(self, total_harvest: float) -> None:
        """Update ecosystem populations using Lotka-Volterra dynamics.

        Args:
            total_harvest: Total fish harvested this time step
        """
        # Current populations
        algae = self._algae_population
        fish = self._fish_population

        # Compute derivatives
        d_algae_dt = self.alpha * algae - self.beta * algae * fish
        d_fish_dt = self.delta * algae * fish - self.gamma * fish - total_harvest

        # Update populations with Euler integration
        self._algae_population = max(0.0, algae + self.dt * d_algae_dt)
        self._fish_population = max(0.0, fish + self.dt * d_fish_dt)
