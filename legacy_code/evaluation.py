"""Evaluation metrics and functions for sustainability assessment.

This module provides functionality to evaluate trained agents with sustainability
metrics that balance economic returns with environmental protection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from legacy_code.environment import FisheryEnvFixed
from legacy_code.config import EPS


@dataclass
class EvaluationMetrics:
    """Container for sustainability and performance metrics."""

    mean_reward: float
    mean_min_fish: float
    collapse_rate: float
    sustainability_penalty: float

    def objective_score(self, sustainability_weight: float) -> float:
        """Compute weighted objective score.

        Args:
            sustainability_weight: Weight for sustainability penalty

        Returns:
            Objective score (reward - weighted penalty)
        """
        return self.mean_reward - sustainability_weight * self.sustainability_penalty

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "mean_reward": self.mean_reward,
            "mean_min_fish": self.mean_min_fish,
            "collapse_rate": self.collapse_rate,
            "sustainability_penalty": self.sustainability_penalty,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"EvaluationMetrics("
            f"reward={self.mean_reward:.3f}, "
            f"min_fish={self.mean_min_fish:.3f}, "
            f"collapse_rate={self.collapse_rate:.2%}, "
            f"penalty={self.sustainability_penalty:.3f})"
        )


def evaluate_mechanism_with_metrics(
    algorithm,
    environment: FisheryEnvFixed,
    num_episodes: int,
    sustainability_threshold: float,
    record_trajectories: bool = False,
) -> Tuple[EvaluationMetrics, Optional[List[Dict]]]:
    """Evaluate trained agents with sustainability metrics.

    Args:
        algorithm: Trained PPO algorithm
        environment: Environment instance
        num_episodes: Number of evaluation episodes
        sustainability_threshold: Fish population collapse threshold
        record_trajectories: Whether to record detailed trajectories

    Returns:
        Tuple of (evaluation metrics, trajectory records if requested)
    """
    policy = algorithm.get_policy("fisher_policy")

    total_reward = 0.0
    episode_min_fish = []
    collapsed_episodes = 0
    trajectory_records: List[Dict] = []

    for episode_idx in range(num_episodes):
        observations, _ = environment.reset()
        terminated = {agent_id: False for agent_id in environment.fishermen}
        truncated = {agent_id: False for agent_id in environment.fishermen}
        episode_min_fish_stock = float("inf")
        step_count = 0

        # Run episode
        while not (any(terminated.values()) or any(truncated.values())):
            # Get actions from policy
            agent_actions = {}
            for agent_id in environment.fishermen:
                action, _, _ = policy.compute_single_action(
                    observations[agent_id], explore=False
                )
                agent_actions[agent_id] = action

            # Execute step
            observations, rewards, terminated, truncated, infos = environment.step(
                agent_actions
            )

            # Accumulate rewards
            total_reward += sum(rewards.values())

            # Track minimum fish stock for this episode
            current_fish = infos[environment.fishermen[0]]["fish"]
            episode_min_fish_stock = min(episode_min_fish_stock, current_fish)

            # Record trajectory if requested
            if record_trajectories:
                total_catch = sum(
                    infos[agent_id]["catch"] for agent_id in environment.fishermen
                )
                trajectory_records.append(
                    {
                        "episode": episode_idx,
                        "step": step_count,
                        "fish_population": current_fish,
                        "algae_population": infos[environment.fishermen[0]]["algae"],
                        "total_harvest": total_catch,
                        "quota_limit": infos[environment.fishermen[0]]["quota_limit"],
                    }
                )

            step_count += 1

        # Record episode statistics
        episode_min_fish.append(episode_min_fish_stock)
        if episode_min_fish_stock < sustainability_threshold:
            collapsed_episodes += 1

    # Compute aggregate metrics
    mean_reward = total_reward / num_episodes
    mean_min_fish = float(np.mean(episode_min_fish))
    collapse_rate = collapsed_episodes / num_episodes

    # Compute sustainability penalty (normalized deficit)
    sustainability_penalties = [
        max(
            0.0,
            (sustainability_threshold - min_fish) / max(EPS, sustainability_threshold),
        )
        for min_fish in episode_min_fish
    ]
    mean_sustainability_penalty = float(np.mean(sustainability_penalties))

    metrics = EvaluationMetrics(
        mean_reward=mean_reward,
        mean_min_fish=mean_min_fish,
        collapse_rate=collapse_rate,
        sustainability_penalty=mean_sustainability_penalty,
    )

    return metrics, (trajectory_records if record_trajectories else None)


def evaluate_multiple_episodes(
    algorithm,
    environment: FisheryEnvFixed,
    num_episodes: int,
    sustainability_threshold: float,
    detailed_logging: bool = False,
) -> Dict:
    """Evaluate algorithm over multiple episodes with detailed statistics.

    Args:
        algorithm: Trained algorithm to evaluate
        environment: Environment instance
        num_episodes: Number of episodes to run
        sustainability_threshold: Fish population collapse threshold
        detailed_logging: Whether to collect detailed episode statistics

    Returns:
        Dictionary with comprehensive evaluation results
    """
    metrics, trajectories = evaluate_mechanism_with_metrics(
        algorithm,
        environment,
        num_episodes,
        sustainability_threshold,
        record_trajectories=detailed_logging,
    )

    results = {
        "metrics": metrics.to_dict(),
        "num_episodes": num_episodes,
        "sustainability_threshold": sustainability_threshold,
    }

    if trajectories:
        # Add trajectory summary statistics
        fish_populations = [t["fish_population"] for t in trajectories]
        algae_populations = [t["algae_population"] for t in trajectories]
        harvests = [t["total_harvest"] for t in trajectories]

        results["trajectory_stats"] = {
            "fish_population": {
                "mean": float(np.mean(fish_populations)),
                "std": float(np.std(fish_populations)),
                "min": float(np.min(fish_populations)),
                "max": float(np.max(fish_populations)),
            },
            "algae_population": {
                "mean": float(np.mean(algae_populations)),
                "std": float(np.std(algae_populations)),
                "min": float(np.min(algae_populations)),
                "max": float(np.max(algae_populations)),
            },
            "harvest": {
                "mean": float(np.mean(harvests)),
                "std": float(np.std(harvests)),
                "min": float(np.min(harvests)),
                "max": float(np.max(harvests)),
            },
        }
        results["trajectories"] = trajectories

    return results


def compute_sustainability_score(
    fish_populations: List[float],
    sustainability_threshold: float,
    penalty_weight: float = 1.0,
) -> float:
    """Compute sustainability score from fish population time series.

    Args:
        fish_populations: List of fish population values over time
        sustainability_threshold: Threshold below which ecosystem is considered collapsed
        penalty_weight: Weight for penalty calculation

    Returns:
        Sustainability score (higher is better)
    """
    if not fish_populations:
        return 0.0

    # Count collapsed time steps
    collapsed_steps = sum(
        1 for pop in fish_populations if pop < sustainability_threshold
    )
    collapse_rate = collapsed_steps / len(fish_populations)

    # Compute penalty for being below threshold
    penalties = [
        max(0.0, (sustainability_threshold - pop) / max(EPS, sustainability_threshold))
        for pop in fish_populations
    ]
    mean_penalty = np.mean(penalties) if penalties else 0.0

    # Higher score for lower collapse rate and lower penalty
    sustainability_score = (1.0 - collapse_rate) * (1.0 - penalty_weight * mean_penalty)

    return max(0.0, sustainability_score)  # Ensure non-negative
