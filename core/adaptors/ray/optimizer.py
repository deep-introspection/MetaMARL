from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.utils.typing import AgentID
from ray.train._internal.checkpoint_manager import _TrainingResult

from core.annotations import override
from core.optimizers.base import Optimizer
from core.world.base import World

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
        world: World,
    ):
        super().__init__(config)
        # self.algo = algo
        self.world = world  # TODO replace by envFactory
        # self.eval_episodes = config.eval_episodes
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.rollout_fragment_length
        )
        self.eval_base_seed = config.eval_base_seed
        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._es_round: int = 0

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        return self.config.rllib_cfg.num_envs_per_env_runner

    # TODO move to utils
    def _build_agent_policy_map(self) -> dict[AgentID, str]:
        agent_to_policy: dict[AgentID, str] = {}

        for agent_type, spec in self.config.agent_specs.items():
            policy_id = spec["policy"]
            count = spec["count"]

            for i in range(count):
                agent_id = f"{agent_type}:{i}"
                agent_to_policy[agent_id] = policy_id

        return agent_to_policy

    def _get_policy_handle(self, policy_id: str):
        # RLModule API (newer)
        try:
            return self.algo.get_module(policy_id)
        except Exception:
            # Policy API (older / classic)
            return self.algo.get_policy(policy_id)

    @override(Optimizer)
    def run(self) -> None:
        logger.info("[PPO] Training step started")
        result = ray.get(self.policy_actor.train.remote())

        # Extract metrics for logging
        ep_reward = result.get("env_runners", result).get(
            "episode_reward_mean", result.get("episode_reward_mean", 0)
        )
        iteration = result.get("training_iteration", 0)
        timesteps = result.get("timesteps_total", 0)

        # Track metrics
        self._training_rewards.append(ep_reward or 0)
        learner_info = result.get("info", {}).get("learner", {})
        # Extract policy loss - policies are named fisher_policy_0, fisher_policy_1, etc.
        policy_loss = 0.0
        if learner_info:
            # Average loss across all policies
            losses = []
            for policy_name, policy_stats in learner_info.items():
                ls = policy_stats.get("learner_stats", {})
                if "policy_loss" in ls:
                    losses.append(ls["policy_loss"])
            if losses:
                policy_loss = sum(losses) / len(losses)
        self._training_losses.append(policy_loss)

        logger.info(
            "[PPO] Training step completed | iter=%d | reward=%.4f | timesteps=%d",
            iteration, ep_reward or 0, timesteps,
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        logger.info("[PPO] Evaluation started")
        ray.get(self.policy_actor.evaluate.remote())
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def reset(self) -> None:
        """Reset policy weights to random initialization."""
        logger.info("[PPO] Resetting policy weights")
        # Plot learning curve before reset (if we have data)
        if self._training_rewards:
            self._plot_learning_curve()
        # Clear metrics and increment round
        self._training_rewards = []
        self._training_losses = []
        self._es_round += 1
        ray.get(self.policy_actor.reset.remote())

    def _plot_learning_curve(self, output_dir: str = "results/ppo_curves") -> None:
        """Plot and save PPO learning curve for current ES round."""
        if not self._training_rewards:
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"PPO Learning - ES Round {self._es_round}")

        # Reward curve
        axes[0].plot(self._training_rewards)
        axes[0].set_xlabel("PPO Iteration")
        axes[0].set_ylabel("Episode Reward Mean")
        axes[0].set_title("Reward")
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)

        # Policy loss curve
        axes[1].plot(self._training_losses)
        axes[1].set_xlabel("PPO Iteration")
        axes[1].set_ylabel("Policy Loss")
        axes[1].set_title("Policy Loss")

        plt.tight_layout()
        out_path = Path(output_dir) / f"es_round_{self._es_round:03d}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        logger.info(f"[PPO] Saved learning curve to {out_path}")

    @override(Optimizer)
    def stop(self) -> None:
        ray.get(self.policy_actor.stop.remote())

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        return self.algo.save(checkpoint_dir)
