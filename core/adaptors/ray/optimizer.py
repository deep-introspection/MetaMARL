from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import ray
from ray.rllib.utils.typing import AgentID, ResultDict
from ray.train._internal.checkpoint_manager import _TrainingResult

from core.adaptors.ray.schema import EvalSchema, RaySchema, TrainSchema

# TODO temporary
from core.adaptors.ray.utils import (
    build_learner,
    build_performance,
    build_rollout,
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
)
from core.annotations import override
from core.metrics.logger import MetricLogger
from core.optimizers.base import Optimizer

# Deprecated
from core.utils import to_float

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
    ):
        super().__init__(config)
        # self.algo = algo

        # TODO maybe this either needs to be an actor. or atleast have method to serialize data
        self.logger = MetricLogger.from_schema(RaySchema)

        # self.eval_episodes = config.eval_episodes
        # TODO fallback if rollout_fragment_length not in eval_cfg
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.evaluation_config.get("rollout_fragment_length")
        )
        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._inner_iter: int = 0
        self._es_round: int = 0

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        num_envs = self.config.rllib_cfg.num_envs_per_env_runner
        num_seeds = len(self.config.seeds)
        num_mechanisms = num_envs // num_seeds
        return num_mechanisms

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

    # TODO (nadine) : in the future this could be separated into a different class if justified
    def _to_logger_payload(
        self, result: ResultDict, is_eval: bool = False
    ) -> RaySchema:
        if is_eval:
            evaluation = EvalSchema(
                rollout=build_rollout(result),
                performance=build_performance(result),
            )
            return RaySchema(train=None, eval=evaluation)

        train = TrainSchema(
            rollout=build_rollout(result),
            learner=build_learner(result),
            performance=build_performance(result),
        )
        eval_result = result.get("evaluation")
        evaluation = None

        if isinstance(eval_result, dict):
            evaluation = EvalSchema(
                rollout=build_rollout(eval_result),
                performance=build_performance(eval_result),
            )
        return RaySchema(train=train, eval=evaluation)

    @override(Optimizer)
    def run(self) -> None:
        logger.info("[PPO] Training step started")
        result = ray.get(self.policy_actor.train.remote())
        # step = int(to_float(result.get("training_iteration")) or 0)

        # Local inner-loop iteration. This resets for each outer ES round.
        self._inner_iter += 1
        step = self._inner_iter
        self.logger.push(key=("iter",), value=step)

        # RLlib's own lifetime training counter, retained only for debugging.
        rllib_training_iteration = int(to_float(result.get("training_iteration")) or 0)

        metrics = self._to_logger_payload(result)
        self.logger.push_data(metrics)

        # TODO temporary to be moved to a logger Extract metrics
        ep_return = get_episode_return_mean(result)
        steps_iter, steps_life = get_env_steps(result)

        # Track metrics
        self._training_rewards.append(ep_return)
        policy_loss = get_policy_loss_if_present(result)
        self._training_losses.append(policy_loss)

        logger.info(
            "[PPO] Training step completed | "
            "outer_iter=%d | inner_iter=%d | rllib_iter_lifetime=%d | "
            "ep_return=%.4f | env_steps_iter=%d | "
            "env_steps_lifetime=%d | policy_loss=%s",
            self._es_round,
            step,
            rllib_training_iteration,
            ep_return,
            steps_iter,
            steps_life,
            f"{policy_loss:.6f}" if np.isfinite(policy_loss) else "NA",
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        logger.info("[PPO] Evaluation started")
        result = ray.get(self.policy_actor.evaluate.remote())
        metrics = self._to_logger_payload(result, is_eval=True)
        self.logger.push_data(metrics)
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def reset(self) -> None:
        """Reset policy weights to random initialization."""
        logger.info("[PPO] Resetting policy weights")
        self._training_rewards = []
        self._training_losses = []
        self._inner_iter = 0
        self._es_round += 1
        ray.get(self.policy_actor.reset.remote())
        self.logger.reset()

    @override(Optimizer)
    def stop(self) -> None:
        ray.get(self.policy_actor.stop.remote())
        return self.logger.reduce(complie=True)

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        # TODO
        pass
