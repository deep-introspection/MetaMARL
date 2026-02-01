from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

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
        # self.algo.train()
        ray.get(self.policy_actor.train.remote())
        logger.info("[PPO] Training step completed")

    @override(Optimizer)
    def evaluate(self) -> None:
        logger.info("[PPO] Evaluation started")
        ray.get(self.policy_actor.evaluate.remote())
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def stop(self) -> None:
        ray.get(self.policy_actor.stop.remote())

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        return self.algo.save(checkpoint_dir)
