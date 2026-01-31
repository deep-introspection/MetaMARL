from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.rllib.utils.typing import AgentID

from core.adaptors.ray.eval_utilis import _evaluation_runner_remote
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
        world: World
    ):
        super().__init__(config)
        # self.algo = algo
        self.world = world # TODO replace by envFactory
        self.eval_episodes = config.eval_episodes
        self.eval_base_seed = config.eval_base_seed
        self.rollout_fragment_length = config.rollout_fragment_length

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
        logger.info(f"[PPO] Training step completed")

    # TODO : parallize this loop
    # @override(Optimizer)
    def evaluate(
        self,
        # parallel_train_future: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> float:
        # TODO Implement manually
        # return self.algo.evaluate()
        # TODO parallelize

        # Build agent -> policy mapping
        agent_to_policy = self._build_agent_policy_map()
        agents = list(agent_to_policy.keys())

        # Cache policy/module handles
        policy_handles = {
            policy_id: self._get_policy_handle(policy_id)
            for policy_id in set(agent_to_policy.values())
        }


        for b in range(self.batch_capacity):
            for ep in range(self.eval_episodes):
                seed = (
                    None
                    if self.eval_base_seed is None
                    else self.eval_base_seed + ep
                )

                env = self.config._env_creator(
                    world=self.world,
                    opt_id=self.opt_id,
                    agents=agents,
                    **self.config.env_config,
                )
                observations, _ = env.reset(seed=seed)
                terminated = {aid: False for aid in agents}
                truncated = {aid: False for aid in agents}
                step_count = 0

                while (
                    not any(terminated.values())
                    and not any(truncated.values())
                    and step_count < env.horizon
                ):
                    actions = {}

                    for agent_id in agents:
                        obs = observations[agent_id]
                        policy_id = agent_to_policy[agent_id]
                        handle = policy_handles[policy_id]

                        # --- action computation (policy OR RLModule) ---
                        if hasattr(handle, "compute_single_action"):
                            # Policy API
                            action, _, _ = handle.compute_single_action(
                                obs,
                                explore=False,
                            )
                        else:
                            # RLModule API
                            out = handle.forward_inference(
                                {"obs": np.asarray(obs)[None, ...]}
                            )
                            dist_cls = handle.get_inference_action_dist_cls()
                            dist = dist_cls.from_logits(out["action_dist_inputs"])
                            action = dist.sample().cpu().numpy()[0]

                        # --- clip to correct action space ---
                         # TODO observation spaces not passed to env_cfg
                        act_space = self.config.env_config["action_spaces"][agent_id]
                        action = np.asarray(action, dtype=act_space.dtype)
                        action = np.clip(action, act_space.low, act_space.high)

                        if not np.isfinite(action).all():
                            raise RuntimeError(
                                f"Invalid action for {agent_id} ({policy_id}): {action}"
                            )

                        actions[agent_id] = action

                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    step_count += 1
        env.close()

    def evaluate_async(self) -> float:
        num_runners = self.batch_capacity # TODO temproarily set to batch_capacity but change later to proper config item

        futures = [
            _evaluation_runner_remote.remote(
                policy_actor=self.policy_actor,
                config=self.config,
                world=self.world,
                opt_id=self.opt_id,
                eval_episodes=self.eval_episodes,
                eval_base_seed=self.eval_base_seed,
            )
            for _ in range(num_runners)
        ]

        ray.get(futures)


    # @override(Optimizer)
    def stop(self) -> None:
        # self.algo.stop()
        ray.get(self.policy_actor.stop.remote())

    # @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        return self.algo.save(checkpoint_dir)
    




