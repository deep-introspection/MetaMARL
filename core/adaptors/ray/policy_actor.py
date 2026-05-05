from __future__ import annotations

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict


@ray.remote(num_cpus=1)
class PolicyActor:
    """
    Owns the RLlib Algorithm.
    The Algorithm never leaves this actor.
    """

    def __init__(self, algo_config: AlgorithmConfig):
        # Store config for reset capability
        self.algo_config = algo_config
        # Build Algorithm INSIDE actor (critical)
        self.algo: Algorithm = algo_config.build_algo()
        # Store initial weights
        self._init_weights = self.algo.get_weights()

    def train(self) -> ResultDict:
        # TODO config ability to debug remote actors
        # import debugpy, os
        # debugpy.listen(("127.0.0.1", 5678))
        # print(f"[debugpy] worker pid={os.getpid()} listening on 5678")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()
        # TODO mapping result
        return self.algo.train()

    def evaluate(self) -> ResultDict:
        return self.algo.evaluate()

    def get_metrics(self):
        reduced = self.algo.metrics.reduce()
        full = self.algo.metrics.peek((), default={})
        return {
            "reduced": reduced,
            "full": full,
        }

    def compute_actions(
        self,
        policy_id: str,
        obs_batch: np.ndarray,
    ) -> np.ndarray:
        """
        obs_batch: [B, obs_dim]
        returns:  [B, act_dim]
        """
        try:
            module = self.algo.get_module(policy_id)
            out = module.forward_inference({"obs": obs_batch})
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(out["action_dist_inputs"])
            return dist.sample().cpu().numpy()
        except Exception:
            policy = self.algo.get_policy(policy_id)
            actions = []
            for obs in obs_batch:
                a, _, _ = policy.compute_single_action(obs, explore=False)
                actions.append(a)
            return np.asarray(actions)

    def reset(self):
        """Reset to initial weights."""
        # TODO verify reset is using the same seed 
        self.algo.set_weights(self._init_weights)

    def stop(self):
        self.algo.stop()
