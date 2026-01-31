from __future__ import annotations

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


@ray.remote(num_cpus=1)
class PolicyActor:
    """
    Owns the RLlib Algorithm.
    The Algorithm never leaves this actor.
    """

    def __init__(self, algo_config: AlgorithmConfig):
        # Build Algorithm INSIDE actor (critical)
        self.algo: Algorithm = algo_config.build()


    def train(self):
        return self.algo.train()


    def compute_action(self, policy_id: str, obs):
        try:
            # RLModule API (preferred)
            module = self.algo.get_module(policy_id)
            out = module.forward_inference(
                {"obs": np.asarray(obs)[None, ...]}
            )
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(out["action_dist_inputs"])
            return dist.sample().cpu().numpy()[0]

        except Exception:
            # Legacy Policy API fallback
            policy = self.algo.get_policy(policy_id)
            action, _, _ = policy.compute_single_action(
                obs, explore=False
            )
            return action

    def stop(self):
        self.algo.stop()
