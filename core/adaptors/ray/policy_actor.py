from __future__ import annotations

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import StateDict

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

    def evaluate(self):
        return self.algo.evaluate()
    
    def set_state(self, state: StateDict):
        return self.algo.set_state(state)
    
    def get_state(self):
        return self.algo.get_state()
    

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

    def stop(self):
        self.algo.stop()
