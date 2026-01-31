from __future__ import annotations

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv


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
        # self.algo.evaluation_config = (
        #     AlgorithmConfig()
        #     .resources(
        #     num_cpus_for_main_process=1,
        #     )
        #     .framework(
        #         framework="torch",
        #     )
        #     .api_stack(
        #         enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        #     )
        #     .environment(
        #         env=FisheryRegulatedEnv,
        #         env_config={
        #             "ecology_cfg": {
        #                 "algae_init": 1.0,
        #                 "fish_init": 0.5,
        #                 "max_fish": 2.0,
        #                 "max_algae": 2.0,
        #                 "alpha": 0.5,
        #                 "beta": 0.1,
        #                 "delta": 0.1,
        #                 "gamma": 0.5,
        #                 "dt": 0.01,
        #                 # "horizon": 200,  
        #             },
        #             "seed": 0},
        #         horizon=200 # must be the same as regulator 200
        #     )
        #     .env_runners(
        #         num_env_runners=1,
        #         num_cpus_per_env_runner=1,
        #         num_gpus_per_env_runner=0,
        #         num_envs_per_env_runner=16, # batch evaluated mechanism or population size for ES 16
        #         rollout_fragment_length=200,  # must be same as env horizon 200
        #         batch_mode="complete_episodes",
        #     )
        #     .training(
        #         gamma=0.99,
        #         lr=3e-4,
        #         train_batch_size=3200, #3200
        #         minibatch_size=512, #512
        #     )
            
        # )
        return self.algo.evaluate()


    # def compute_action(self, policy_id: str, obs):
    #     try:
    #         # RLModule API (preferred)
    #         module = self.algo.get_module(policy_id)
    #         out = module.forward_inference(
    #             {"obs": np.asarray(obs)[None, ...]}
    #         )
    #         dist_cls = module.get_inference_action_dist_cls()
    #         dist = dist_cls.from_logits(out["action_dist_inputs"])
    #         return dist.sample().cpu().numpy()[0]

    #     except Exception:
    #         # Legacy Policy API fallback
    #         policy = self.algo.get_policy(policy_id)
    #         action, _, _ = policy.compute_single_action(
    #             obs, explore=False
    #         )
    #         return action

    
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
