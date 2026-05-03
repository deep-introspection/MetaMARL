from __future__ import annotations

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict


@ray.remote(num_cpus=1)
class PolicyActor:
    """Ray remote actor that exclusively owns and operates an RLlib ``Algorithm``.

    The ``Algorithm`` object is constructed inside this actor and never
    serialised or moved to another process.  All interactions with the
    algorithm (training, evaluation, weight access) are mediated through
    Ray remote method calls, which keeps the main driver process free of
    RLlib's internal thread pools and GPU contexts.

    The actor is decorated with ``@ray.remote(num_cpus=1)`` so that Ray
    reserves a dedicated CPU for it, preventing resource contention with
    env-runner workers.

    Parameters
    ----------
    algo_config : AlgorithmConfig
        A fully configured (and optionally frozen) RLlib ``AlgorithmConfig``
        instance.  ``build_algo()`` is called inside ``__init__`` to
        construct the ``Algorithm`` and all its sub-actors.

    Attributes
    ----------
    algo_config : AlgorithmConfig
        The config stored for use by ``reset()``.
    algo : Algorithm
        The live RLlib algorithm instance.
    _init_weights : dict
        Snapshot of the initial policy weights taken immediately after
        construction, used to restore the policy to its random initialisation
        on ``reset()``.
    """

    def __init__(self, algo_config: AlgorithmConfig):
        # Store config for reset capability
        self.algo_config = algo_config
        # Build Algorithm INSIDE actor (critical)
        self.algo: Algorithm = algo_config.build_algo()
        # Store initial weights
        self._init_weights = self.algo.get_weights()

    def train(self) -> ResultDict:
        """Execute one RLlib training iteration.

        Calls ``Algorithm.train()`` which collects rollouts from all
        env-runner workers, runs the learner update, and computes metrics.

        Returns
        -------
        ResultDict
            RLlib result dictionary containing training metrics such as
            ``episode_return_mean``, policy losses, and env-step counts.
        """
        # TODO config ability to debug remote actors
        # import debugpy, os
        # debugpy.listen(("127.0.0.1", 5678))
        # print(f"[debugpy] worker pid={os.getpid()} listening on 5678")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()
        # TODO mapping result
        return self.algo.train()

    def evaluate(self) -> ResultDict:
        """Run one evaluation pass using the algorithm's built-in evaluator.

        Returns
        -------
        ResultDict
            RLlib result dictionary containing evaluation metrics.
        """
        return self.algo.evaluate()

    def get_metrics(self):
        """Retrieve both reduced and full metric snapshots from the algorithm.

        Uses RLlib's ``MetricsLogger`` to read the current state without
        triggering a training step.  ``reduce()`` collapses per-worker
        metrics into a single dict; ``peek()`` returns the raw accumulated
        values.

        Returns
        -------
        dict
            Dictionary with two keys:

            ``"reduced"``
                Scalar summary metrics after aggregation across workers.
            ``"full"``
                Raw (un-reduced) metric tree from the logger.
        """
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
        """Sample actions for a batch of observations using a specific policy.

        Attempts the new ``RLModule`` inference API first; falls back to the
        classic per-observation ``Policy.compute_single_action`` loop when
        the module is not available.

        Parameters
        ----------
        policy_id : str
            ID of the policy (or RLModule) to use for inference.
        obs_batch : np.ndarray
            Array of shape ``[B, obs_dim]`` containing the observations for
            which actions should be computed.

        Returns
        -------
        np.ndarray
            Array of shape ``[B, act_dim]`` containing sampled actions.
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
        self.algo.set_weights(self._init_weights)

    def stop(self):
        """Shut down the algorithm and release all associated resources.

        Calls ``Algorithm.stop()``, which terminates all env-runner and
        learner worker Ray actors spawned by this algorithm.
        """
        self.algo.stop()
