from __future__ import annotations

import logging
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict
from core.adaptors.ray.utils import hash_weights

logger = logging.getLogger(__name__)

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

    def _stop_algo(self):
        """Stop the Algorithm INCLUDING its local APPO learner thread.

        Algorithm.stop() calls LearnerGroup.shutdown(), but shutdown() only
        terminates REMOTE learner backends; with num_learners=0 the learner is
        local and its _LearnerThread (a busy polling loop) survives forever.
        Each per-generation rebuild then leaks one such thread, GIL contention
        grows, and train-iteration time climbs linearly (measured 0.24 s ->
        6.5 s over 121 generations without any stop(); still +1 thread/gen
        with stop() alone). Flagging thread.stopped exits its run() loop.
        """
        learner = getattr(self.algo.learner_group, "_learner", None)
        thread = getattr(learner, "_learner_thread", None)
        if thread is not None:
            thread.stopped = True
            # Setting `stopped` is not enough: the batch-wait loops
            # (CircularBuffer.sample() and the deque path in
            # _LearnerThread.step()) spin on an empty buffer at 10 kHz WITHOUT
            # re-checking `stopped`, so a stopped thread never leaves the
            # wait. Feed one dummy entry to unblock it: step() re-checks
            # `stopped` right after the dequeue and returns before touching
            # the batch.
            try:
                in_queue = thread._in_queue
                if hasattr(in_queue, "add"):
                    in_queue.add(object())
                else:
                    in_queue.append(object())
            except Exception:
                logger.warning(
                    "[PPO] could not unblock _LearnerThread", exc_info=True
                )
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "[PPO] _LearnerThread still alive after join timeout"
                )
        self.algo.stop()

    def reset(self):
        """Reset to initial weights."""
        # TODO verify reset is using the same seed
        # self.algo.set_weights(self._init_weights)
        self._stop_algo()
        self.algo = self.algo_config.build_algo()
        # TODO tie the init_weights with seeding
        self.algo.set_weights(self._init_weights)

        weights = self.algo.get_weights()
        logger.info(
            "[PPO] Initial policy weight hash: %s",
            hash_weights(weights),
        )

    def stop(self):
        self._stop_algo()
