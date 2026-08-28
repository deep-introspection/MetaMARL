"""RLlib ``ResultDict`` -> typed schemas (``core.adaptors.ray.utils``) and ``RayOptimizer._to_logger_payload``."""

import pytest

from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.schema import RaySchema
from core.adaptors.ray.utils import (
    build_episode_aggregate,
    build_learner,
    build_performance,
    build_rollout,
)
from core.envs.schema import EpisodeRolloutSchema
from core.metrics.logger import MetricLogger


def episode(mechanism_id, seed, reward):
    return EpisodeRolloutSchema(
        mechanism_id=mechanism_id, seed=seed, reward_mean=reward
    )


def result_dict(with_eval=False):
    result = {
        "env_runners": {
            "episode_return_mean": 1.5,
            "episode_return_min": 1.0,
            "episode_return_max": 2.0,
            "episode_len_mean": 20.0,
            "episode_len_min": 20.0,
            "episode_len_max": 20.0,
            "num_episodes": 4,
            "num_episodes_lifetime": 8,
            "num_env_steps_sampled": 80,
            "num_env_steps_sampled_lifetime": 160,
            "num_env_steps_sampled_lifetime_throughput": {
                "throughput_since_last_reduce": 50.0
            },
            "num_agent_steps_sampled": {"a": 80, "b": 80},
            "num_agent_steps_sampled_lifetime": {"a": 160, "b": 160},
            "weights_seq_no": 3,
            "by_episode": {
                "e1": episode(0, 100, 1.0),
                "e2": episode(0, 100, 2.0),
                "e3": episode(1, 100, 3.0),
                "e4": episode(1, 200, 4.0),
            },
        },
        "timers": {
            "training_iteration": 1.0,
            "training_step": 0.5,
            "sample": 0.3,
            "learner_update_timer": 0.2,
        },
        "learners": {
            "__all_modules__": {"learner_thread_in_queue_wait_timer": 0.1},
            "p0": {
                "module_train_batch_size_mean": 100,
                "total_loss": 0.5,
                "policy_loss": 0.2,
                "entropy": 1.0,
                "curr_entropy_coeff": 0.01,
                "kl": 0.02,
                "curr_kl_coeff": 0.2,
                "vf_loss": 0.3,
                "gradients_default_optimizer_global_norm": 4.0,
                "diff_num_grad_updates_vs_sampler_policy": 1.0,
                "num_module_steps_trained_lifetime_throughput": {
                    "throughput_since_last_reduce": 7.0
                },
                "non_numeric": "skip",
            },
        },
        "learner_group": {"actor_manager_num_outstanding_async_reqs": 0.0},
        "mean_num_training_step_calls_since_last_synch_worker_weights": 2.0,
    }
    if with_eval:
        result["evaluation"] = {
            k: v for k, v in result.items() if k in ("env_runners", "timers")
        }
    return result


@pytest.mark.unit
def test_episode_aggregate_and_performance():
    agg = build_episode_aggregate(result_dict())
    assert agg.reward_mean == 1.5 and agg.reward_max == 2.0 and agg.num_episodes == 4
    perf = build_performance(result_dict())
    assert perf.env_steps_this_iter == 80 and perf.env_steps_lifetime == 160
    assert (
        perf.agent_steps_this_iter_sum == 160 and perf.agent_steps_lifetime_sum == 320
    )
    assert perf.env_steps_throughput == 50.0 and perf.training_iteration_s == 1.0
    assert perf.learner_update_s == 0.2 and perf.weights_seq_no == 3
    empty = build_performance({})
    assert empty.env_steps_this_iter is None and empty.agent_steps_this_iter_sum is None


@pytest.mark.unit
def test_rollout_groups_by_mechanism_and_seed():
    rollout = build_rollout(result_dict())
    assert set(rollout.by_mechanism) == {"0", "1"}
    assert set(rollout.by_mechanism["0"].by_seed) == {"100"}
    assert set(rollout.by_mechanism["1"].by_seed) == {"100", "200"}
    assert set(rollout.by_mechanism["0"].by_seed["100"].by_episode) == {"e1", "e2"}
    assert rollout.by_mechanism["1"].by_seed["200"].by_episode["e4"].reward_mean == 4.0
    assert rollout.aggregate.reward_mean == 1.5


@pytest.mark.unit
def test_learner_mapping_and_derived_metrics():
    learner = build_learner(result_dict())
    assert set(learner.by_policy) == {"__all_modules__", "p0"}
    p0 = learner.by_policy["p0"]
    assert p0.batch_size == 100 and p0.total_loss == 0.5 and p0.policy_loss == 0.2
    assert p0.policy_entropy == 1.0 and p0.policy_entropy_coeff == 0.01
    assert p0.policy_relative_entropy == pytest.approx(100.0)  # entropy / coeff
    assert p0.entropy_pressure == pytest.approx(0.01)
    assert p0.policy_kl == 0.02 and p0.policy_kl_coeff == 0.2 and p0.value_loss == 0.3
    assert p0.gradient_norm == 4.0
    # staleness = lag1 + mean calls since sync + outstanding reqs + queue wait
    assert p0.sample_staleness == pytest.approx(1.0 + 2.0 + 0.0 + 0.1)


@pytest.mark.unit
def test_ray_payload_train_and_eval_and_logger_ingestion():
    payload = RayOptimizer._to_logger_payload(None, result_dict(with_eval=True))
    assert isinstance(payload, RaySchema)
    assert payload.train.rollout.aggregate.reward_mean == 1.5
    assert (
        payload.eval is not None and payload.eval.performance.env_steps_this_iter == 80
    )
    assert payload.train.learner.by_policy["p0"].total_loss == 0.5

    eval_only = RayOptimizer._to_logger_payload(None, result_dict(), is_eval=True)
    assert (
        eval_only.train is None and eval_only.eval.rollout.aggregate.reward_min == 1.0
    )

    logger = MetricLogger.from_schema(RaySchema)
    for it in (1, 2):
        logger.push(("iter",), it)
        logger.push_data(payload)
    peeked = logger.peek()
    assert peeked.iter == [1, 2]
    assert peeked.train.rollout.aggregate.reward_mean == [1.5, 1.5]
    assert peeked.train.rollout.by_mechanism["1"].by_seed["200"].by_episode[
        "e4"
    ].reward_mean == [4.0, 4.0]
    reduced = logger.reduce()
    assert reduced.train.rollout.aggregate.reward_mean == 1.5 and reduced.iter == 2
