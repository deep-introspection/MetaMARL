"""Unit tests for ``core.adaptors.ray.utils``.

The metric getters are fed hand-built RLlib result dictionaries in both the
new API stack layout (``env_runners/...``) and the classic one (top-level
``episode_reward_mean``, ``timesteps_total``, ``info/learner/...``).
``hash_weights`` is checked for determinism, key-order independence and
sensitivity to values across tensors, arrays and plain leaves. The
``build_*`` functions are fed the same kind of hand-built results and their
typed output (``RolloutSchema``, ``PerformanceSchema``, ``LearnerSchema``) is
compared field by field, including the derived learner quantities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.adaptors.ray.schema import (
    LearnerSchema,
    PerformanceSchema,
    RolloutSchema,
)
from core.adaptors.ray.utils import (
    _get_env,
    build_episode_aggregate,
    build_learner,
    build_performance,
    build_rollout,
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
    hash_weights,
)
from core.envs.schema import EpisodeRolloutSchema

# ---------------------------------------------------------------------------
# _get_env
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "result, expected",
    [
        ({}, {}),
        ({"env_runners": None}, {}),
        ({"env_runners": {}}, {}),
        ({"env_runners": {"a": 1}}, {"a": 1}),
    ],
)
def test_get_env(result, expected):
    assert _get_env(result) == expected


# ---------------------------------------------------------------------------
# get_episode_return_mean
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_episode_return_prefers_new_stack_key():
    result = {
        "env_runners": {"episode_return_mean": np.float32(1.5)},
        "episode_reward_mean": 99.0,
    }
    assert get_episode_return_mean(result) == 1.5


@pytest.mark.unit
def test_episode_return_legacy_top_level():
    assert get_episode_return_mean({"episode_reward_mean": 2.0}) == 2.0


@pytest.mark.unit
def test_episode_return_legacy_inside_env_runners():
    assert get_episode_return_mean({"env_runners": {"episode_reward_mean": 3}}) == 3.0


@pytest.mark.unit
def test_episode_return_defaults_to_zero():
    assert get_episode_return_mean({}) == 0.0
    assert (
        get_episode_return_mean({"env_runners": {"episode_return_mean": "nan?"}}) == 0.0
    )


@pytest.mark.unit
def test_episode_return_zero_top_level_falls_through_to_env_runner_value():
    # ``or`` treats 0.0 as missing: the env_runners legacy key wins.
    result = {"episode_reward_mean": 0.0, "env_runners": {"episode_reward_mean": 4.0}}
    assert get_episode_return_mean(result) == 4.0


@pytest.mark.unit
def test_episode_return_negative_value_is_kept():
    assert get_episode_return_mean({"episode_reward_mean": -3.5}) == -3.5


# ---------------------------------------------------------------------------
# get_env_steps
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_steps_new_stack():
    result = {
        "env_runners": {
            "num_env_steps_sampled": 200,
            "num_env_steps_sampled_lifetime": np.int64(4000),
        }
    }
    assert get_env_steps(result) == (200, 4000)


@pytest.mark.unit
def test_env_steps_legacy_fallback():
    result = {"timesteps_this_iter": 10.0, "timesteps_total": "30"}
    assert get_env_steps(result) == (10, 30)


@pytest.mark.unit
def test_env_steps_mixed_and_missing():
    result = {"env_runners": {"num_env_steps_sampled": 5}, "timesteps_total": 50}
    assert get_env_steps(result) == (5, 50)
    assert get_env_steps({}) == (0, 0)
    assert get_env_steps({"env_runners": {"num_env_steps_sampled": "x"}}) == (0, 0)


@pytest.mark.unit
def test_env_steps_truncates_floats():
    result = {"timesteps_this_iter": 7.9, "timesteps_total": 12.1}
    assert get_env_steps(result) == (7, 12)


# ---------------------------------------------------------------------------
# get_policy_loss_if_present
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_policy_loss_averages_across_policies():
    result = {
        "info": {
            "learner": {
                "p0": {"learner_stats": {"policy_loss": 1.0}},
                "p1": {"learner_stats": {"policy_loss": np.float32(3.0)}},
                "p2": {"learner_stats": {}},
                "p3": None,
                "p4": {"learner_stats": None},
                "p5": {"learner_stats": {"policy_loss": "bad"}},
            }
        }
    }
    assert get_policy_loss_if_present(result) == 2.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [
        {},
        {"info": None},
        {"info": {}},
        {"info": {"learner": None}},
        {"info": {"learner": {}}},
        {"info": {"learner": ["not", "a", "dict"]}},
        {"info": {"learner": {"p0": {"learner_stats": {"other": 1}}}}},
        {"learners": {"m0": {"policy_loss": 0.5}}},  # new API stack: not read
    ],
)
def test_policy_loss_nan_when_absent(result):
    assert math.isnan(get_policy_loss_if_present(result))


# ---------------------------------------------------------------------------
# hash_weights
# ---------------------------------------------------------------------------


def _weights(scale: float = 1.0) -> dict:
    return {
        "policy": {
            "fc.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale,
            "fc.bias": np.array([0.1, 0.2]) * scale,
        },
        "meta": {"lr": 0.01 * scale, "name": "adam"},
    }


@pytest.mark.unit
def test_hash_weights_is_deterministic_hex_sha256():
    h1 = hash_weights(_weights())
    h2 = hash_weights(_weights())
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)


@pytest.mark.unit
def test_hash_weights_independent_of_dict_insertion_order():
    a = {"x": np.ones(2), "y": np.zeros(2)}
    b = {"y": np.zeros(2), "x": np.ones(2)}
    assert hash_weights(a) == hash_weights(b)


@pytest.mark.unit
def test_hash_weights_changes_with_values_and_keys():
    base = hash_weights(_weights())
    assert hash_weights(_weights(scale=2.0)) != base
    renamed = _weights()
    renamed["policy"]["fc.w"] = renamed["policy"].pop("fc.weight")
    assert hash_weights(renamed) != base


@pytest.mark.unit
def test_hash_weights_tensor_and_equivalent_array_hash_alike():
    t = torch.tensor([1.0, 2.0], dtype=torch.float64)
    a = np.array([1.0, 2.0], dtype=np.float64)
    assert hash_weights(t) == hash_weights(a)
    # Non-contiguous inputs are made contiguous before hashing.
    t2 = torch.arange(4, dtype=torch.float64).reshape(2, 2).t()
    a2 = np.arange(4, dtype=np.float64).reshape(2, 2).T
    assert hash_weights(t2) == hash_weights(a2)
    assert hash_weights(t2) != hash_weights(
        torch.arange(4, dtype=torch.float64).reshape(2, 2)
    )


@pytest.mark.unit
def test_hash_weights_plain_leaves_use_repr():
    assert hash_weights(1) != hash_weights(1.0)
    assert hash_weights("a") == hash_weights("a")
    assert hash_weights(None) != hash_weights({})
    assert hash_weights({"k": 1}) != hash_weights({"j": 1})


# ---------------------------------------------------------------------------
# build_episode_aggregate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_episode_aggregate_reads_env_runner_stats():
    result = {
        "env_runners": {
            "episode_return_mean": np.float32(1.5),
            "episode_return_min": -2.0,
            "episode_return_max": 4,
            "episode_len_mean": 10.5,
            "episode_len_min": 8,
            "episode_len_max": 12,
            "num_episodes": 3,
            "num_episodes_lifetime": 30,
        }
    }
    agg = build_episode_aggregate(result)

    assert isinstance(agg, EpisodeRolloutSchema)
    assert agg.reward_mean == 1.5
    assert agg.reward_min == -2.0
    assert agg.reward_max == 4.0
    assert (agg.episode_len_mean, agg.episode_len_min, agg.episode_len_max) == (
        10.5,
        8.0,
        12.0,
    )
    assert (agg.num_episodes, agg.num_episodes_lifetime) == (3, 30)
    # RLlib provides no aggregate-level totals or terminal statistics.
    assert agg.reward_total is None and agg.reward_terminal is None
    assert agg.value_terminal is None and agg.value_penultimate is None
    assert agg.by_agent == {}


@pytest.mark.unit
@pytest.mark.parametrize("result", [{}, {"env_runners": None}, {"env_runners": {}}])
def test_build_episode_aggregate_missing_block_gives_none_fields(result):
    agg = build_episode_aggregate(result)
    assert agg.reward_mean is None and agg.num_episodes is None


@pytest.mark.unit
def test_build_episode_aggregate_rejects_non_finite_values():
    result = {
        "env_runners": {"episode_return_mean": float("nan"), "episode_len_mean": "x"}
    }
    agg = build_episode_aggregate(result)
    assert agg.reward_mean is None and agg.episode_len_mean is None


# ---------------------------------------------------------------------------
# build_performance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_performance_sums_agent_steps_and_reads_timers():
    result = {
        "env_runners": {
            "num_env_steps_sampled": 100,
            "num_env_steps_sampled_lifetime": np.int64(700),
            "num_agent_steps_sampled": {"a:0": 60, "a:1": np.float32(40)},
            "num_agent_steps_sampled_lifetime": {"a:0": 400, "a:1": "bad"},
            "num_env_steps_sampled_lifetime_throughput": {
                "throughput_since_last_reduce": 55.5,
                "throughput_since_last_restore": 11.0,
            },
            "weights_seq_no": 4,
        },
        "timers": {
            "training_iteration": 2.5,
            "training_step": 2.0,
            "sample": 1.0,
            "learner_update_timer": 0.5,
        },
    }
    perf = build_performance(result)

    assert isinstance(perf, PerformanceSchema)
    assert perf.env_steps_this_iter == 100.0
    assert perf.env_steps_lifetime == 700.0
    assert perf.agent_steps_this_iter_sum == 100.0
    # Unconvertible per-agent entries count as zero.
    assert perf.agent_steps_lifetime_sum == 400.0
    assert perf.env_steps_throughput == 55.5
    assert perf.training_iteration_s == 2.5
    assert perf.training_step_s == 2.0
    assert perf.sample_s == 1.0
    assert perf.learner_update_s == 0.5
    assert perf.weights_seq_no == 4.0


@pytest.mark.unit
def test_build_performance_throughput_falls_back_to_since_restore():
    env = {
        "num_env_steps_sampled_lifetime_throughput": {
            "throughput_since_last_reduce": None,
            "throughput_since_last_restore": 11.0,
        }
    }
    assert build_performance({"env_runners": env}).env_steps_throughput == 11.0

    # Documented oddity: a zero ``since_last_reduce`` throughput is treated as
    # missing by the ``or`` chain and the restore-window value wins instead.
    env["num_env_steps_sampled_lifetime_throughput"]["throughput_since_last_reduce"] = (
        0.0
    )
    assert build_performance({"env_runners": env}).env_steps_throughput == 11.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "env",
    [
        {},
        {
            "num_env_steps_sampled_lifetime_throughput": 3.0,
            "num_agent_steps_sampled": 12,
            "num_agent_steps_sampled_lifetime": [1, 2],
        },
    ],
)
def test_build_performance_non_dict_blocks_give_none(env):
    perf = build_performance({"env_runners": env, "timers": None})
    assert perf.env_steps_throughput is None
    assert perf.agent_steps_this_iter_sum is None
    assert perf.agent_steps_lifetime_sum is None
    assert perf.training_iteration_s is None


# ---------------------------------------------------------------------------
# build_rollout
# ---------------------------------------------------------------------------


def _episode(mechanism_id, seed, reward_mean=0.0) -> EpisodeRolloutSchema:
    return EpisodeRolloutSchema(
        mechanism_id=mechanism_id, seed=seed, reward_mean=reward_mean
    )


@pytest.mark.unit
def test_build_rollout_groups_episodes_by_mechanism_then_seed():
    e00 = _episode(0, 11, 1.0)
    e01 = _episode(0, 22, 2.0)
    e10 = _episode(1, 11, 3.0)
    e00b = _episode(0, 11, 4.0)
    result = {
        "env_runners": {
            "episode_return_mean": 2.5,
            "by_episode": {
                "env=0|m=0|ps=11|ss=11": e00,
                "env=1|m=0|ps=22|ss=22": e01,
                "env=2|m=1|ps=11|ss=11": e10,
                "env=0|m=0|ps=11|ss=11#2": e00b,
            },
        }
    }
    rollout = build_rollout(result)

    assert isinstance(rollout, RolloutSchema)
    assert rollout.aggregate.reward_mean == 2.5
    assert set(rollout.by_mechanism) == {"0", "1"}
    mech0 = rollout.by_mechanism["0"]
    assert set(mech0.by_seed) == {"11", "22"}
    assert mech0.by_seed["11"].by_episode == {
        "env=0|m=0|ps=11|ss=11": e00,
        "env=0|m=0|ps=11|ss=11#2": e00b,
    }
    assert mech0.by_seed["22"].by_episode == {"env=1|m=0|ps=22|ss=22": e01}
    assert rollout.by_mechanism["1"].by_seed["11"].by_episode == {
        "env=2|m=1|ps=11|ss=11": e10
    }
    # The episode objects are filed as-is, not copied.
    assert mech0.by_seed["11"].by_episode["env=0|m=0|ps=11|ss=11"] is e00


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [{}, {"env_runners": {"by_episode": None}}, {"env_runners": {"by_episode": {}}}],
)
def test_build_rollout_without_episodes_is_empty(result):
    rollout = build_rollout(result)
    assert rollout.by_mechanism == {}
    assert rollout.aggregate.reward_mean is None


@pytest.mark.unit
def test_build_rollout_stringifies_missing_identity_as_none_key():
    # Documented oddity: episodes without ``mechanism_id`` / ``seed`` are not
    # rejected; they land under the literal ``"None"`` keys.
    result = {"env_runners": {"by_episode": {"ep": _episode(None, None)}}}
    rollout = build_rollout(result)
    assert list(rollout.by_mechanism) == ["None"]
    assert list(rollout.by_mechanism["None"].by_seed) == ["None"]


# ---------------------------------------------------------------------------
# build_learner
# ---------------------------------------------------------------------------


def _learner_result(**overrides):
    stats = {
        "module_train_batch_size_mean": 128,
        "total_loss": 0.9,
        "policy_loss": 0.25,
        "entropy": 0.8,
        "curr_entropy_coeff": 0.01,
        "kl": 0.02,
        "curr_kl_coeff": 0.2,
        "vf_loss": 0.4,
        "value_mean": 1.1,
        "value_target": 1.2,
        "gradients_default_optimizer_global_norm": 3.0,
        "grad_gnorm": 2.0,
        "gradient_noise": 0.05,
        "diff_num_grad_updates_vs_sampler_policy": 1.0,
        "num_module_steps_trained_lifetime_throughput": {
            "throughput_since_last_reduce": 10.0,
            "throughput_since_last_restore": 5.0,
        },
        "ignored_nan": float("nan"),
        "ignored_text": "n/a",
    }
    stats.update(overrides)
    return {
        "learners": {
            "fisher_m0_s11": stats,
            "__all_modules__": {"learner_thread_in_queue_wait_timer": 0.5},
        },
        "learner_group": {"actor_manager_num_outstanding_async_reqs": 2},
        "mean_num_training_step_calls_since_last_synch_worker_weights": 3,
    }


@pytest.mark.unit
def test_build_learner_copies_stats_and_derives_quantities():
    learner = build_learner(_learner_result())

    assert isinstance(learner, LearnerSchema)
    policy = learner.by_policy["fisher_m0_s11"]
    assert policy.batch_size == 128
    assert policy.total_loss == 0.9
    assert policy.policy_loss == 0.25
    assert policy.policy_entropy == 0.8
    assert policy.policy_entropy_coeff == 0.01
    assert policy.policy_relative_entropy == pytest.approx(0.8 / 0.01)
    assert policy.entropy_pressure == pytest.approx(0.8 * 0.01)
    assert policy.policy_kl == 0.02 and policy.policy_kl_coeff == 0.2
    assert policy.value_loss == 0.4
    assert policy.value_mean == 1.1 and policy.value_target == 1.2
    assert policy.gradient_norm == 3.0
    assert policy.gradient_noise == 0.05
    # lag1 + training calls since sync + outstanding reqs + queue wait.
    assert policy.sample_staleness == pytest.approx(1.0 + 3 + 2 + 0.5)
    assert policy.residual_variance is None


@pytest.mark.unit
def test_build_learner_includes_all_modules_pseudo_policy():
    # Documented oddity: the ``__all_modules__`` aggregate block is iterated
    # like a real learner module and becomes an (almost empty) policy entry.
    learner = build_learner(_learner_result())
    assert set(learner.by_policy) == {"fisher_m0_s11", "__all_modules__"}
    pseudo = learner.by_policy["__all_modules__"]
    assert pseudo.policy_loss is None
    # Its staleness still sums the global lag indicators.
    assert pseudo.sample_staleness == pytest.approx(3 + 2 + 0.5)


@pytest.mark.unit
def test_build_learner_gradient_norm_falls_back_to_grad_gnorm():
    learner = build_learner(
        _learner_result(gradients_default_optimizer_global_norm=None)
    )
    assert learner.by_policy["fisher_m0_s11"].gradient_norm == 2.0
    # Documented oddity: a zero global norm is falsy and falls back as well.
    learner = build_learner(
        _learner_result(gradients_default_optimizer_global_norm=0.0)
    )
    assert learner.by_policy["fisher_m0_s11"].gradient_norm == 2.0


@pytest.mark.unit
def test_build_learner_without_entropy_or_lags_leaves_derived_none():
    result = {"learners": {"p": {"policy_loss": 0.1, "curr_entropy_coeff": 0.01}}}
    policy = build_learner(result).by_policy["p"]
    assert policy.policy_relative_entropy is None
    assert policy.entropy_pressure is None
    assert policy.sample_staleness is None
    assert policy.gradient_norm is None

    # Zero entropy coefficient: ratio undefined, pressure still computed.
    result = {"learners": {"p": {"entropy": 0.5, "curr_entropy_coeff": 0.0}}}
    policy = build_learner(result).by_policy["p"]
    assert policy.policy_relative_entropy is None
    assert policy.entropy_pressure == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("result", [{}, {"learners": None}, {"learners": {}}])
def test_build_learner_without_learners_is_empty(result):
    assert build_learner(result).by_policy == {}


@pytest.mark.unit
def test_build_learner_rejects_fractional_batch_size():
    # ``PolicyLearnerSchema.batch_size`` is an ``int`` field while RLlib reports
    # ``module_train_batch_size_mean`` as a mean; a non-integral mean makes the
    # builder raise instead of rounding. Documented as current behaviour.
    with pytest.raises(ValidationError):
        build_learner(_learner_result(module_train_batch_size_mean=128.5))
