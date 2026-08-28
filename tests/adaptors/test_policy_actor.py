"""Unit tests for ``core.adaptors.ray.policy_actor.PolicyActor``.

``PolicyActor`` is a Ray actor that owns the RLlib ``Algorithm`` of the inner
optimizer. These tests bypass the Ray runtime by instantiating the undecorated
class (``PolicyActor.__ray_metadata__.modified_class``) with a mocked
``AlgorithmConfig`` whose ``build_algo()`` returns a mocked ``Algorithm``. Only
the forwarding logic, the initial-weights bookkeeping and the two action
computation paths are exercised; no algorithm is ever trained.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.adaptors.ray.policy_actor import PolicyActor

# The plain Python class behind the ``@ray.remote`` decorator.
PolicyActorClass = PolicyActor.__ray_metadata__.modified_class


def make_actor(weights=None):
    """Return ``(actor, algo_config, algo)`` with a mocked RLlib stack.

    ``build_algo`` always returns the same mocked ``Algorithm`` so that the
    tests can assert on calls made either at construction or after ``reset``.
    """
    algo = MagicMock(name="algo")
    algo.get_weights.return_value = (
        weights if weights is not None else {"policy": np.zeros(3, dtype=np.float32)}
    )
    algo_config = MagicMock(name="algo_config")
    algo_config.build_algo.return_value = algo
    actor = PolicyActorClass(algo_config)
    return actor, algo_config, algo


@pytest.mark.unit
def test_init_builds_algorithm_and_stores_initial_weights():
    weights = {"policy": np.arange(3, dtype=np.float32)}
    actor, algo_config, algo = make_actor(weights)

    algo_config.build_algo.assert_called_once_with()
    assert actor.algo is algo
    assert actor.algo_config is algo_config
    assert actor._init_weights is weights


@pytest.mark.unit
def test_train_and_evaluate_forward_to_algorithm():
    actor, _, algo = make_actor()
    algo.train.return_value = {"training_iteration": 1}
    algo.evaluate.return_value = {"evaluation": {}}

    assert actor.train() == {"training_iteration": 1}
    assert actor.evaluate() == {"evaluation": {}}
    algo.train.assert_called_once_with()
    algo.evaluate.assert_called_once_with()


@pytest.mark.unit
def test_get_metrics_returns_reduced_and_full_trees():
    actor, _, algo = make_actor()
    algo.metrics.reduce.return_value = {"reduced": 1}
    algo.metrics.peek.return_value = {"full": 2}

    out = actor.get_metrics()

    assert out == {"reduced": {"reduced": 1}, "full": {"full": 2}}
    algo.metrics.reduce.assert_called_once_with()
    algo.metrics.peek.assert_called_once_with((), default={})


@pytest.mark.unit
def test_compute_actions_uses_rl_module_inference_path():
    actor, _, algo = make_actor()
    obs_batch = np.ones((2, 4), dtype=np.float32)
    expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    module = algo.get_module.return_value
    module.forward_inference.return_value = {"action_dist_inputs": "logits"}
    dist = module.get_inference_action_dist_cls.return_value.from_logits.return_value
    dist.sample.return_value.cpu.return_value.numpy.return_value = expected

    actions = actor.compute_actions("fisher", obs_batch)

    np.testing.assert_array_equal(actions, expected)
    algo.get_module.assert_called_once_with("fisher")
    module.forward_inference.assert_called_once()
    assert module.forward_inference.call_args.args[0]["obs"] is obs_batch
    module.get_inference_action_dist_cls.return_value.from_logits.assert_called_once_with(
        "logits"
    )
    algo.get_policy.assert_not_called()


@pytest.mark.unit
def test_compute_actions_falls_back_to_policy_api():
    actor, _, algo = make_actor()
    algo.get_module.side_effect = AttributeError("no RLModule on this stack")
    policy = algo.get_policy.return_value
    policy.compute_single_action.side_effect = [
        (np.array([1.0]), None, {}),
        (np.array([2.0]), None, {}),
    ]
    obs_batch = np.zeros((2, 3), dtype=np.float32)

    actions = actor.compute_actions("fisher", obs_batch)

    np.testing.assert_array_equal(actions, np.array([[1.0], [2.0]]))
    algo.get_policy.assert_called_once_with("fisher")
    assert policy.compute_single_action.call_count == 2
    for call in policy.compute_single_action.call_args_list:
        assert call.kwargs == {"explore": False}


@pytest.mark.unit
def test_reset_rebuilds_algorithm_and_restores_initial_weights(caplog):
    init_weights = {"policy": np.arange(3, dtype=np.float32)}
    actor, algo_config, algo = make_actor(init_weights)
    # The rebuilt algorithm reports different weights before restoration and
    # the initial ones afterwards; only the latter must be hashed and logged.
    algo.get_weights.side_effect = [init_weights, init_weights]

    with caplog.at_level("INFO", logger="core.adaptors.ray.policy_actor"):
        actor.reset()

    assert algo_config.build_algo.call_count == 2
    algo.set_weights.assert_called_once_with(init_weights)
    assert actor.algo is algo
    assert "Initial policy weight hash" in caplog.text


@pytest.mark.unit
def test_stop_forwards_to_algorithm():
    actor, _, algo = make_actor()
    actor.stop()
    algo.stop.assert_called_once_with()
