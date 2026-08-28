"""Unit tests for ``core.adaptors.ray.optimizer.RayOptimizer``.

``RayOptimizer`` forwards ``run`` / ``evaluate`` / ``reset`` / ``stop`` to a
``PolicyActor``, keeps light bookkeeping (inner iteration counter, per
iteration return and loss) and feeds every RLlib result into a ``MetricLogger``
built from ``RaySchema``. The actor is replaced here by an in-memory stub
exposing the ``<method>.remote()`` call shape, ``ray.get`` is patched to the
identity, and the frozen ``RayOptimizerConfig`` is replaced by a namespace
carrying only the attributes the optimizer reads. The logger content is read
back through ``stop()``, which reduces it into a ``RaySchema``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import ray
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

import core.adaptors.ray.policy_actor as policy_actor_module
from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.schema import RaySchema
from core.envs.schema import EpisodeRolloutSchema


class StubPolicyActor:
    """Stand-in for a ``PolicyActor`` handle.

    Every remote method returns ``train_result`` (for ``train``),
    ``eval_result`` (for ``evaluate``) or ``None`` and appends its name to
    ``calls``.
    """

    def __init__(self, algo_config, train_result=None, eval_result=None):
        self.algo_config = algo_config
        self.calls: list[str] = []
        self.train_result = train_result if train_result is not None else {}
        self.eval_result = eval_result if eval_result is not None else {}
        self.train = SimpleNamespace(remote=lambda: self._call("train"))
        self.evaluate = SimpleNamespace(remote=lambda: self._call("evaluate"))
        self.reset = SimpleNamespace(remote=lambda: self._call("reset"))
        self.stop = SimpleNamespace(remote=lambda: self._call("stop"))

    def _call(self, name):
        self.calls.append(name)
        if name == "train":
            return self.train_result
        if name == "evaluate":
            return self.eval_result
        return None


class StubPolicyActorClass:
    """Mimics the ``PolicyActor.remote(...)`` factory of a Ray actor class."""

    def __init__(self, train_result=None, eval_result=None):
        self.train_result = train_result
        self.eval_result = eval_result
        self.instances: list[StubPolicyActor] = []

    def remote(self, algo_config):
        actor = StubPolicyActor(algo_config, self.train_result, self.eval_result)
        self.instances.append(actor)
        return actor


def make_config(
    *,
    evaluation_duration=10,
    rollout_fragment_length=5,
    num_envs_per_env_runner=6,
    seeds=(1, 2),
):
    """Namespace carrying the attributes ``RayOptimizer`` reads from its config."""
    eval_cfg = {}
    if rollout_fragment_length is not None:
        eval_cfg["rollout_fragment_length"] = rollout_fragment_length
    rllib_cfg = SimpleNamespace(
        evaluation_duration=evaluation_duration,
        evaluation_config=eval_cfg,
        num_envs_per_env_runner=num_envs_per_env_runner,
    )
    return SimpleNamespace(
        env=None,
        stats_cls_lookup=DEFAULT_STATS_CLS_LOOKUP,
        rllib_cfg=rllib_cfg,
        seeds=list(seeds),
    )


def train_result(
    *, return_mean=1.5, iteration=7, steps=100, lifetime=700, evaluation=None
):
    result = {
        "training_iteration": iteration,
        "env_runners": {
            "episode_return_mean": return_mean,
            "episode_return_min": return_mean - 1,
            "episode_return_max": return_mean + 1,
            "num_env_steps_sampled": steps,
            "num_env_steps_sampled_lifetime": lifetime,
            "by_episode": {
                "env=0|m=0|ps=1|ss=1": EpisodeRolloutSchema(
                    mechanism_id=0, seed=1, reward_mean=return_mean
                )
            },
        },
        "learners": {"fisher_m0_s1": {"policy_loss": 0.25, "entropy": 0.5}},
        "timers": {"training_iteration": 2.0},
        "info": {"learner": {"p0": {"learner_stats": {"policy_loss": 0.25}}}},
    }
    if evaluation is not None:
        result["evaluation"] = evaluation
    return result


@pytest.fixture
def stub_actor_class(monkeypatch):
    """Patch ``PolicyActor`` (imported lazily inside ``__init__``) and ``ray.get``."""
    stub = StubPolicyActorClass()
    monkeypatch.setattr(policy_actor_module, "PolicyActor", stub)
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    return stub


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_init_spawns_actor_logger_and_derives_eval_episodes(stub_actor_class):
    cfg = make_config(evaluation_duration=10, rollout_fragment_length=5)

    opt = RayOptimizer(config=cfg)

    assert opt.eval_episodes == 2
    assert len(stub_actor_class.instances) == 1
    assert opt.policy_actor is stub_actor_class.instances[0]
    assert opt.policy_actor.algo_config is cfg.rllib_cfg
    assert opt._inner_iter == 0 and opt._es_round == 0
    assert opt._training_rewards == [] and opt._training_losses == []
    assert opt.logger._schema is RaySchema
    assert isinstance(opt.logger.peek(), RaySchema)


@pytest.mark.unit
def test_init_raises_when_rollout_fragment_length_missing(stub_actor_class):
    with pytest.raises(TypeError):
        RayOptimizer(config=make_config(rollout_fragment_length=None))


@pytest.mark.unit
def test_batch_capacity_divides_envs_by_seeds(stub_actor_class):
    opt = RayOptimizer(config=make_config(num_envs_per_env_runner=6, seeds=(1, 2)))
    assert opt.batch_capacity == 3


@pytest.mark.unit
def test_batch_capacity_raises_without_seeds(stub_actor_class):
    opt = RayOptimizer(config=make_config(seeds=()))
    with pytest.raises(ZeroDivisionError):
        _ = opt.batch_capacity


# --------------------------------------------------------------------------- #
# _to_logger_payload
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_to_logger_payload_train_without_evaluation(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    payload = opt._to_logger_payload(train_result())

    assert isinstance(payload, RaySchema)
    assert payload.eval is None
    assert payload.train.rollout.aggregate.reward_mean == 1.5
    assert list(payload.train.rollout.by_mechanism) == ["0"]
    assert payload.train.learner.by_policy["fisher_m0_s1"].policy_loss == 0.25
    assert payload.train.performance.env_steps_this_iter == 100.0
    assert payload.train.performance.training_iteration_s == 2.0


@pytest.mark.unit
def test_to_logger_payload_train_with_nested_evaluation(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    evaluation = {"env_runners": {"episode_return_mean": 9.0}}
    payload = opt._to_logger_payload(train_result(evaluation=evaluation))

    assert payload.train.rollout.aggregate.reward_mean == 1.5
    assert payload.eval.rollout.aggregate.reward_mean == 9.0
    assert payload.eval.performance.env_steps_this_iter is None

    # A non-dict ``evaluation`` entry is ignored.
    payload = opt._to_logger_payload(train_result(evaluation="pending"))
    assert payload.eval is None


@pytest.mark.unit
def test_to_logger_payload_eval_only(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    result = {
        "env_runners": {"episode_return_mean": 3.0, "num_env_steps_sampled": 40},
        "learners": {"fisher_m0_s1": {"policy_loss": 0.1}},
    }
    payload = opt._to_logger_payload(result, is_eval=True)

    assert payload.train is None
    assert payload.eval.rollout.aggregate.reward_mean == 3.0
    assert payload.eval.performance.env_steps_this_iter == 40.0
    assert not hasattr(payload.eval, "learner")


# --------------------------------------------------------------------------- #
# run / evaluate / reset / stop / save
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_run_tracks_return_steps_loss_and_logs_metrics(stub_actor_class, caplog):
    stub_actor_class.train_result = train_result()
    opt = RayOptimizer(config=make_config())

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        opt.run()

    assert opt.policy_actor.calls == ["train"]
    assert opt._inner_iter == 1
    assert opt._training_rewards == [1.5]
    assert opt._training_losses == [0.25]
    assert "rllib_iter_lifetime=7" in caplog.text
    assert "env_steps_iter=100" in caplog.text
    assert "policy_loss=0.250000" in caplog.text

    # The logger received the inner iteration and the typed payload.
    assert opt.logger.peek_value(("iter",)) == 1
    peeked = opt.logger.peek()
    assert peeked.train.rollout.aggregate.reward_mean == [1.5]
    assert peeked.train.learner.by_policy["fisher_m0_s1"].policy_loss == [0.25]
    assert "0" in peeked.train.rollout.by_mechanism
    assert peeked.eval is None or peeked.eval.rollout.aggregate.reward_mean in (
        None,
        [],
    )


@pytest.mark.unit
def test_run_twice_accumulates_in_logger(stub_actor_class):
    stub_actor_class.train_result = train_result(return_mean=1.0)
    opt = RayOptimizer(config=make_config())
    opt.run()
    opt.policy_actor.train_result = train_result(return_mean=3.0, iteration=8)
    opt.run()

    assert opt._inner_iter == 2
    assert opt._training_rewards == [1.0, 3.0]
    reduced = opt.stop()
    assert opt.policy_actor.calls == ["train", "train", "stop"]
    assert isinstance(reduced, RaySchema)
    assert reduced.iter == 2
    assert reduced.train.rollout.aggregate.reward_mean == pytest.approx(2.0)
    assert reduced.train.rollout.aggregate.reward_min == pytest.approx(0.0)
    assert reduced.train.rollout.aggregate.reward_max == pytest.approx(4.0)
    assert reduced.train.performance.env_steps_lifetime == 700.0


@pytest.mark.unit
def test_run_logs_na_when_loss_absent(stub_actor_class, caplog):
    # New API stack results carry no ``info/learner`` block: the loss is NaN
    # and the summary line prints ``NA`` instead of a number.
    stub_actor_class.train_result = {"env_runners": {"episode_return_mean": 0.0}}
    opt = RayOptimizer(config=make_config())

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        opt.run()
        opt.run()

    assert opt._inner_iter == 2
    assert len(opt._training_losses) == 2
    assert all(math.isnan(loss) for loss in opt._training_losses)
    assert "policy_loss=NA" in caplog.text
    assert "rllib_iter_lifetime=0" in caplog.text


@pytest.mark.unit
def test_evaluate_forwards_to_actor_and_logs_eval_branch(stub_actor_class, caplog):
    stub_actor_class.eval_result = {
        "env_runners": {"episode_return_mean": 4.0, "num_env_steps_sampled": 12}
    }
    opt = RayOptimizer(config=make_config())

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        opt.evaluate()

    assert opt.policy_actor.calls == ["evaluate"]
    assert "Evaluation started" in caplog.text
    assert "Evaluation completed" in caplog.text
    reduced = opt.stop()
    assert reduced.eval.rollout.aggregate.reward_mean == 4.0
    assert reduced.eval.performance.env_steps_this_iter == 12.0
    assert reduced.train.rollout.aggregate.reward_mean is None


@pytest.mark.unit
def test_reset_clears_tracking_logger_and_bumps_es_round(stub_actor_class):
    stub_actor_class.train_result = train_result(return_mean=2.0)
    opt = RayOptimizer(config=make_config())
    opt.run()
    assert opt._training_rewards == [2.0]

    opt.reset()

    assert opt.policy_actor.calls == ["train", "reset"]
    assert opt._inner_iter == 0
    assert opt._es_round == 1
    assert opt._training_rewards == [] and opt._training_losses == []
    reduced = opt.stop()
    assert reduced.iter is None
    assert reduced.train.rollout.aggregate.reward_mean is None
    # Dynamic nodes created by the first run survive the reset, emptied.
    assert (
        reduced.train.rollout.by_mechanism["0"]
        .by_seed["1"]
        .by_episode["env=0|m=0|ps=1|ss=1"]
        .reward_mean
        is None
    )


@pytest.mark.unit
def test_stop_forwards_to_actor_and_returns_reduced_schema(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    reduced = opt.stop()
    assert opt.policy_actor.calls == ["stop"]
    assert isinstance(reduced, RaySchema)
    assert reduced.train is not None and reduced.eval is not None


@pytest.mark.unit
def test_save_is_a_stub_returning_none(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    assert opt.save() is None
    assert opt.save(checkpoint_dir="/nowhere") is None
