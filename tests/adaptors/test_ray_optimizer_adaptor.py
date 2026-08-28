"""Unit tests for ``core.adaptors.ray.optimizer.RayOptimizer``.

``RayOptimizer`` forwards ``run`` / ``evaluate`` / ``reset`` / ``stop`` to a
``PolicyActor`` and keeps light bookkeeping (inner iteration counter, per
iteration return and loss). The actor is replaced here by an in-memory stub
exposing the ``<method>.remote()`` call shape, ``ray.get`` is patched to the
identity, and the frozen ``RayOptimizerConfig`` is replaced by a namespace
carrying only the attributes the optimizer reads.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import ray
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

import core.adaptors.ray.policy_actor as policy_actor_module
from core.adaptors.ray.optimizer import RayOptimizer


class StubPolicyActor:
    """Stand-in for a ``PolicyActor`` handle.

    Every remote method returns ``train_result`` (for ``train``) or ``None``
    and appends its name to ``calls``.
    """

    def __init__(self, algo_config, train_result=None):
        self.algo_config = algo_config
        self.calls: list[str] = []
        self.train_result = train_result if train_result is not None else {}
        self.train = SimpleNamespace(remote=lambda: self._call("train"))
        self.evaluate = SimpleNamespace(remote=lambda: self._call("evaluate"))
        self.reset = SimpleNamespace(remote=lambda: self._call("reset"))
        self.stop = SimpleNamespace(remote=lambda: self._call("stop"))

    def _call(self, name):
        self.calls.append(name)
        return self.train_result if name == "train" else None


class StubPolicyActorClass:
    """Mimics the ``PolicyActor.remote(...)`` factory of a Ray actor class."""

    def __init__(self, train_result=None):
        self.train_result = train_result
        self.instances: list[StubPolicyActor] = []

    def remote(self, algo_config):
        actor = StubPolicyActor(algo_config, self.train_result)
        self.instances.append(actor)
        return actor


def make_config(
    *,
    evaluation_duration=10,
    rollout_fragment_length=5,
    num_envs_per_env_runner=6,
    seeds=(1, 2),
    env_reducers=None,
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
    cfg = SimpleNamespace(
        env=None,
        stats_cls_lookup=DEFAULT_STATS_CLS_LOOKUP,
        rllib_cfg=rllib_cfg,
        seeds=list(seeds),
    )
    if env_reducers is not None:
        cfg.env_reducers = env_reducers
    return cfg


@pytest.fixture
def stub_actor_class(monkeypatch):
    """Patch ``PolicyActor`` (imported lazily inside ``__init__``) and ``ray.get``."""
    stub = StubPolicyActorClass()
    monkeypatch.setattr(policy_actor_module, "PolicyActor", stub)
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    return stub


@pytest.mark.unit
def test_init_spawns_actor_and_derives_eval_episodes(stub_actor_class):
    cfg = make_config(evaluation_duration=10, rollout_fragment_length=5)

    opt = RayOptimizer(config=cfg)

    assert opt.eval_episodes == 2
    assert len(stub_actor_class.instances) == 1
    assert opt.policy_actor is stub_actor_class.instances[0]
    assert opt.policy_actor.algo_config is cfg.rllib_cfg
    assert opt._inner_iter == 0 and opt._es_round == 0
    assert opt._training_rewards == [] and opt._training_losses == []


@pytest.mark.unit
def test_init_uses_default_fishery_reducers_when_config_has_none(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    assert len(opt._env_reducers) > 0


@pytest.mark.unit
def test_init_keeps_explicit_env_reducers(stub_actor_class):
    reducers = ["custom-spec"]
    opt = RayOptimizer(config=make_config(env_reducers=reducers))
    assert opt._env_reducers == reducers


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


@pytest.mark.unit
def test_run_tracks_return_steps_and_loss(stub_actor_class, caplog):
    stub_actor_class.train_result = {
        "training_iteration": 7,
        "env_runners": {
            "episode_return_mean": 1.5,
            "num_env_steps_sampled": 100,
            "num_env_steps_sampled_lifetime": 700,
        },
        "info": {"learner": {"p0": {"learner_stats": {"policy_loss": 0.25}}}},
    }
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
def test_evaluate_forwards_to_actor(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    opt.evaluate()
    assert opt.policy_actor.calls == ["evaluate"]


@pytest.mark.unit
def test_reset_clears_tracking_and_bumps_es_round(stub_actor_class):
    stub_actor_class.train_result = {"env_runners": {"episode_return_mean": 2.0}}
    opt = RayOptimizer(config=make_config())
    opt.run()
    assert opt._training_rewards == [2.0]

    opt.reset()

    assert opt.policy_actor.calls == ["train", "reset"]
    assert opt._inner_iter == 0
    assert opt._es_round == 1
    assert opt._training_rewards == [] and opt._training_losses == []


@pytest.mark.unit
def test_stop_forwards_to_actor(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    opt.stop()
    assert opt.policy_actor.calls == ["stop"]


@pytest.mark.unit
def test_save_is_a_stub_returning_none(stub_actor_class):
    opt = RayOptimizer(config=make_config())
    assert opt.save() is None
    assert opt.save(checkpoint_dir="/nowhere") is None
