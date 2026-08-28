"""Unit tests for the RLlib hooks in ``core.callbacks``.

Two hooks are covered with duck-typed fakes and no Ray runtime:

- ``tag_episode_with_env_idx`` rewrites the episode ID from the identity of the
  sub-environment (``mechanism_id``, ``seed``, ``policy_seed``) and guards the
  immutability of ``env_id``.
- ``_evaluate_with_fixed_duration_once`` is the strict single-round evaluation
  loop. A fake ``EnvRunnerGroup`` runs the remote function synchronously on
  fake workers so the exact-once guards (missing results, incomplete units,
  stale iterations) can be triggered deterministically.

The old API stack branch (``enable_env_runner_and_connector_v2=False``),
kept verbatim from RLlib on this branch, is driven the same way: fake
``RolloutWorker`` batches are returned by a fake
``foreach_env_runner_async_fetch_ready`` and ``summarize_episodes`` is
replaced by a recorder, since building real ``RolloutMetrics`` is beside the
point. ``log_and_report_episode_metrics`` (the episode-end hook) is checked
with a real ``MetricLogger`` and the recording reporter of the env logging
tests: peek, report, reduce, then hand the reduced schema to RLlib's logger.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS, NUM_EPISODES

import core.callbacks as callbacks_module
from core.callbacks import (
    _evaluate_with_fixed_duration_once,
    log_and_report_episode_metrics,
    tag_episode_with_env_idx,
)
from core.metrics.logger import MetricLogger
from core.reporting.query import Query
from tests.envs.test_env_logging import RecordingReporter, StepSchema

# --------------------------------------------------------------------------- #
# tag_episode_with_env_idx
# --------------------------------------------------------------------------- #


class FakeSubEnv:
    """Minimal stand-in for a ``BaseEnv`` created by the env creator."""

    def __init__(self, mechanism_id=0, seed=101, policy_seed=101, env_id=None):
        self.mechanism_id = mechanism_id
        self.seed = seed
        self.policy_seed = policy_seed
        self.env_id = env_id


def make_runner(*sub_envs):
    """Build ``env_runner.env.envs[i].unwrapped`` around the given sub-envs."""
    wrapped = [SimpleNamespace(unwrapped=sub_env) for sub_env in sub_envs]
    return SimpleNamespace(env=SimpleNamespace(envs=wrapped))


def tag(episode, runner, env_index):
    tag_episode_with_env_idx(
        episode=episode, env_runner=runner, env=None, env_index=env_index
    )


@pytest.mark.unit
def test_tag_rewrites_episode_id_and_sets_env_id():
    sub_env = FakeSubEnv(mechanism_id=2, seed=7, policy_seed=11)
    episode = SimpleNamespace(id_="abc123")

    tag(episode, make_runner(FakeSubEnv(), sub_env), env_index=1)

    assert episode.id_ == "env=1|m=2|ps=11|ss=7|raw=abc123"
    assert sub_env.env_id == 1


@pytest.mark.unit
def test_tag_leaves_already_tagged_ids_untouched():
    sub_env = FakeSubEnv(env_id=0)
    episode = SimpleNamespace(id_="env=0|m=0|ps=1|ss=1|raw=x")

    tag(episode, make_runner(sub_env), env_index=0)

    assert episode.id_ == "env=0|m=0|ps=1|ss=1|raw=x"


@pytest.mark.unit
def test_tag_accepts_same_env_id_on_later_episodes():
    sub_env = FakeSubEnv(env_id=3)
    runner = make_runner(FakeSubEnv(), FakeSubEnv(), FakeSubEnv(), sub_env)
    episode = SimpleNamespace(id_="raw")

    tag(episode, runner, env_index=3)

    assert episode.id_.startswith("env=3|")


@pytest.mark.unit
def test_tag_raises_when_env_id_changes():
    sub_env = FakeSubEnv(env_id=0)
    runner = make_runner(FakeSubEnv(), sub_env)
    with pytest.raises(RuntimeError, match="Immutable env_id changed"):
        tag(SimpleNamespace(id_="raw"), runner, env_index=1)


@pytest.mark.unit
def test_tag_raises_without_mechanism_id():
    sub_env = FakeSubEnv(mechanism_id=None)
    with pytest.raises(RuntimeError, match="no mechanism_id"):
        tag(SimpleNamespace(id_="raw"), make_runner(sub_env), env_index=0)


@pytest.mark.unit
def test_tag_raises_without_seed():
    sub_env = FakeSubEnv(seed=None)
    with pytest.raises(RuntimeError, match="no seed"):
        tag(SimpleNamespace(id_="raw"), make_runner(sub_env), env_index=0)


# --------------------------------------------------------------------------- #
# _evaluate_with_fixed_duration_once
# --------------------------------------------------------------------------- #


class FakeStat:
    """A metrics ``Stats`` leaf exposing ``peek()``."""

    def __init__(self, value):
        self.value = value

    def peek(self):
        return self.value


class FakeEpisode:
    def __init__(self, env_steps, agent_steps):
        self._env_steps = env_steps
        self._agent_steps = agent_steps

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class FakeWorker:
    """Evaluation env runner producing ``num_episodes`` episodes on ``sample``.

    ``episodes_returned`` overrides the count actually delivered (to simulate
    an incomplete round) and ``report_num_episodes=False`` omits the
    ``NUM_EPISODES`` key from the metrics.
    """

    def __init__(
        self,
        worker_index,
        *,
        env_steps_per_episode=10,
        agent_steps_per_episode=20,
        episodes_returned=None,
        report_num_episodes=True,
    ):
        self.worker_index = worker_index
        self.env_steps_per_episode = env_steps_per_episode
        self.agent_steps_per_episode = agent_steps_per_episode
        self.episodes_returned = episodes_returned
        self.report_num_episodes = report_num_episodes
        self.sample_calls: list[dict] = []
        self._last_count = 0

    def sample(self, *, num_timesteps, num_episodes, force_reset):
        self.sample_calls.append(
            {
                "num_timesteps": num_timesteps,
                "num_episodes": num_episodes,
                "force_reset": force_reset,
            }
        )
        if self.episodes_returned is not None:
            count = self.episodes_returned
        elif num_episodes is not None:
            count = num_episodes
        else:
            count = max(1, num_timesteps // self.env_steps_per_episode)
        self._last_count = count
        return [
            FakeEpisode(self.env_steps_per_episode, self.agent_steps_per_episode)
            for _ in range(count)
        ]

    def get_metrics(self):
        if not self.report_num_episodes:
            return {}
        return {NUM_EPISODES: FakeStat(self._last_count)}


class FakeGroup:
    """``EnvRunnerGroup`` stand-in running the remote function in-process."""

    def __init__(self, workers, *, drop_results=0):
        self.workers = list(workers)
        self.drop_results = drop_results
        self.foreach_calls: list[dict] = []

    def num_healthy_remote_workers(self):
        return len(self.workers)

    def healthy_worker_ids(self):
        return [w.worker_index for w in self.workers]

    def foreach_env_runner(self, *, func, kwargs, local_env_runner, timeout_seconds):
        self.foreach_calls.append(
            {
                "kwargs": kwargs,
                "local_env_runner": local_env_runner,
                "timeout_seconds": timeout_seconds,
            }
        )
        results = [func(w, **kwargs) for w in self.workers]
        if self.drop_results:
            results = results[: len(results) - self.drop_results]
        return results


class FakeMetrics:
    """``MetricsLogger`` stand-in recording ``aggregate`` and answering ``peek``."""

    def __init__(self, num_episodes=None, eval_results=None):
        self.aggregate_calls: list[dict] = []
        self.num_episodes = num_episodes
        self.eval_results = eval_results if eval_results is not None else {"ok": True}

    def aggregate(self, stats, *, key):
        self.aggregate_calls.append({"stats": stats, "key": key})

    def peek(self, key, *, default=None, latest_merged_only=False):
        assert latest_merged_only is True
        if key == (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES):
            if self.num_episodes is None:
                # Derive from what was aggregated, like a real logger would.
                total = 0
                for call in self.aggregate_calls:
                    for met in call["stats"]:
                        if NUM_EPISODES in met:
                            total += met[NUM_EPISODES].peek()
                return total if self.aggregate_calls else default
            return self.num_episodes
        if key == EVALUATION_RESULTS:
            return self.eval_results
        return default


def make_algo(
    group,
    *,
    unit="episodes",
    duration=4,
    count_steps_by="env_steps",
    iteration=3,
    force_reset=True,
    timeout=12.5,
    metrics=None,
    new_stack=True,
    evaluation_config=None,
    reward_estimators=None,
):
    config = SimpleNamespace(
        evaluation_duration_unit=unit,
        evaluation_duration=duration,
        evaluation_num_env_runners=len(group.workers),
        evaluation_force_reset_envs_before_iteration=force_reset,
        evaluation_sample_timeout_s=timeout,
        enable_env_runner_and_connector_v2=new_stack,
        count_steps_by=count_steps_by,
    )
    return SimpleNamespace(
        config=config,
        evaluation_config=(
            evaluation_config if evaluation_config is not None else SimpleNamespace()
        ),
        eval_env_runner_group=group,
        iteration=iteration,
        metrics=metrics if metrics is not None else FakeMetrics(),
        reward_estimators=reward_estimators,
    )


@pytest.mark.unit
def test_episodes_are_split_evenly_across_runners_in_one_round():
    workers = [FakeWorker(1), FakeWorker(2)]
    group = FakeGroup(workers)
    algo = make_algo(group, unit="episodes", duration=4)

    eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
        algo, group
    )

    # Exactly one round, two episodes per runner, force reset on round 0.
    assert len(group.foreach_calls) == 1
    call = group.foreach_calls[0]
    assert call["kwargs"]["num"] == [None, 2, 2]
    assert call["kwargs"]["round"] == 0
    assert call["kwargs"]["iter"] == 3
    assert call["local_env_runner"] is False
    assert call["timeout_seconds"] == 12.5
    for worker in workers:
        assert worker.sample_calls == [
            {"num_timesteps": None, "num_episodes": 2, "force_reset": True}
        ]

    assert env_steps == 4 * 10
    assert agent_steps == 4 * 20
    assert eval_results == {"ok": True}
    assert len(algo.metrics.aggregate_calls) == 1
    assert algo.metrics.aggregate_calls[0]["key"] == (
        EVALUATION_RESULTS,
        ENV_RUNNER_RESULTS,
    )
    assert len(algo.metrics.aggregate_calls[0]["stats"]) == 2


@pytest.mark.unit
def test_uneven_episode_split_gives_remainder_to_first_runners():
    group = FakeGroup([FakeWorker(1), FakeWorker(2), FakeWorker(3)])
    algo = make_algo(group, unit="episodes", duration=5)

    _evaluate_with_fixed_duration_once(algo, group)

    assert group.foreach_calls[0]["kwargs"]["num"] == [None, 2, 2, 1]


@pytest.mark.unit
def test_timesteps_unit_counts_env_steps():
    workers = [
        FakeWorker(1, env_steps_per_episode=10),
        FakeWorker(2, env_steps_per_episode=10),
    ]
    group = FakeGroup(workers)
    algo = make_algo(group, unit="timesteps", duration=40, count_steps_by="env_steps")

    _, env_steps, agent_steps = _evaluate_with_fixed_duration_once(algo, group)

    assert env_steps == 40
    assert agent_steps == 80
    for worker in workers:
        assert worker.sample_calls[0]["num_timesteps"] == 20
        assert worker.sample_calls[0]["num_episodes"] is None


@pytest.mark.unit
def test_timesteps_unit_counts_agent_steps_when_configured():
    workers = [
        FakeWorker(1, env_steps_per_episode=10, agent_steps_per_episode=20),
        FakeWorker(2, env_steps_per_episode=10, agent_steps_per_episode=20),
    ]
    group = FakeGroup(workers)
    # 40 requested timesteps -> 20 per worker -> 2 episodes -> 40 agent steps
    # each -> 80 agent steps in total, which differs from the request.
    algo = make_algo(group, unit="timesteps", duration=40, count_steps_by="agent_steps")

    with pytest.raises(RuntimeError, match="completed=80"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_force_reset_flag_is_forwarded():
    worker = FakeWorker(1)
    group = FakeGroup([worker])
    algo = make_algo(group, duration=1, force_reset=False)

    _evaluate_with_fixed_duration_once(algo, group)

    assert worker.sample_calls[0]["force_reset"] is False


@pytest.mark.unit
def test_missing_runner_result_raises_instead_of_retrying():
    group = FakeGroup([FakeWorker(1), FakeWorker(2)], drop_results=1)
    algo = make_algo(group, duration=4)

    with pytest.raises(RuntimeError, match="expected=2, received=1"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_incomplete_round_raises():
    group = FakeGroup([FakeWorker(1), FakeWorker(2, episodes_returned=1)])
    algo = make_algo(group, duration=4)

    with pytest.raises(RuntimeError, match="requested=4, completed=3"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_metrics_without_num_episodes_count_as_zero_units():
    group = FakeGroup([FakeWorker(1, report_num_episodes=False)])
    algo = make_algo(group, duration=2)

    with pytest.raises(RuntimeError, match="completed=0"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_stale_iteration_results_are_discarded():
    group = FakeGroup([FakeWorker(1)])
    algo = make_algo(group, duration=2, iteration=5)

    # The remote function echoes ``algo.iteration`` read before sampling; a
    # worker that answers for an older iteration is ignored, so the round is
    # incomplete.
    original = group.foreach_env_runner

    def foreach_with_stale_iter(**kwargs):
        results = original(**kwargs)
        return [(e, a, m, it - 1) for e, a, m, it in results]

    group.foreach_env_runner = foreach_with_stale_iter

    with pytest.raises(RuntimeError, match="completed=0"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_no_healthy_workers_warns_and_returns_empty(caplog):
    group = FakeGroup([])
    metrics = FakeMetrics(num_episodes=0, eval_results={})
    algo = make_algo(group, duration=4, metrics=metrics)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
            algo, group
        )

    assert (eval_results, env_steps, agent_steps) == ({}, 0, 0)
    assert group.foreach_calls == []
    assert "all workers crashing" in caplog.text
    assert "empty set of episode summary" in caplog.text


@pytest.mark.unit
def test_zero_duration_skips_sampling(caplog):
    group = FakeGroup([FakeWorker(1)])
    metrics = FakeMetrics(num_episodes=0)
    algo = make_algo(group, duration=0, metrics=metrics)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    assert env_steps == 0
    assert group.foreach_calls == []
    assert "all workers crashing" not in caplog.text
    assert "empty set of episode summary" in caplog.text


# --------------------------------------------------------------------------- #
# _evaluate_with_fixed_duration_once: old API stack branch
# --------------------------------------------------------------------------- #


class FakeBatch:
    """``SampleBatch`` stand-in exposing the two step counters."""

    def __init__(self, env_steps, agent_steps):
        self._env_steps = env_steps
        self._agent_steps = agent_steps

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class FakeOldWorker:
    """``RolloutWorker`` stand-in: one batch and one metrics entry per call."""

    def __init__(self, worker_index, *, env_steps=10, agent_steps=20):
        self.worker_index = worker_index
        self.env_steps = env_steps
        self.agent_steps = agent_steps
        self.sample_calls = 0

    def sample(self):
        self.sample_calls += 1
        return FakeBatch(self.env_steps, self.agent_steps)

    def get_metrics(self):
        return [f"metrics-{self.worker_index}-{self.sample_calls}"]


class FakeOldGroup:
    """``EnvRunnerGroup`` stand-in for the old stack.

    ``foreach_env_runner_async_fetch_ready`` runs ``func`` on the selected
    workers in-process; ``results_per_round`` can override the returned list
    per round (``None`` entries mean "return nothing this round").
    """

    def __init__(self, workers, *, results_per_round=None):
        self.workers = list(workers)
        self.results_per_round = results_per_round
        self.calls: list[dict] = []

    def num_healthy_remote_workers(self):
        return len(self.workers)

    def healthy_worker_ids(self):
        return [w.worker_index for w in self.workers]

    def foreach_env_runner(self, **kwargs):  # pragma: no cover - new stack only
        raise AssertionError("old stack must not use foreach_env_runner")

    def foreach_env_runner_async_fetch_ready(self, *, func, remote_worker_ids, tag):
        round_index = len(self.calls)
        self.calls.append({"remote_worker_ids": list(remote_worker_ids), "tag": tag})
        if self.results_per_round is not None:
            spec = self.results_per_round[round_index]
            if spec is None:
                return []
        selected = [w for w in self.workers if w.worker_index in remote_worker_ids]
        return [func(w) for w in selected]


@pytest.fixture
def fake_summarize(monkeypatch):
    """Replace ``summarize_episodes`` by a recorder counting the metrics."""
    calls = []

    def summarize(episodes, new_episodes, keep_custom_metrics):
        calls.append(
            {
                "episodes": list(episodes),
                "new_episodes": list(new_episodes),
                "keep_custom_metrics": keep_custom_metrics,
            }
        )
        return {NUM_EPISODES: len(episodes), "episode_reward_mean": 1.0}

    monkeypatch.setattr(callbacks_module, "summarize_episodes", summarize)
    return calls


def make_old_algo(group, **kwargs):
    evaluation_config = SimpleNamespace(
        rollout_fragment_length=5,
        num_envs_per_env_runner=2,
        keep_per_episode_custom_metrics=True,
    )
    kwargs.setdefault("evaluation_config", evaluation_config)
    return make_algo(group, new_stack=False, **kwargs)


@pytest.mark.unit
def test_old_stack_episodes_loop_until_duration_is_reached(fake_summarize):
    workers = [FakeOldWorker(1), FakeOldWorker(2)]
    group = FakeOldGroup(workers)
    algo = make_old_algo(group, unit="episodes", duration=3, iteration=4)

    eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
        algo, group
    )

    # Round 0: both workers (2 episodes left -> ids 1 and 2); round 1: one
    # episode left, only the first worker is selected.
    assert [c["remote_worker_ids"] for c in group.calls] == [[1, 2], [1]]
    assert all(c["tag"] == "env_runner_sample_and_get_metrics" for c in group.calls)
    assert (workers[0].sample_calls, workers[1].sample_calls) == (2, 1)
    assert env_steps == 30 and agent_steps == 60

    (call,) = fake_summarize
    assert len(call["episodes"]) == 3
    assert call["episodes"] == call["new_episodes"]
    assert call["keep_custom_metrics"] is True
    assert eval_results == {
        ENV_RUNNER_RESULTS: {NUM_EPISODES: 3, "episode_reward_mean": 1.0}
    }
    # The new-stack metrics logger is untouched on this branch.
    assert algo.metrics.aggregate_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "count_steps_by, expected_rounds",
    [("env_steps", 2), ("agent_steps", 1)],
)
def test_old_stack_timesteps_unit_counts_configured_steps(
    fake_summarize, count_steps_by, expected_rounds
):
    # Each worker delivers 10 env steps / 20 agent steps per batch; one
    # worker per 10 units (rollout_fragment_length * num_envs) is selected.
    workers = [FakeOldWorker(1), FakeOldWorker(2)]
    group = FakeOldGroup(workers)
    algo = make_old_algo(
        group, unit="timesteps", duration=40, count_steps_by=count_steps_by
    )

    _, env_steps, agent_steps = _evaluate_with_fixed_duration_once(algo, group)

    assert len(group.calls) == expected_rounds
    assert group.calls[0]["remote_worker_ids"] == [1, 2]
    if count_steps_by == "env_steps":
        assert env_steps == 40 and agent_steps == 80
    else:
        assert env_steps == 20 and agent_steps == 40


@pytest.mark.unit
def test_old_stack_keeps_batches_for_reward_estimators(fake_summarize):
    group = FakeOldGroup([FakeOldWorker(1)])
    algo = make_old_algo(group, duration=1, reward_estimators={"ope": object()})

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)
    assert env_steps == 10


@pytest.mark.unit
def test_old_stack_discards_stale_iteration_then_completes(fake_summarize):
    group = FakeOldGroup([FakeOldWorker(1)])
    algo = make_old_algo(group, duration=1, iteration=6)
    original = group.foreach_env_runner_async_fetch_ready
    stale_once = {"done": False}

    def fetch_with_stale_first(**kwargs):
        results = original(**kwargs)
        if not stale_once["done"]:
            stale_once["done"] = True
            return [(b, m, it - 1) for b, m, it in results]
        return results

    group.foreach_env_runner_async_fetch_ready = fetch_with_stale_first

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    # Two rounds: the stale batch is skipped but still counted as a unit
    # ("1 episode per returned batch"), so the loop ends after round 0 with
    # no steps recorded. Documented RLlib behaviour.
    assert len(group.calls) == 1
    assert env_steps == 0
    assert fake_summarize[0]["episodes"] == []


@pytest.mark.unit
def test_old_stack_times_out_without_results(fake_summarize, caplog):
    group = FakeOldGroup([FakeOldWorker(1)], results_per_round=[None])
    # A negative timeout makes the first empty round exceed it immediately.
    algo = make_old_algo(group, duration=2, timeout=-1.0)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
            algo, group
        )

    assert len(group.calls) == 1
    assert (env_steps, agent_steps) == (0, 0)
    assert eval_results[ENV_RUNNER_RESULTS][NUM_EPISODES] == 0
    assert "empty set of episode summary" in caplog.text


@pytest.mark.unit
def test_old_stack_empty_round_within_timeout_retries(fake_summarize):
    group = FakeOldGroup([FakeOldWorker(1)], results_per_round=[None, "go", "go", "go"])
    algo = make_old_algo(group, duration=2, timeout=60.0)

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    # Round 0 returned nothing (still within the timeout), rounds 1-2 delivered
    # one episode each.
    assert len(group.calls) == 3
    assert env_steps == 20


# --------------------------------------------------------------------------- #
# log_and_report_episode_metrics
# --------------------------------------------------------------------------- #


def make_logging_env(values=(2.0,), iteration=1):
    """Return ``(env, env_runner)`` with a populated ``MetricLogger``."""
    logger = MetricLogger.from_schema(StepSchema)
    logger.push(("iter",), iteration)
    for value in values:
        logger.push(("value",), value)
    reporter = RecordingReporter("env")
    reporter.add_query(Query(title="v", x=("iter",), y=("value",)))
    env = SimpleNamespace(logger=logger, reporter=reporter)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)])
    )
    return env, env_runner


@pytest.mark.unit
def test_episode_end_callback_reports_then_reduces():
    env, env_runner = make_logging_env(values=(2.0,))
    logged = []
    metrics_logger = SimpleNamespace(log_value=lambda **kw: logged.append(kw))

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="env=0|m=1|ps=2|ss=3|raw=abc"),
        env_runner=env_runner,
        env=None,
        env_index=0,
        metrics_logger=metrics_logger,
    )

    assert env.reporter.reports == [("v", [1], [[2.0]])]
    assert len(env.logger._refs[("value",)]) == 0  # reduced (destructive)
    (call,) = logged
    assert call["key"] == ("by_episode", "env=0|m=1|ps=2|ss=3")
    assert call["reduce"] == "item"
    assert call["value"].value == [2.0] and call["value"].iter == 1


@pytest.mark.unit
def test_episode_end_callback_picks_sub_env_by_index_and_keeps_untagged_id():
    env, _ = make_logging_env(values=(3.0,), iteration=4)
    other = SimpleNamespace(logger=None, reporter=None)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(
            envs=[SimpleNamespace(unwrapped=other), SimpleNamespace(unwrapped=env)]
        )
    )
    logged = []
    metrics_logger = SimpleNamespace(log_value=lambda **kw: logged.append(kw))

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="plain-id"),
        env_runner=env_runner,
        env=None,
        env_index=1,
        metrics_logger=metrics_logger,
        extra="ignored",
    )

    # An ID without the ``|raw=`` marker is used unchanged.
    assert logged[0]["key"] == ("by_episode", "plain-id")
    assert logged[0]["value"].value == [3.0]
    assert env.reporter.reports == [("v", [4], [[3.0]])]
