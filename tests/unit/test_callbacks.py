"""Unit tests for the RLlib hooks in ``core.callbacks``.

Two hooks are covered with duck-typed fakes and no Ray runtime:

- ``tag_episode_with_env_idx`` rewrites the episode ID from the identity of the
  sub-environment (``mechanism_id``, ``seed``, ``policy_seed``) and guards the
  immutability of ``env_id``.
- ``_evaluate_with_fixed_duration_once`` is the strict single-round evaluation
  loop. A fake ``EnvRunnerGroup`` runs the remote function synchronously on
  fake workers so the exact-once guards (missing results, incomplete units,
  stale iterations) can be triggered deterministically.

The old API stack branch of the evaluation function is not exercised by this
project and is left uncovered on purpose.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS, NUM_EPISODES

from core.callbacks import _evaluate_with_fixed_duration_once, tag_episode_with_env_idx

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
):
    config = SimpleNamespace(
        evaluation_duration_unit=unit,
        evaluation_duration=duration,
        evaluation_num_env_runners=len(group.workers),
        evaluation_force_reset_envs_before_iteration=force_reset,
        evaluation_sample_timeout_s=timeout,
        enable_env_runner_and_connector_v2=True,
        count_steps_by=count_steps_by,
    )
    return SimpleNamespace(
        config=config,
        evaluation_config=SimpleNamespace(),
        eval_env_runner_group=group,
        iteration=iteration,
        metrics=metrics if metrics is not None else FakeMetrics(),
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
