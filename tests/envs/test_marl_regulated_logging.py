"""Logging and bookkeeping paths of ``MultiAgentRegulatedEnv`` (in-memory World).

The step pipeline itself is covered by ``test_marl_regulated.py``; this module
checks what the logging feature adds on top of it: the metric logger wired
from ``schema``, the env-level reporter built from ``reporter_cfg``, the
episode identity logged at ``reset``, the ``iter`` counter on inert steps,
the World fetch discipline and the ``infos`` helper. Reward logging has its
own module (``test_marl_reward_logging.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.envs import hooks
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.world.context import MechanismContext, MechanismStatus

AGENTS = ["a", "b"]
BOX1 = spaces.Box(0.0, 1.0, (1,), np.float32)


class StockEnv(MultiAgentRegulatedEnv):
    """Stock depleted by the first action component of every agent."""

    @hooks.reset
    def init_state(self):
        return {"stock": 1.0}

    @hooks.reward
    def base_reward(self, A_t):
        return {aid: float(a[0]) for aid, a in A_t.items()}

    @hooks.transition
    def dynamics(self, *, A_t, S_t):
        return {"stock": S_t["stock"] - sum(float(a[0]) for a in A_t.values())}

    @hooks.observation
    def obs(self, observation_dict):
        return {
            aid: np.array([self.S_t["stock"]], dtype=np.float32) for aid in self.agents
        }


class RecordingReporter(Reporter):
    def __init__(self, label):
        self.label = label
        self.reports = []

    def _report(self, query, series):
        self.reports.append((query.title, series))

    def close(self):
        pass


class RecordingConfig(ReporterConfig):
    def build(self, *, label=None):
        return RecordingReporter(label)


def inert_mechanism():
    return SubsidyMechanism(subsidy=0.0, cost=0.0)


def publish(fake_world, mechanism, calls=None):
    def remote(**kw):
        if calls is not None:
            calls.append(kw)
        return MechanismContext(
            index=0,
            env_id=None,
            seed=kw.get("seed"),
            status=MechanismStatus.published,
            mechanism=mechanism,
            metrics=None,
        )

    fake_world.get_mechanism_by_id = SimpleNamespace(remote=remote)


@pytest.fixture
def world(fake_world):
    fake_world.get_mechanism_by_id = SimpleNamespace(remote=lambda **kw: None)
    return fake_world


def make_env(world, **kwargs):
    kwargs.setdefault("mechanism", inert_mechanism())
    kwargs.setdefault("action_spaces", {aid: BOX1 for aid in AGENTS})
    kwargs.setdefault("mechanism_id", 0)
    return StockEnv(world=world, agents=AGENTS, horizon=3, **kwargs)


@pytest.mark.unit
def test_spaces_and_agents_are_wired_from_kwargs(world):
    env = make_env(
        world,
        action_spaces={aid: BOX1 for aid in AGENTS},
        observation_spaces={aid: BOX1 for aid in AGENTS},
    )
    assert env.possible_agents == AGENTS and env.possible_agents is not env.agents
    assert isinstance(env.action_space, spaces.Dict)
    assert set(env.action_space.spaces) == set(AGENTS)
    assert set(env.observation_space.spaces) == set(AGENTS)
    assert make_env(world, action_spaces=None).action_space == spaces.Dict({})


@pytest.mark.unit
def test_env_without_schema_or_reporter_is_silent(world):
    env = make_env(world)
    assert env.logger is None and env.reporter is None
    env.reset(seed=99)  # the reset path skips the logger entirely
    env.step({aid: np.ones(1) for aid in AGENTS})


@pytest.mark.unit
def test_reporter_is_built_with_env_identity_and_queries(world):
    cfg = RecordingConfig(project="p")
    query = Query(title="r", x=("iter",), y=("reward_mean",))
    env = make_env(
        world,
        env_name="stock",
        mode="train",
        seed=5,
        policy_seed=3,
        reporter_cfg=cfg,
        schema=EpisodeRolloutSchema,
        queries=(query,),
    )
    assert env.reporter.label == "stock|mode=train|m=0|ps=3|ss=5"
    assert env.reporter.schema is EpisodeRolloutSchema
    assert env.reporter.queries == (query,)


@pytest.mark.unit
def test_reset_keeps_construction_seed_and_logs_identity(world):
    env = make_env(world, seed=3, policy_seed=4, schema=EpisodeRolloutSchema)
    env.reset(seed=99)
    assert env.seed == 3
    peeked = env.logger.peek()
    assert peeked.seed == [3] and peeked.policy_seed == [4]
    assert peeked.mechanism_id == [0]
    assert peeked.iter == []  # reset flushes iter; nothing logged yet


@pytest.mark.unit
def test_inert_step_logs_iter_only(world):
    env = make_env(world, schema=EpisodeRolloutSchema)
    env.reset()
    assert not env.published_mechanism_assigned
    env.step({aid: np.ones(1) for aid in AGENTS})
    env.step({aid: np.ones(1) for aid in AGENTS})
    peeked = env.logger.peek()
    assert peeked.iter == [1, 2]
    assert peeked.reward_mean == []  # inert steps log no rewards
    assert world.contexts == []


@pytest.mark.unit
def test_reset_fetches_once_then_keeps_its_mechanism(world):
    calls = []
    mechanism = inert_mechanism()
    publish(world, mechanism, calls)
    env = make_env(world, mechanism_id=2, policy_seed=5, mode="eval", seed=1)
    env.reset()
    assert env.published_mechanism_assigned
    assert env.mechanism is mechanism and env.m_ctx.mechanism is mechanism
    assert calls == [{"mechanism_id": 2, "seed": 5, "mode": MechanismStatus.eval}]
    env.reset()
    assert len(calls) == 1  # no second fetch once a candidate is assigned


@pytest.mark.unit
def test_reset_requires_a_mechanism_id(world):
    env = make_env(world, mechanism_id=None)
    with pytest.raises(RuntimeError, match="no mechanism_id"):
        env.reset()


@pytest.mark.unit
def test_update_infos_broadcasts_scalars(world):
    env = make_env(world)
    env._update_infos("flag", 1.0)
    env._update_infos("u", {"a": 2.0, "b": 4.0})
    assert env._infos == {"a": {"flag": 1.0, "u": 2.0}, "b": {"flag": 1.0, "u": 4.0}}
