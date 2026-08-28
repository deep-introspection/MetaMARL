"""Unit tests for ``MultiAgentRegulatedEnv`` with the in-memory World.

The multi-agent env steps inertly (zero rewards, no dynamics, nothing
published) until the World hands it a published mechanism; once assigned it
runs the generic ``_step`` (intrinsic utility, shaped reward, transition,
observation with the mechanism vector appended) and publishes exactly one
``EnvStepContext`` per step. Reward logging has its own module
(``test_marl_reward_logging.py``); this one covers the remaining paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.envs.schema import EpisodeRolloutSchema
from core.world.context import EnvStepContext, MechanismContext, MechanismStatus

AGENTS = ["a", "b"]
BOX1 = spaces.Box(0.0, 1.0, (1,), np.float32)


def mechanism(theta):
    return SimpleNamespace(to_vector=lambda: list(theta), param_names=lambda: ["p"])


class DefaultSpace:
    dimension = 1

    @classmethod
    def default(cls):
        return mechanism([0.5])


class StockEnv(MultiAgentRegulatedEnv):
    """Stock depleted by the harvest of every agent; penalty on harvest."""

    def _reset(self):
        self.S_t = {"stock": 1.0}
        return {aid: self.observation(aid, self.S_t) for aid in self.agents}

    def transition_kernel(self, *, A_t, S_t, **kwargs):
        return {"stock": S_t["stock"] - sum(float(a[0]) for a in A_t.values())}

    def intrinsic_utility(self, *, A_t, **kwargs):
        return {aid: float(a[0]) for aid, a in A_t.items()}

    def violation_signal(self, u_i, aid=None, **kwargs):
        return u_i

    def penalty(self, u_i, **kwargs):
        return 0.5

    def _observation(self, agent_id, S_t):
        return np.array([S_t["stock"]], dtype=np.float32)

    def _is_truncated(self):
        return self.horizon is not None and self._t + 1 >= self.horizon


def publish(fake_world, theta):
    fake_world.get_mechanism_by_id = SimpleNamespace(
        remote=lambda **kw: MechanismContext(
            index=0,
            env_id=None,
            seed=kw.get("seed"),
            status=MechanismStatus.published,
            mechanism=mechanism(theta),
            metrics=None,
        )
    )


@pytest.fixture
def world(fake_world):
    fake_world.get_mechanism_by_id = SimpleNamespace(remote=lambda **kw: None)
    return fake_world


def make_env(world, **kwargs):
    kwargs.setdefault("mechanism_space", DefaultSpace)
    return StockEnv(world=world, mechanism_id=0, agents=AGENTS, horizon=3, **kwargs)


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
    assert make_env(world).action_space == spaces.Dict({})


@pytest.mark.unit
def test_inert_step_before_publication(world):
    env = make_env(world)
    obs, infos = env.reset()
    assert not env.published_mechanism_assigned
    assert infos == {"a": {}, "b": {}}
    # default mechanism vector appended to the base observation
    np.testing.assert_allclose(obs["a"], [1.0, 0.5])

    obs2, rewards, term, trunc, infos = env.step({aid: np.ones(1) for aid in AGENTS})
    assert rewards == {"a": 0.0, "b": 0.0}
    assert term == {"a": False, "b": False, "__all__": False}
    assert trunc == {"a": False, "b": False, "__all__": False}
    assert env.S_t == {"stock": 1.0}  # no dynamics
    np.testing.assert_allclose(obs2["a"], obs["a"])
    assert world.contexts == []  # nothing published on inert steps
    assert env._t == 1


@pytest.mark.unit
def test_full_step_pipeline_with_published_mechanism(world):
    publish(world, [0.25])
    env = make_env(world, seed=7, policy_seed=11)
    obs, _ = env.reset()
    assert env.published_mechanism_assigned
    np.testing.assert_allclose(obs["b"], [1.0, 0.25])

    raw = {"a": np.array([0.4]), "b": np.array([0.2])}
    obs, rewards, term, trunc, infos = env.step(raw)

    # u_i - penalty * violation = u_i - 0.5 * u_i, then shared as the mean
    shaped = {"a": 0.4 - 0.5 * 0.4, "b": 0.2 - 0.5 * 0.2}
    expected = float(np.mean(list(shaped.values())))
    assert rewards == {"a": pytest.approx(expected), "b": pytest.approx(expected)}
    assert env.S_t["stock"] == pytest.approx(0.4)
    np.testing.assert_allclose(obs["a"], [0.4, 0.25], rtol=1e-6)
    assert infos["a"]["intrinsic_utility"] == pytest.approx(0.4)
    assert infos["b"]["intrinsic_utility"] == pytest.approx(0.2)
    assert term["__all__"] is False and trunc["__all__"] is False

    assert len(world.contexts) == 1
    payload = world.contexts[0].payload
    assert isinstance(payload, EnvStepContext)
    assert (payload.mechanism, payload.seed, payload.policy_seed) == (0, 7, 11)
    assert payload.status == MechanismStatus.train
    assert payload.reward == rewards and payload.action is raw
    assert payload.info == infos  # pydantic copies the dict


@pytest.mark.unit
def test_horizon_truncates_and_reset_rewinds(world):
    publish(world, [0.0])
    env = make_env(world)
    env.reset()
    zeros = {aid: np.zeros(1) for aid in AGENTS}
    for expected in (False, False, True):
        *_, trunc, _ = env.step(zeros)
        assert trunc["__all__"] is expected
    assert env._t == 3
    obs, infos = env.reset()
    assert env._t == 0 and infos == {"a": {}, "b": {}}


@pytest.mark.unit
def test_reset_keeps_construction_seed_and_logs_identity(world):
    env = make_env(world, seed=3, policy_seed=4, schema=EpisodeRolloutSchema)
    env.reset(seed=99)
    assert env.seed == 3
    peeked = env.logger.peek()
    assert peeked.seed == [3] and peeked.policy_seed == [4]
    assert peeked.mechanism_id == [0]
    assert peeked.iter == []  # reset flushes iter; nothing logged yet
    # without a schema the reset path skips the logger entirely
    make_env(world).reset(seed=99)


@pytest.mark.unit
def test_update_infos_broadcasts_scalars_and_aggregate_shares_mean(world):
    env = make_env(world)
    env._update_infos("flag", 1.0)
    env._update_infos("u", {"a": 2.0, "b": 4.0})
    assert env._infos == {"a": {"flag": 1.0, "u": 2.0}, "b": {"flag": 1.0, "u": 4.0}}
    assert env._aggregate_rewards({"a": 1.0, "b": 3.0}) == {"a": 2.0, "b": 2.0}
