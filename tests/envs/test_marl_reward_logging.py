"""``MultiAgentRegulatedEnv`` logs rewards exactly once per step (logging branch)."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from pydantic import Field

from core.envs import hooks
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.metrics.enums import ReduceProtocol
from core.world.context import MechanismContext, MechanismStatus


class AgentSchema(AgentEnvStepSchema):
    pass


class EnvSchema(EpisodeRolloutSchema):
    stock: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    by_agent: dict[str, AgentSchema] = Field(default_factory=dict)


class StockEnv(MultiAgentRegulatedEnv):
    """Benchmark whose transition hook logs its own ``stock`` series."""

    @hooks.reset
    def init_state(self):
        return {"stock": 1.0}

    @hooks.reward
    def base_reward(self, A_t):
        return {aid: float(a[0]) for aid, a in A_t.items()}

    @hooks.transition
    def dynamics(self, *, A_t, S_t):
        stock = S_t["stock"] - 0.1 * len(A_t)
        self._log(("stock",), stock)
        return {"stock": stock}

    @hooks.observation
    def obs(self, observation_dict):
        return {
            aid: np.array([self.S_t["stock"]], dtype=np.float32) for aid in self.agents
        }


@pytest.mark.unit
def test_rewards_logged_once_per_step_after_the_mechanism(fake_world):
    # subsidy 0.5 on component 1 (raw 40 -> effort ~1): reward = u_i + 0.5 * effort
    mechanism = SubsidyMechanism(subsidy=0.5, cost=0.0, action_component=1)
    fake_world.get_mechanism_by_id = SimpleNamespace(
        remote=lambda **kw: MechanismContext(
            index=0,
            env_id=None,
            seed=None,
            status=MechanismStatus.published,
            mechanism=mechanism,
            metrics=None,
        )
    )
    env = StockEnv(
        world=fake_world,
        mechanism_id=0,
        agents=["a", "b"],
        mechanism=mechanism,
        schema=EnvSchema,
        horizon=5,
    )
    env.reset()
    assert env.published_mechanism_assigned

    raw = {"a": np.array([0.0, 40.0]), "b": np.array([0.0, -40.0])}
    _, rewards, *_ = env.step(raw)
    # normalized harvest 0.5 each; effort ~1 for a, ~0 for b
    assert rewards["a"] == pytest.approx(1.0, abs=1e-4)
    assert rewards["b"] == pytest.approx(0.5, abs=1e-4)

    env.step(raw)
    peeked = env.logger.peek()
    # one entry per step, the mechanism-shaped reward (not the intrinsic one)
    assert peeked.by_agent["a"].reward == pytest.approx([1.0, 1.0], abs=1e-4)
    assert peeked.by_agent["b"].reward == pytest.approx([0.5, 0.5], abs=1e-4)
    assert peeked.reward_mean == pytest.approx([0.75, 0.75], abs=1e-4)
    assert peeked.stock == pytest.approx([0.8, 0.6])
    assert peeked.iter == [1, 2]  # no iter at reset: aligned with per-step metrics
    assert peeked.mechanism_id == [0]
