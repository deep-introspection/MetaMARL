"""``MultiAgentRegulatedEnv`` logs rewards exactly once per step (logging branch)."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from pydantic import Field

from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
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
    """Generic-path benchmark: uses the base ``_step`` (utility - penalty * violation)."""

    def _reset(self):
        self.S_t = {"stock": 1.0}
        return {aid: np.array([1.0], dtype=np.float32) for aid in self.agents}

    def transition_kernel(self, *, A_t, S_t):
        self.S_t = {"stock": S_t["stock"] - 0.1 * len(A_t)}
        self._log(("stock",), self.S_t["stock"])
        return self.S_t

    def intrinsic_utility(self, A_t):
        return {aid: float(a[0]) for aid, a in A_t.items()}

    def violation_signal(self, u_i, aid=None, **kwargs):
        return 1.0

    def penalty(self, u_i, **kwargs):
        return 0.25

    def _observation(self, agent_id, S_t):
        return np.array([S_t["stock"]], dtype=np.float32)

    def observation(self, agent_id, S_t):
        return self._observation(agent_id, S_t)

    def _is_truncated(self):
        return self.horizon is not None and self._t + 1 >= self.horizon


@pytest.mark.unit
def test_rewards_logged_once_and_penalty_subtracts_from_utility(fake_world):
    mechanism = SimpleNamespace(to_vector=lambda: [], param_names=lambda: [])
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
        world=fake_world, mechanism_id=0, agents=["a", "b"], schema=EnvSchema, horizon=5
    )
    env.reset()
    assert env.published_mechanism_assigned

    _, rewards, *_ = env.step({"a": np.array([0.8]), "b": np.array([0.4])})
    # shaped: u_i - penalty * violation = (0.55, 0.15); the base reward() then
    # shares the mean across agents (cooperative reward), as on dev
    assert rewards["a"] == pytest.approx(0.35) and rewards["b"] == pytest.approx(0.35)

    env.step({"a": np.array([0.8]), "b": np.array([0.4])})
    peeked = env.logger.peek()
    assert peeked.by_agent["a"].reward == pytest.approx([0.35, 0.35])  # one entry per step
    assert peeked.reward_mean == pytest.approx([0.35, 0.35])
    assert peeked.stock == pytest.approx([0.8, 0.6])
    assert peeked.iter == [1, 2]  # no iter at reset: aligned with per-step metrics
    assert peeked.mechanism_id == [0]
