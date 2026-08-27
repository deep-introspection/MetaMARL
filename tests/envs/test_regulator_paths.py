"""Action decoding and guard paths of ``RegulatorEnv``."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from core.envs.regulator import RegulatorEnv
from core.mechanism.algorithms.subsidy import SubsidyMechanism


class Dummy(RegulatorEnv):
    def _pre_reset(self, seed=None):
        pass

    def aggregate_rewards(self, ctxs):
        return 0.0


class Inner:
    def __init__(self):
        self.evaluated = 0
        self.runs = 0

    def run(self):
        self.runs += 1

    def evaluate(self):
        self.evaluated += 1


@pytest.mark.unit
def test_decode_accepts_list_1d_2d_and_torch(fake_world):
    env = Dummy(
        world=fake_world,
        optimizer=Inner(),
        mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1),
    )
    for action in ([0.2], np.array([0.2]), np.array([[0.2]]), torch.tensor([[0.2]])):
        mechs = env.action(action)
        assert len(mechs) == 1 and isinstance(mechs[0], SubsidyMechanism)
        assert mechs[0].subsidy == pytest.approx(0.1)
    with pytest.raises(TypeError, match="Unsupported"):
        env.action({"not": "supported"})


@pytest.mark.unit
def test_without_template_only_mechanisms_are_accepted(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner())
    m = SubsidyMechanism(subsidy=0.1, cost=0.1)
    assert env.action(m) == [m]
    assert env.action([m, m]) == [m, m]
    with pytest.raises(TypeError, match="template"):
        env.action(np.array([[0.1]]))


@pytest.mark.unit
def test_analytic_path_passes_action_through(fake_world):
    env = Dummy(world=fake_world, optimizer=None)
    x = np.array([[0.1]])
    assert env.action(x) is x
    with pytest.raises(NotImplementedError):
        env._step(x)


@pytest.mark.unit
def test_step_rejects_non_mechanism_lists(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner())
    with pytest.raises(TypeError, match="list\\[Mechanism\\]"):
        env._step([1, 2])


@pytest.mark.unit
def test_eval_seeds_trigger_evaluation_and_reset_gives_zeros(fake_world):
    inner = Inner()
    env = Dummy(
        world=fake_world,
        optimizer=inner,
        train_iters=2,
        seeds=[1],
        eval_seeds=[9],
        mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1),
    )
    env.observation_space = spaces.Box(0, 1, (3,), np.float32)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs, np.zeros(3))
    env.step(np.array([[0.5], [0.6]]))
    assert inner.runs == 2 and inner.evaluated == 1
    assert fake_world.flushed_status.count(None) == 0
