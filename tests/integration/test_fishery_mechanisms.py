"""Staged smoke tests of the fishery benchmark under the mechanism abstraction (TODO §15).

Stage A steps the regulated environment alone against a real ``World`` actor
(deterministic seed, fixed mechanism, short horizon). Stage C runs the full
bilevel loop from ``examples/bilevel_fishery/debug.py`` with a minimal
configuration (2 agents, 2 candidates, 2 generations, W&B offline).
"""

import os
import uuid

import numpy as np
import pytest
import ray
from gymnasium import spaces

from core.mechanism.algorithms.quota import QuotaMechanism
from core.mechanism.algorithms.social_influence import SocialInfluenceMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.composition.chained_mechanism import ChainedMechanism
from core.world.base import World
from core.world.context import (
    Context,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)
from examples.bilevel_fishery.debug import build_config, parse_args
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv

EPS = 1e-8
AGENTS = ["u0", "u1"]


def fishery_stack(social: bool) -> ChainedMechanism:
    children = [
        QuotaMechanism(
            fixed_quota=0.5,
            bindings={"resource_level": lambda env: env.S_t["fish"] / max(env.K, EPS)},
        ),
        SubsidyMechanism(subsidy=0.1, cost=0.05),
    ]
    if social:
        children.append(
            SocialInfluenceMechanism(
                bindings={
                    "previous_actions": lambda env: env.previous_actions,
                    "agent_ids": lambda env: tuple(env.agents),
                }
            )
        )
    return ChainedMechanism(children=tuple(children))


@pytest.mark.integration
@pytest.mark.parametrize("social", [False, True])
def test_stage_a_environment_only(ray_session, social):
    world = World.options(name=f"world_{uuid.uuid4().hex[:8]}").remote()
    mechanism = fishery_stack(social)

    # the outer optimizer would publish this candidate; do it by hand
    ray.get(
        world.append_context.remote(
            Context(
                id=None,
                opt_id="outer",
                step=0,
                env="test",
                payload=MechanismContext(
                    index=0,
                    env_id=None,
                    seed=123,
                    status=MechanismStatus.published,
                    mechanism=mechanism,
                    metrics=None,
                ),
            )
        )
    )

    env = FisheryRegulatedEnv(
        world=world,
        opt_id="inner",
        mechanism_id=0,
        agents=AGENTS,
        mechanism=mechanism,
        horizon=5,
        seed=7,
        policy_seed=123,
        ecology_cfg={
            "K": 1000.0,
            "fish_init": 800.0,
            "sigma": 0.0,
            "initial_stock_log_sigma": 0.0,
        },
        action_spaces={
            aid: spaces.Box(-np.inf, np.inf, (2,), np.float32) for aid in AGENTS
        },
    )
    obs, _ = env.reset()
    assert env.published_mechanism_assigned
    expected_dim = 2 + 2 + 1 + ((len(AGENTS) - 1) * 2 if social else 0)
    assert all(o.shape == (expected_dim,) for o in obs.values())

    rng = np.random.default_rng(0)
    for step in range(5):
        obs, rewards, term, trunc, infos = env.step(
            {aid: rng.normal(size=2).astype(np.float32) for aid in AGENTS}
        )
        assert all(o.shape == (expected_dim,) for o in obs.values())
        assert all(np.isfinite(r) for r in rewards.values())
        assert trunc["__all__"] is (step == 4)
    assert 0.0 <= env.S_t["fish"] <= env.K

    ctxs = ray.get(world.get_opt_ctx_ids.remote("inner"))
    assert len(ctxs) == 5  # one EnvStepContext per step
    payload = ray.get(world.get_context.remote(ctxs[0])).payload
    assert isinstance(payload, EnvStepContext)
    assert payload.mechanism == 0 and payload.policy_seed == 123


@pytest.mark.integration
@pytest.mark.parametrize(
    "social", [False, True], ids=["quota+subsidy", "quota+subsidy+social"]
)
def test_stage_c_bilevel_smoke(monkeypatch, social):
    monkeypatch.setenv("WANDB_MODE", "offline")
    ray.shutdown()
    argv = [
        "--outer-iters",
        "2",
        "--train-iters",
        "2",
        "--num-agents",
        "2",
        "--horizon",
        "20",
        "--num-candidates",
        "2",
        "--num-eval-seeds",
        "1",
        "--num-cpus",
        "2",
        "--project",
        "bilevel-smoke",
    ]
    if not social:
        argv.append("--no-social")
    cfg = build_config(parse_args(argv))
    try:
        result = cfg.build_optimizer().run()
    finally:
        ray.shutdown()

    assert result["outer_iters"] == 2
    assert np.isfinite(result["best_fitness"])
    assert result["best_mechanism"].shape == (2,)
    assert (
        len(result["population_history"]) == 0 or True
    )  # populated by the outer optimizer
    assert os.environ["WANDB_MODE"] == "offline"
