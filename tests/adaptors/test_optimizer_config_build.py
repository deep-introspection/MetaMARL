"""Unit tests for ``core.adaptors.ray.optimizer_config.RayOptimizerConfig``.

Covers, without a Ray runtime:

- the pure builders (``evaluation`` seed derivation, ``debugging`` env
  scaling, ``agents``, ``model``) and the helpers ``_parse_episode_identity``
  and ``_seeded_xavier_uniform``;
- the deferred one-line ``RLlibConfigOp`` bodies, replayed against a
  ``MagicMock`` standing in for an ``AlgorithmConfig``;
- ``_apply_agents_to_rllib`` against a real PPO ``AlgorithmConfig`` (module
  naming, spaces written into ``env_config``, ``policy_mapping_fn``);
- ``build_optimizer`` with the ``FakeWorld`` fixture, ``register_env`` and
  ``RayOptimizer`` patched in the module namespace, including the train/eval
  layouts of the inner ``env_creator`` and the reporter plumbing (optimizer
  level reporter built from ``reporter_cfg``, env-level config copied into
  every environment).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

import core.adaptors.ray.optimizer_config as optimizer_config_module
from core.adaptors.ray.optimizer_config import RayOptimizerConfig, RLlibConfigOp
from core.callbacks import _evaluate_with_fixed_duration_once
from core.metrics.schemas import MetricSchema
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
ACT_SPACE = Discrete(2)


def fisher_specs(count=2, policy="fisher"):
    return {
        "fisher": {
            "count": count,
            "policy": policy,
            "observation_space": OBS_SPACE,
            "action_space": ACT_SPACE,
        }
    }


class RecordingEnv:
    """Environment class recording the constructor kwargs it received."""

    instances: list[RecordingEnv] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingEnv.instances.append(self)


class RecordingReporter(Reporter):
    """Reporter that only records what it is asked to render."""

    def __init__(self, label):
        self.label = label
        self.reports = []

    def _report(self, query, series):
        self.reports.append((query.title, series))

    def close(self):
        pass


class RecordingReporterConfig(ReporterConfig):
    """``ReporterConfig`` building ``RecordingReporter`` instances."""

    built: list[RecordingReporter] = []

    def build(self, *, label=None):
        reporter = RecordingReporter(label)
        RecordingReporterConfig.built.append(reporter)
        return reporter


class OptSchema(MetricSchema):
    pass


class EnvSchema(MetricSchema):
    pass


OPT_QUERY = Query(title="opt", x=("iter",), y=("iter",))
ENV_QUERY = Query(title="env", x=("iter",), y=("iter",))


class FakeEnvContext(dict):
    """``EnvContext`` stand-in: a dict with a ``worker_index`` attribute."""

    def __init__(self, *args, worker_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_index = worker_index


# --------------------------------------------------------------------------- #
# Construction and pure builders
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_subclass_without_algo_class_is_rejected():
    class _NoAlgo(RayOptimizerConfig):
        pass

    with pytest.raises(ValueError, match="must define `algo_class`"):
        _NoAlgo()


@pytest.mark.unit
def test_evaluation_with_explicit_seeds_records_op_and_forces_manual_eval():
    cfg = PPOptimizerConfig()
    out = cfg.evaluation(seeds=[3, "4"], evaluation_config={"env_config": {"x": 1}})

    assert out is cfg
    assert cfg.eval_seeds == [3, 4]
    op = cfg._cfg_ops["_evaluation_rllib"]
    assert op.kwargs["evaluation_interval"] is None
    assert op.kwargs["evaluation_parallel_to_training"] is False
    assert op.kwargs["evaluation_config"]["env_config"] == {"x": 1, "mode": "eval"}


@pytest.mark.unit
def test_evaluation_derives_seeds_from_base_seed_deterministically():
    a = PPOptimizerConfig().evaluation(base_seed=42, num_seeds=3)
    b = PPOptimizerConfig().evaluation(base_seed=42, num_seeds=3)
    c = PPOptimizerConfig().evaluation(base_seed=43, num_seeds=3)

    assert len(a.eval_seeds) == 3
    assert a.eval_seeds == b.eval_seeds
    assert a.eval_seeds != c.eval_seeds
    assert all(isinstance(s, int) for s in a.eval_seeds)
    # ``num_seeds`` defaults to one.
    assert len(PPOptimizerConfig().evaluation(base_seed=1).eval_seeds) == 1


@pytest.mark.unit
def test_evaluation_without_seeds_leaves_eval_seeds_unset():
    cfg = PPOptimizerConfig().evaluation()
    assert cfg.eval_seeds is None
    assert cfg._cfg_ops["_evaluation_rllib"].kwargs["evaluation_config"] == {
        "env_config": {"mode": "eval"}
    }


@pytest.mark.unit
def test_evaluation_rejects_empty_seeds_and_orphan_num_seeds():
    with pytest.raises(ValueError, match="at least one seed"):
        PPOptimizerConfig().evaluation(seeds=[])
    with pytest.raises(ValueError, match="requires `base_seed`"):
        PPOptimizerConfig().evaluation(num_seeds=2)


@pytest.mark.unit
def test_debugging_scales_env_count_by_seeds_when_env_runners_recorded():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)
    assert cfg.num_mechanisms == 4

    cfg.debugging(seed=7, num_seeds=3, log_level="ERROR")

    assert len(cfg.seeds) == 3
    assert cfg.num_mechanisms == 4
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 12
    debug_op = cfg._cfg_ops["_debugging_rllib"]
    assert debug_op.kwargs == {"seed": 7, "log_level": "ERROR"}


@pytest.mark.unit
def test_debugging_without_env_runners_or_seed_does_not_scale():
    cfg = PPOptimizerConfig().debugging(seed=7, num_seeds=2)
    assert len(cfg.seeds) == 2
    assert cfg.num_mechanisms is None

    # ``env_runners`` called afterwards records the raw count, unscaled.
    cfg.env_runners(num_envs_per_env_runner=5)
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 5

    unseeded = PPOptimizerConfig().env_runners(num_envs_per_env_runner=5).debugging()
    assert unseeded.seeds == []
    assert unseeded._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 5


@pytest.mark.unit
def test_env_runners_defaults_num_mechanisms_to_one():
    cfg = PPOptimizerConfig().env_runners(num_env_runners=2)
    assert cfg.num_mechanisms == 1


@pytest.mark.unit
def test_agents_stores_specs():
    specs = fisher_specs()
    cfg = PPOptimizerConfig()
    assert cfg.agents(specs) is cfg
    assert cfg.agent_specs is specs


@pytest.mark.unit
def test_parse_episode_identity_extracts_fields_and_ignores_bare_segments():
    cfg = PPOptimizerConfig()
    identity = cfg._parse_episode_identity("env=3|m=1|ps=101|ss=202|raw=a=b|junk")
    assert identity == {"env": "3", "m": "1", "ps": "101", "ss": "202", "raw": "a=b"}


@pytest.mark.unit
def test_parse_episode_identity_requires_all_keys():
    cfg = PPOptimizerConfig()
    with pytest.raises(RuntimeError, match="missing identity keys"):
        cfg._parse_episode_identity("env=3|m=1|raw=x")


@pytest.mark.unit
def test_seeded_xavier_uniform_none_returns_default_name():
    assert PPOptimizerConfig()._seeded_xavier_uniform(None) == "xavier_uniform_"


@pytest.mark.unit
def test_seeded_xavier_uniform_is_deterministic_and_leaves_global_rng_alone():
    cfg = PPOptimizerConfig()
    init_a = cfg._seeded_xavier_uniform(123)
    init_b = cfg._seeded_xavier_uniform(123)
    init_c = cfg._seeded_xavier_uniform(124)

    torch.manual_seed(0)
    state_before = torch.random.get_rng_state()

    a1, a2 = torch.empty(4, 3), torch.empty(4, 3)
    b1, b2 = torch.empty(4, 3), torch.empty(4, 3)
    c1 = torch.empty(4, 3)
    init_a(a1)
    init_a(a2)
    init_b(b1)
    init_b(b2)
    init_c(c1)

    # Same seed, same layer order -> identical weights per layer position.
    assert torch.equal(a1, b1) and torch.equal(a2, b2)
    # Successive layers of one closure differ (counter advances).
    assert not torch.equal(a1, a2)
    # Different seeds differ.
    assert not torch.equal(a1, c1)
    # The global RNG state is restored after each call.
    assert torch.equal(torch.random.get_rng_state(), state_before)


# --------------------------------------------------------------------------- #
# Deferred RLlib ops replayed on a mocked AlgorithmConfig
# --------------------------------------------------------------------------- #

# ``(builder name, RLlib method name, op key)``
PASSTHROUGH_BUILDERS = [
    ("validate", "validate", "validate"),
    ("get_config_for_module", "get_config_for_module", "get_config_for_module"),
    ("python_environment", "python_environment", "python_environment"),
    ("resources", "resources", "resources"),
    ("framework", "framework", "framework"),
    ("api_stack", "api_stack", "api_stack"),
    ("env_runners", "env_runners", "_env_runners"),
    ("learners", "learners", "learners"),
    ("callbacks", "callbacks", "callbacks"),
    ("offline_data", "offline_data", "offline_data"),
    ("multi_agent", "multi_agent", "multi_agent"),
    ("_reporting_rllib", "reporting", "_reporting_rllib"),
    ("checkpointing", "checkpointing", "checkpointing"),
    ("fault_tolerance", "fault_tolerance", "fault_tolerance"),
    ("rl_module", "rl_module", "rl_module"),
    ("experimental", "experimental", "experimental"),
    ("freeze", "freeze", "freeze"),
    ("training", "training", "training"),
]


@pytest.mark.unit
@pytest.mark.parametrize("builder, rllib_method, op_key", PASSTHROUGH_BUILDERS)
def test_deferred_builders_replay_on_rllib_config(builder, rllib_method, op_key):
    cfg = PPOptimizerConfig()
    kwargs = {"num_envs_per_env_runner": 2} if builder == "env_runners" else {"opt": 1}

    out = getattr(cfg, builder)(**kwargs)

    assert out is cfg
    op = cfg._cfg_ops[op_key]
    assert isinstance(op, RLlibConfigOp)
    assert op.kwargs == kwargs

    rllib_cfg = MagicMock(name="AlgorithmConfig")
    result = op(rllib_cfg)
    getattr(rllib_cfg, rllib_method).assert_called_once_with(**kwargs)
    assert result is getattr(rllib_cfg, rllib_method).return_value


@pytest.mark.unit
def test_evaluation_and_debugging_ops_replay_on_rllib_config():
    cfg = (
        PPOptimizerConfig()
        .evaluation(evaluation_duration=3)
        .debugging(seed=5, num_seeds=1, log_level="INFO")
    )
    rllib_cfg = MagicMock(name="AlgorithmConfig")

    cfg._cfg_ops["_evaluation_rllib"](rllib_cfg)
    cfg._cfg_ops["_debugging_rllib"](rllib_cfg)

    rllib_cfg.evaluation.assert_called_once_with(
        evaluation_duration=3,
        evaluation_config={"env_config": {"mode": "eval"}},
        evaluation_interval=None,
        evaluation_parallel_to_training=False,
    )
    rllib_cfg.debugging.assert_called_once_with(seed=5, log_level="INFO")


@pytest.mark.unit
def test_reporting_stores_schema_and_queries_then_defers_rllib_kwargs():
    cfg = PPOptimizerConfig()
    out = cfg.reporting(
        queries=(OPT_QUERY,), schema=OptSchema, metrics_num_episodes_for_smoothing=5
    )

    assert out is cfg
    assert cfg._reporting_schema is OptSchema
    assert cfg._reporting_queries == (OPT_QUERY,)
    op = cfg._cfg_ops["_reporting_rllib"]
    assert op.kwargs == {"metrics_num_episodes_for_smoothing": 5}
    rllib_cfg = MagicMock(name="AlgorithmConfig")
    op(rllib_cfg)
    rllib_cfg.reporting.assert_called_once_with(metrics_num_episodes_for_smoothing=5)


@pytest.mark.unit
def test_reporting_with_none_leaves_previous_declaration():
    cfg = PPOptimizerConfig().reporting(queries=(OPT_QUERY,), schema=OptSchema)
    cfg.reporting(queries=None, schema=None)
    assert cfg._reporting_schema is OptSchema
    assert cfg._reporting_queries == (OPT_QUERY,)
    assert cfg._cfg_ops["_reporting_rllib"].kwargs == {}


@pytest.mark.unit
def test_model_merges_kwargs_into_rllib_model_dict():
    cfg = PPOptimizerConfig().model(fcnet_hiddens=[8, 8])
    rllib_cfg = MagicMock(name="AlgorithmConfig")
    rllib_cfg.model = {"existing": True}

    assert cfg._cfg_ops["model"](rllib_cfg) is rllib_cfg
    assert rllib_cfg.model == {"existing": True, "fcnet_hiddens": [8, 8]}


@pytest.mark.unit
def test_repeated_builder_call_overwrites_previous_op():
    cfg = PPOptimizerConfig().training(lr=1e-3).training(lr=5e-4)
    assert cfg._cfg_ops["training"].kwargs == {"lr": 5e-4}


# --------------------------------------------------------------------------- #
# _apply_agents_to_rllib against a real PPO config
# --------------------------------------------------------------------------- #


def make_applied_config(num_envs=4, seeds=(11, 22), count=2):
    cfg = PPOptimizerConfig().agents(fisher_specs(count=count))
    cfg.seeds = list(seeds)
    cfg.rllib_cfg = PPO.get_default_config().env_runners(
        num_envs_per_env_runner=num_envs
    )
    return cfg


@pytest.mark.unit
def test_apply_agents_declares_one_module_per_mechanism_and_seed():
    cfg = make_applied_config(num_envs=4, seeds=(11, 22), count=2)

    agents = cfg._apply_agents_to_rllib()

    assert agents == ["fisher:0", "fisher:1"]
    expected_modules = {
        "fisher_m0_s11",
        "fisher_m1_s11",
        "fisher_m0_s22",
        "fisher_m1_s22",
    }
    spec = cfg.rllib_cfg.rl_module_spec
    assert isinstance(spec, MultiRLModuleSpec)
    assert set(spec.rl_module_specs) == expected_modules
    module = spec.rl_module_specs["fisher_m0_s11"]
    # RLlib deep-copies the spec, so compare spaces by value.
    assert module.observation_space == OBS_SPACE
    assert module.action_space == ACT_SPACE
    assert module.model_config.vf_share_layers is False
    assert callable(module.model_config.fcnet_kernel_initializer)
    assert module.model_config.fcnet_bias_initializer == "zeros_"

    assert set(cfg.rllib_cfg.policies) == expected_modules
    assert set(cfg.rllib_cfg.policies_to_train) == expected_modules
    assert cfg.env_config["observation_spaces"] == {
        "fisher:0": OBS_SPACE,
        "fisher:1": OBS_SPACE,
    }
    assert cfg.env_config["action_spaces"] == {
        "fisher:0": ACT_SPACE,
        "fisher:1": ACT_SPACE,
    }


@pytest.mark.unit
def test_apply_agents_policy_mapping_reads_episode_identity():
    cfg = make_applied_config(num_envs=4, seeds=(11, 22))
    cfg._apply_agents_to_rllib()
    mapping = cfg.rllib_cfg.policy_mapping_fn

    episode = MagicMock(id_="env=3|m=1|ps=22|ss=22|raw=abc")
    assert mapping("fisher:1", episode) == "fisher_m1_s22"

    unknown = MagicMock(id_="env=3|m=7|ps=22|ss=22|raw=abc")
    with pytest.raises(RuntimeError, match="Unknown policy"):
        mapping("fisher:0", unknown)


@pytest.mark.unit
def test_apply_agents_rejects_env_count_not_divisible_by_seeds():
    cfg = make_applied_config(num_envs=3, seeds=(11, 22))
    with pytest.raises(ValueError, match="must be divisible"):
        cfg._apply_agents_to_rllib()


@pytest.mark.unit
def test_apply_agents_without_seeds_uses_single_unseeded_layout():
    cfg = make_applied_config(num_envs=2, seeds=(), count=1)
    cfg.seeds = None
    # ``num_seeds`` falls back to one but the loop over ``self.seeds`` needs an
    # iterable, so the documented "null seed" case is a ``None`` seed entry.
    cfg.seeds = [None]

    agents = cfg._apply_agents_to_rllib()

    assert agents == ["fisher:0"]
    assert set(cfg.rllib_cfg.rl_module_spec.rl_module_specs) == {
        "fisher_m0_sNone",
        "fisher_m1_sNone",
    }
    module = cfg.rllib_cfg.rl_module_spec.rl_module_specs["fisher_m0_sNone"]
    assert module.model_config.fcnet_kernel_initializer == "xavier_uniform_"


# --------------------------------------------------------------------------- #
# build_optimizer
# --------------------------------------------------------------------------- #


class StubRayOptimizer:
    """Records the frozen config handed over by ``build_optimizer``."""

    def __init__(self, config):
        self.config = config
        self.world = None
        self.reporting = None
        self.opt_id = None

    def set_id(self, opt_id):
        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID already set")
        self.opt_id = opt_id


@pytest.fixture
def build_env(monkeypatch):
    """Patch ``register_env`` and ``RayOptimizer`` in the config module."""
    registered = {}
    monkeypatch.setattr(
        optimizer_config_module,
        "register_env",
        lambda name, creator: registered.setdefault(name, creator),
    )
    monkeypatch.setattr(optimizer_config_module, "RayOptimizer", StubRayOptimizer)
    RecordingEnv.instances = []
    RecordingReporterConfig.built = []
    return registered


def make_buildable_config(
    *, mechanisms=2, num_seeds=2, eval_seeds=(7,), with_reporting=False
):
    cfg = (
        PPOptimizerConfig()
        .environment(
            env=RecordingEnv,
            horizon=5,
            env_config={"alpha": 0.5},
            queries=(ENV_QUERY,) if with_reporting else None,
            schema=EnvSchema if with_reporting else None,
        )
        .env_runners(num_envs_per_env_runner=mechanisms)
        .debugging(seed=3, num_seeds=num_seeds)
        .agents(fisher_specs())
    )
    if eval_seeds is not None:
        cfg.evaluation(seeds=list(eval_seeds), evaluation_duration=1)
    if with_reporting:
        cfg.reporting(queries=(OPT_QUERY,), schema=OptSchema)
        cfg.reporter_cfg = RecordingReporterConfig(project="unit")
    return cfg


@pytest.mark.unit
def test_build_optimizer_resolves_config_and_registers_env(build_env, fake_world):
    cfg = make_buildable_config(mechanisms=2, num_seeds=2, eval_seeds=(7, 8))

    opt = cfg.build_optimizer(world=fake_world, world_name="w")

    # Evaluation kwargs patched: one runner per (eval seed, train seed) pair.
    eval_kwargs = cfg._cfg_ops["_evaluation_rllib"].kwargs
    assert eval_kwargs["evaluation_num_env_runners"] == 4
    assert eval_kwargs["evaluation_duration"] == 4 * 2
    assert (
        eval_kwargs["custom_evaluation_function"] is _evaluate_with_fixed_duration_once
    )

    # Ops replayed onto a PPO config, env registered and pointed to.
    assert cfg.rllib_cfg is not None
    assert cfg.rllib_cfg.num_envs_per_env_runner == 4
    assert cfg.rllib_cfg.evaluation_num_env_runners == 4
    assert (
        cfg.rllib_cfg.custom_evaluation_function is _evaluate_with_fixed_duration_once
    )
    assert len(build_env) == 1
    env_name = next(iter(build_env))
    assert env_name.startswith("regulated_env_")
    assert cfg.rllib_cfg.env == env_name
    assert cfg.rllib_cfg.env_config["alpha"] == 0.5
    assert "observation_spaces" in cfg.rllib_cfg.env_config

    # Optimizer built on a deep copy, registered in the World. Note that
    # ``copy(copy_frozen=True)`` calls ``freeze()``, which on
    # ``RayOptimizerConfig`` is the deferred RLlib mutator: it records a
    # ``freeze`` op on the copy instead of freezing the Python object.
    assert isinstance(opt, StubRayOptimizer)
    assert opt.config is not cfg
    assert opt.config._is_frozen is False
    assert "freeze" in opt.config._cfg_ops
    assert "freeze" not in cfg._cfg_ops
    assert opt.world is fake_world
    # No ``reporter_cfg``: the optimizer-level reporter is ``None``.
    assert opt.reporting is None
    assert opt.opt_id == fake_world.opt_ids[-1]
    assert cfg.world_name == "w"


@pytest.mark.unit
def test_build_optimizer_builds_reporter_and_forwards_env_reporting(
    build_env, fake_world
):
    cfg = make_buildable_config(with_reporting=True)

    opt = cfg.build_optimizer(world=fake_world, world_name="w")

    # Optimizer-level reporter: built once, labelled with the optimizer class
    # name, carrying the optimizer schema and queries.
    assert isinstance(opt.reporting, RecordingReporter)
    assert RecordingReporterConfig.built == [opt.reporting]
    # The label is ``opt_class.__name__`` (the stub class under this patch).
    assert opt.reporting.label == cfg.opt_class.__name__ == "StubRayOptimizer"
    assert opt.reporting.schema is OptSchema
    assert opt.reporting.queries == (OPT_QUERY,)

    # Each environment receives a *copy* of the reporter config plus the
    # env-level queries and schema, and the registered env name.
    env_creator = next(iter(build_env.values()))
    env_a = env_creator(FakeEnvContext({}, worker_index=1))
    env_b = env_creator(FakeEnvContext({}, worker_index=1))
    for env in (env_a, env_b):
        assert isinstance(env.kwargs["reporter_cfg"], RecordingReporterConfig)
        assert env.kwargs["reporter_cfg"] is not cfg.reporter_cfg
        assert env.kwargs["reporter_cfg"].project_name == "unit"
        assert env.kwargs["queries"] == (ENV_QUERY,)
        assert env.kwargs["schema"] is EnvSchema
        assert env.kwargs["env_name"] == cfg.rllib_cfg.env
    assert env_a.kwargs["reporter_cfg"] is not env_b.kwargs["reporter_cfg"]
    # The env creator does not build reporters itself.
    assert len(RecordingReporterConfig.built) == 1


@pytest.mark.unit
def test_build_optimizer_env_creator_train_layout(build_env, fake_world):
    cfg = make_buildable_config(mechanisms=2, num_seeds=2)
    cfg.build_optimizer(world=fake_world, world_name="w")
    env_creator = next(iter(build_env.values()))
    seeds = cfg.seeds

    # Four training envs per runner: mechanism k % 2, seed index k // 2, then
    # the counter wraps for the next runner's worth of envs.
    expected = [
        (0, seeds[0]),
        (1, seeds[0]),
        (0, seeds[1]),
        (1, seeds[1]),
        (0, seeds[0]),
    ]
    for mechanism_idx, seed in expected:
        env = env_creator(FakeEnvContext({"alpha": 0.5}, worker_index=1))
        assert isinstance(env, RecordingEnv)
        assert env.kwargs["mechanism_id"] == mechanism_idx
        assert env.kwargs["seed"] == seed
        assert env.kwargs["policy_seed"] == seed

    env = RecordingEnv.instances[0]
    assert env.kwargs["world"] is fake_world
    assert env.kwargs["opt_id"] == fake_world.opt_ids[-1]
    assert env.kwargs["agents"] == ["fisher:0", "fisher:1"]
    assert env.kwargs["alpha"] == 0.5
    # ``mode`` is only read (default ``train``), never written back.
    assert "mode" not in env.kwargs
    # Without reporting configured, the env-level reporting kwargs are empty.
    assert env.kwargs["reporter_cfg"] is None
    assert env.kwargs["queries"] is None and env.kwargs["schema"] is None
    assert env.kwargs["env_name"] == cfg.rllib_cfg.env


@pytest.mark.unit
def test_build_optimizer_env_creator_eval_layout(build_env, fake_world):
    cfg = make_buildable_config(mechanisms=2, num_seeds=2, eval_seeds=(70, 80))
    cfg.build_optimizer(world=fake_world, world_name="w")
    env_creator = next(iter(build_env.values()))
    train_seeds, eval_seeds = cfg.seeds, cfg.eval_seeds

    # Runner index r = worker_index - 1: policy seed r % 2, eval seed r // 2;
    # mechanisms cycle with the per-process eval counter.
    cases = [
        (1, 0, train_seeds[0], eval_seeds[0]),
        (2, 1, train_seeds[1], eval_seeds[0]),
        (3, 0, train_seeds[0], eval_seeds[1]),
        (4, 1, train_seeds[1], eval_seeds[1]),
    ]
    for worker_index, mechanism_idx, policy_seed, env_seed in cases:
        env = env_creator(FakeEnvContext({"mode": "eval"}, worker_index=worker_index))
        assert env.kwargs["mode"] == "eval"
        assert env.kwargs["mechanism_id"] == mechanism_idx
        assert env.kwargs["policy_seed"] == policy_seed
        assert env.kwargs["seed"] == env_seed

    # Train and eval counters are independent.
    train_env = env_creator(FakeEnvContext({}, worker_index=1))
    assert train_env.kwargs["mechanism_id"] == 0


@pytest.mark.unit
def test_build_optimizer_without_evaluation_skips_eval_patching(build_env, fake_world):
    cfg = make_buildable_config(eval_seeds=None)
    cfg.build_optimizer(world=fake_world, world_name="w")
    assert "_evaluation_rllib" not in cfg._cfg_ops
    assert cfg.rllib_cfg.custom_evaluation_function is None

    # Unseeded evaluation envs fall back to ``None`` seeds.
    env_creator = next(iter(build_env.values()))
    env = env_creator(FakeEnvContext({"mode": "eval"}, worker_index=1))
    assert env.kwargs["seed"] is None
    assert env.kwargs["policy_seed"] == cfg.seeds[0]


@pytest.mark.unit
def test_build_optimizer_reuses_resolved_rllib_config(build_env, fake_world):
    cfg = make_buildable_config()
    cfg.build_optimizer(world=fake_world, world_name="w")
    resolved = cfg.rllib_cfg
    cfg._cfg_ops["training"] = RLlibConfigOp(
        fn=lambda c, **k: pytest.fail("ops must not be replayed twice"),
        args=(),
        kwargs={},
    )

    cfg.build_optimizer(world=fake_world, world_name="w")

    assert cfg.rllib_cfg is resolved
    assert len(build_env) == 2
    assert len(fake_world.opt_ids) == 2


@pytest.mark.unit
def test_build_optimizer_requires_world_name_with_world(build_env, fake_world):
    cfg = make_buildable_config()
    with pytest.raises(ValueError, match="world_name must be provided"):
        cfg.build_optimizer(world=fake_world)


@pytest.mark.unit
def test_build_optimizer_requires_opt_class(build_env, fake_world):
    cfg = make_buildable_config()
    cfg.opt_class = None
    with pytest.raises(ValueError, match="no opt_class"):
        cfg.build_optimizer(world=fake_world, world_name="w")


@pytest.mark.unit
def test_build_optimizer_without_world_leaves_env_creator_unbound(build_env):
    # Documented limitation: with no World, ``opt_id`` and ``agents`` are
    # never bound, so the first env instantiation fails.
    cfg = make_buildable_config()
    cfg.agent_specs = None

    opt = cfg.build_optimizer(world=None)

    assert opt.opt_id is None
    assert opt.world is None
    env_creator = next(iter(build_env.values()))
    with pytest.raises(NameError):
        env_creator(FakeEnvContext({}, worker_index=1))
