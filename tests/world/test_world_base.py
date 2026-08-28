"""Unit tests for the ``World`` shared-state container.

``World`` is decorated with ``@ray.remote``; these tests bypass the actor
machinery and instantiate the undecorated class through
``World.__ray_metadata__.modified_class`` so no Ray runtime is needed. Every
registry (``_contexts``, ``_mechanism_registry``, ``_opt_ctx_map``) is
exercised through the public accessors and mutators, including the error
paths (singleton violation, duplicate IDs, missing ``env_id``, unknown context
on update).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.world.base import World
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

WorldCls = World.__ray_metadata__.modified_class


class _OtherSchema(ContextSchema):
    """A payload type unrelated to mechanisms or env steps."""

    value: int = 0


def _mech(
    index: int = 0,
    seed: int | None = 0,
    status: MechanismStatus = MechanismStatus.published,
    env_id: str | None = None,
) -> MechanismContext:
    return MechanismContext(
        index=index,
        env_id=env_id,
        seed=seed,
        status=status,
        mechanism=SimpleNamespace(name="fake"),
        metrics=None,
    )


def _step(step: int, status: MechanismStatus = MechanismStatus.train) -> EnvStepContext:
    return EnvStepContext(
        env_id=0,
        seed=0,
        policy_seed=0,
        status=status,
        mechanism=0,
        observation=[float(step)],
        observation_map=None,
        reward=1.0,
        action=None,
        info={},
    )


def _ctx(payload: ContextSchema, opt_id: str | None = "opt", step: int = 0) -> Context:
    return Context(id=None, opt_id=opt_id, step=step, env="env", payload=payload)


@pytest.fixture
def world() -> WorldCls:
    return WorldCls()


# ---------------------------------------------------------------------------
# Construction and trivial accessors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_creates_empty_registries(world):
    assert world.get_ctx_registry() == {}
    assert world.get_mechanism_registry() == {}
    assert list(world.get_opt_registry()) == []
    assert world.get_ctx_ids() == set()
    assert world.get_opt_ids() == set()
    assert world.reporting is None
    assert world.get_context("missing") is None
    assert world.get_opt_ctx_ids("missing") == []


@pytest.mark.unit
def test_init_stores_reporting_handle():
    sentinel = object()
    assert WorldCls(reporting=sentinel).reporting is sentinel


@pytest.mark.unit
def test_copy_and_deepcopy_return_same_instance(world):
    import copy

    assert copy.copy(world) is world
    assert copy.deepcopy(world) is world


# ---------------------------------------------------------------------------
# append_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_append_context_registers_mechanism_and_optimizer(world):
    ctx = _ctx(_mech(index=3, seed=7))
    cid = world.append_context(ctx)

    assert ctx.id == cid
    assert world.get_context(cid) is ctx
    assert world.get_ctx_ids() == {cid}
    assert world.get_mechanism_registry() == {cid: ctx.payload}
    assert world.get_opt_ctx_ids("opt") == [cid]
    assert world.get_opt_ids() == {"opt"}
    assert "opt" in world.get_opt_registry()
    assert world.get_mechanism_by_index(cid) is ctx.payload


@pytest.mark.unit
def test_append_context_without_opt_id_and_non_mechanism_payload(world):
    ctx = _ctx(_OtherSchema(), opt_id=None)
    cid = world.append_context(ctx)

    assert world.get_context(cid) is ctx
    assert world.get_mechanism_registry() == {}
    assert world.get_opt_ids() == set()


@pytest.mark.unit
def test_append_context_overwrites_caller_supplied_id(world):
    ctx = Context(id="custom", opt_id="opt", step=0, env="e", payload=_OtherSchema())
    cid = world.append_context(ctx)
    assert cid != "custom"
    assert ctx.id == cid


@pytest.mark.unit
def test_append_context_rejects_existing_id(world):
    first = _ctx(_OtherSchema())
    cid = world.append_context(first)
    dup = Context(id=cid, opt_id="opt", step=0, env="e", payload=_OtherSchema())
    with pytest.raises(ValueError, match="already exists"):
        world.append_context(dup)


@pytest.mark.unit
def test_append_context_singleton_enforced(world):
    world.append_context(_ctx(_OtherSchema()), singleton=True)
    with pytest.raises(ValueError, match="Singleton Context Schema _OtherSchema"):
        world.append_context(_ctx(_OtherSchema()), singleton=True)
    # A different payload type is still accepted as a singleton.
    world.append_context(_ctx(_step(0)), singleton=True)


@pytest.mark.unit
def test_validate_ctx_schema_exists_ignores_subclasses(world):
    class _Child(_OtherSchema):
        pass

    world.append_context(_ctx(_Child()))
    # Exact type comparison: parent schema is not considered present.
    world._validate_ctx_schema_exists(_OtherSchema)
    with pytest.raises(ValueError):
        world._validate_ctx_schema_exists(_Child)


# ---------------------------------------------------------------------------
# _set_new_opt_id
# ---------------------------------------------------------------------------
# ``register_optimizer`` was removed on this branch: ``build_optimizer`` calls
# ``_set_new_opt_id`` directly with the optimizer's ID.


@pytest.mark.unit
def test_set_new_opt_id_with_explicit_id_is_idempotent(world):
    assert world._set_new_opt_id("reg") == "reg"
    assert world.get_opt_ctx_ids("reg") == []
    # Idempotent: the existing (possibly populated) list is kept.
    world.append_context(_ctx(_OtherSchema(), opt_id="reg"))
    assert world._set_new_opt_id("reg") == "reg"
    assert len(world.get_opt_ctx_ids("reg")) == 1


@pytest.mark.unit
def test_set_new_opt_id_none_draws_fresh_uuid(world):
    a = world._set_new_opt_id(None)
    b = world._set_new_opt_id(None)
    assert a != b
    assert world.get_opt_ids() == {a, b}
    assert not hasattr(world, "register_optimizer")


# ---------------------------------------------------------------------------
# set_new_context / update_context / remove_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_new_context_generates_id_and_tracks_mechanism(world):
    ctx = _ctx(_mech(env_id="env-0"))
    cid = world.set_new_context(ctx)
    assert ctx.id == cid
    assert world.get_mechanism_registry() == {cid: ctx.payload}
    assert world.get_opt_ctx_ids("opt") == [cid]


@pytest.mark.unit
def test_set_new_context_appends_to_existing_optimizer_list(world):
    first = world.set_new_context(_ctx(_OtherSchema(), opt_id="opt"))
    second = world.set_new_context(_ctx(_OtherSchema(), opt_id="opt"))
    assert world.get_opt_ctx_ids("opt") == [first, second]
    assert world.get_opt_ids() == {"opt"}


@pytest.mark.unit
def test_set_new_context_keeps_caller_id(world):
    ctx = Context(id="keep", opt_id=None, step=0, env="e", payload=_OtherSchema())
    assert world.set_new_context(ctx) == "keep"
    assert world.get_context("keep") is ctx
    with pytest.raises(ValueError, match="already exists"):
        world.set_new_context(
            Context(id="keep", opt_id=None, step=0, env="e", payload=_OtherSchema())
        )


@pytest.mark.unit
def test_set_new_context_singleton_and_missing_env_id(world):
    world.set_new_context(_ctx(_OtherSchema()), singleton=True)
    with pytest.raises(ValueError, match="Singleton"):
        world.set_new_context(_ctx(_OtherSchema()), singleton=True)
    with pytest.raises(ValueError, match="env_id"):
        world.set_new_context(_ctx(_mech(env_id=None)))


@pytest.mark.unit
def test_update_context_replaces_payload(world):
    ctx = _ctx(_mech(env_id="env-0"))
    cid = world.set_new_context(ctx)

    new_ctx = Context(
        id=cid, opt_id="opt", step=1, env="e", payload=_mech(index=9, env_id="env-1")
    )
    world.update_context(new_ctx)
    assert world.get_context(cid) is new_ctx
    assert world.get_mechanism_registry()[cid].index == 9

    plain = Context(id=cid, opt_id="opt", step=2, env="e", payload=_OtherSchema())
    world.update_context(plain)
    assert world.get_context(cid) is plain


@pytest.mark.unit
def test_update_context_errors(world):
    with pytest.raises(KeyError, match="not registered"):
        world.update_context(_ctx(_OtherSchema()))

    cid = world.set_new_context(_ctx(_mech(env_id="env-0")))
    with pytest.raises(ValueError, match="env_id"):
        world.update_context(
            Context(id=cid, opt_id="opt", step=0, env="e", payload=_mech(env_id=None))
        )


@pytest.mark.unit
def test_remove_context_cleans_both_registries(world):
    a = _ctx(_OtherSchema())
    b = _ctx(_OtherSchema())
    world.append_context(a)
    world.append_context(b)

    world.remove_context(a)
    assert a.id not in world.get_ctx_ids()
    assert world.get_opt_ctx_ids("opt") == [b.id]

    world.remove_context(b)
    assert world.get_opt_ids() == set()

    # Unknown context / optimizer: no error.
    world.remove_context(
        Context(id="x", opt_id="nobody", step=0, env="e", payload=b.payload)
    )


@pytest.mark.unit
def test_remove_context_id_missing_from_opt_list(world):
    a = _ctx(_OtherSchema())
    world.append_context(a)
    stranger = Context(
        id="ghost", opt_id="opt", step=0, env="e", payload=_OtherSchema()
    )
    world.remove_context(stranger)
    # The optimizer still owns ``a``.
    assert world.get_opt_ctx_ids("opt") == [a.id]


# ---------------------------------------------------------------------------
# Env-step accessors
# ---------------------------------------------------------------------------


@pytest.fixture
def populated(world):
    """Two optimizers, two episodes for ``opt_a``, one mechanism in between."""
    ids = {}
    ids["a0"] = world.append_context(_ctx(_step(0), opt_id="a", step=0))
    ids["a1"] = world.append_context(_ctx(_step(1), opt_id="a", step=1))
    ids["mech"] = world.append_context(_ctx(_mech(), opt_id="a"))
    ids["a0b"] = world.append_context(_ctx(_step(0), opt_id="a", step=0))
    ids["a1b"] = world.append_context(_ctx(_step(1), opt_id="a", step=1))
    ids["b0"] = world.append_context(_ctx(_step(0), opt_id="b", step=0))
    return world, ids


@pytest.mark.unit
def test_get_env_step_contexts_filters_by_optimizer(populated):
    world, ids = populated
    all_steps = world.get_env_step_contexts()
    assert [c.id for c in all_steps] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
        ids["b0"],
    ]
    assert [c.id for c in world.get_env_step_contexts("a")] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
    ]
    assert world.get_env_step_contexts("unknown") == []


@pytest.mark.unit
def test_get_env_step_contexts_skips_flushed_ids(populated):
    world, ids = populated
    world.flush_ctx([ids["a1b"]])
    assert [c.id for c in world.get_env_step_contexts("a")] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
    ]


@pytest.mark.unit
def test_get_latest_env_step_contexts(populated):
    world, ids = populated
    assert [c.id for c in world.get_latest_env_step_contexts("a")] == [
        ids["a0b"],
        ids["a1b"],
    ]
    # Without an optimizer, the latest episode belongs to ``b``.
    assert [c.id for c in world.get_latest_env_step_contexts()] == [ids["b0"]]
    assert world.get_latest_env_step_contexts("none") == []


@pytest.mark.unit
def test_get_latest_env_step_contexts_without_reset_returns_all(world):
    world.append_context(_ctx(_step(1), step=1))
    world.append_context(_ctx(_step(2), step=2))
    assert len(world.get_latest_env_step_contexts("opt")) == 2


@pytest.mark.unit
def test_get_new_env_step_contexts_advances_cursor(populated):
    world, ids = populated
    first = world.get_new_env_step_contexts("a")
    assert [c.id for c in first] == [ids["a0"], ids["a1"], ids["a0b"], ids["a1b"]]
    assert world.get_new_env_step_contexts("a") == []

    new_id = world.append_context(_ctx(_step(0), opt_id="a", step=0))
    assert [c.id for c in world.get_new_env_step_contexts("a")] == [new_id]

    # Global cursor (opt_id=None) is independent of the per-optimizer one.
    assert len(world.get_new_env_step_contexts()) == 6
    assert world.get_new_env_step_contexts() == []


@pytest.mark.unit
def test_get_new_env_step_contexts_ignores_missing_contexts(populated):
    world, ids = populated
    world.flush_ctx([ids["a0"]])
    assert [c.id for c in world.get_new_env_step_contexts("a")] == [
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
    ]


# ---------------------------------------------------------------------------
# Mechanism accessors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_mechanism_claims_first_published(world):
    world.append_context(_ctx(_mech(index=0, status=MechanismStatus.train)))
    world.append_context(_ctx(_mech(index=1)))
    world.append_context(_ctx(_mech(index=2)))

    m = world.get_mechanism()
    assert m.index == 1 and m.status == MechanismStatus.assigned
    m = world.get_mechanism()
    assert m.index == 2 and m.status == MechanismStatus.assigned
    with pytest.raises(RuntimeError, match="no available mechanisms"):
        world.get_mechanism()


@pytest.mark.unit
def test_try_get_mechanism_returns_none_when_exhausted(world):
    assert world.try_get_mechanism() is None
    world.append_context(_ctx(_mech(index=4)))
    m = world.try_get_mechanism()
    assert m.index == 4 and m.status == MechanismStatus.assigned
    assert world.try_get_mechanism() is None


@pytest.mark.unit
def test_get_mechanism_by_id_lifecycle(world):
    world.append_context(_ctx(_mech(index=1, seed=10)))
    world.append_context(_ctx(_mech(index=1, seed=20)))

    # Wrong seed / index: nothing matches.
    assert world.get_mechanism_by_id(1, 99, MechanismStatus.train) is None
    assert world.get_mechanism_by_id(5, 10, MechanismStatus.train) is None

    # Eval requires train/eval predecessors: a published entry is not eligible.
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is None

    m = world.get_mechanism_by_id(1, 10, MechanismStatus.train)
    assert m.seed == 10 and m.status == MechanismStatus.train
    # Handed out once only.
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.train) is None

    # train -> eval, then eval -> eval stays fetchable.
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is m
    assert m.status == MechanismStatus.eval
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is m

    # The other seed is untouched.
    assert world.get_mechanism_registry()[list(world.get_ctx_ids())[1]].status in {
        MechanismStatus.published,
        MechanismStatus.eval,
    }


@pytest.mark.unit
def test_get_mechanism_by_id_invalid_mode_raises_type_error(world):
    world.append_context(_ctx(_mech(index=0, seed=0)))
    with pytest.raises(TypeError):
        world.get_mechanism_by_id(0, 0, MechanismStatus.done)


@pytest.mark.unit
def test_get_mechanism_by_index_requires_registry_key(world):
    cid = world.append_context(_ctx(_mech(index=0)))
    assert world.get_mechanism_by_index(cid).index == 0
    with pytest.raises(KeyError):
        world.get_mechanism_by_index(0)


# ---------------------------------------------------------------------------
# flush / flush_ctx
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flush_by_status_and_all(world):
    c_pub = world.append_context(_ctx(_mech(index=0)))
    c_eval = world.append_context(_ctx(_mech(index=1, status=MechanismStatus.eval)))
    c_train = world.append_context(_ctx(_mech(index=2, status=MechanismStatus.train)))

    world.flush(status=MechanismStatus.eval)
    assert set(world.get_mechanism_registry()) == {c_pub, c_train}
    # Contexts are untouched by ``flush``.
    assert world.get_ctx_ids() == {c_pub, c_eval, c_train}

    world.flush()
    assert world.get_mechanism_registry() == {}
    assert world.get_ctx_ids() == {c_pub, c_eval, c_train}


@pytest.mark.unit
def test_flush_ctx_removes_contexts_only(world):
    cid = world.append_context(_ctx(_mech(index=0)))
    other = world.append_context(_ctx(_OtherSchema()))

    world.flush_ctx([])
    assert world.get_ctx_ids() == {cid, other}

    world.flush_ctx([cid, "unknown-id"])
    assert world.get_ctx_ids() == {other}
    # Mechanism registry and optimizer map still reference the flushed ID.
    assert cid in world.get_mechanism_registry()
    assert cid in world.get_opt_ctx_ids("opt")
