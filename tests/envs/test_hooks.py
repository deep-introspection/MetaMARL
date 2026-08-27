"""Tests for hook declaration and discovery (TODO §8)."""

import pytest

from core.envs import hooks
from core.envs.marl_regulated import MultiAgentRegulatedEnv


@pytest.mark.unit
@pytest.mark.parametrize(
    "hook", ["reset", "action", "reward", "observation", "transition"]
)
def test_decorator_marks_and_subclass_registers(hook):
    decorator = getattr(hooks, hook)

    class Env(MultiAgentRegulatedEnv):
        @decorator
        def my_hook(self, *args, **kwargs):
            return None

    assert getattr(Env.my_hook, hook) is True
    assert getattr(Env, f"_{hook}") == "my_hook"
    # other hooks stay unset
    for other in set(["reset", "action", "reward", "observation", "transition"]) - {
        hook
    }:
        assert getattr(Env, f"_{other}") is None


@pytest.mark.unit
def test_inherited_hooks_are_kept_and_can_be_redeclared():
    class Parent(MultiAgentRegulatedEnv):
        @hooks.reset
        def parent_reset(self):
            return {}

    class Child(Parent):
        @hooks.transition
        def child_transition(self, *, A_t, S_t):
            return S_t

    class Override(Parent):
        @hooks.reset
        def child_reset(self):
            return {}

    assert Child._reset == "parent_reset"
    assert Child._transition == "child_transition"
    assert Override._reset == "child_reset"
    assert Parent._transition is None


@pytest.mark.unit
def test_two_hooks_of_same_type_fail_fast():
    with pytest.raises(TypeError, match="several @reset hooks"):

        class Env(MultiAgentRegulatedEnv):
            @hooks.reset
            def first(self):
                return {}

            @hooks.reset
            def second(self):
                return {}
