"""Decorators declaring benchmark hooks on a ``MultiAgentRegulatedEnv`` subclass.

Each decorator tags a method so that ``MultiAgentRegulatedEnv.__init_subclass__``
can discover it. A benchmark declares at most one method per hook:

- ``@reset``: ``() -> S_0``, initial benchmark state;
- ``@action``: ``(action_dict) -> action_dict``, benchmark-side action adjustment
  applied after normalization and before the mechanism;
- ``@reward``: ``(action_dict) -> reward_dict``, intrinsic reward on the current
  state and delivered actions, before the mechanism's reward transform;
- ``@transition``: ``(*, A_t, S_t) -> S_{t+1}``, the dynamics;
- ``@observation``: ``(observation_dict) -> observation_dict``, ``o_i = O_i(S_t)``.

Example
-------
>>> class MyEnv(MultiAgentRegulatedEnv):
...     @reset
...     def init_state(self):
...         return {"stock": 1.0}
...     @transition
...     def dynamics(self, *, A_t, S_t):
...         return {"stock": S_t["stock"] - sum(a[0] for a in A_t.values())}
"""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _hook(func: F, name: str) -> F:
    setattr(func, name, True)
    return func


def action(func: F) -> F:
    """Mark the benchmark action hook."""
    return _hook(func, "action")


def observation(func: F) -> F:
    """Mark the benchmark observation hook."""
    return _hook(func, "observation")


def reset(func: F) -> F:
    """Mark the benchmark reset hook (returns the initial state)."""
    return _hook(func, "reset")


def reward(func: F) -> F:
    """Mark the benchmark intrinsic-reward hook."""
    return _hook(func, "reward")


def transition(func: F) -> F:
    """Mark the benchmark transition hook (returns the next state)."""
    return _hook(func, "transition")
