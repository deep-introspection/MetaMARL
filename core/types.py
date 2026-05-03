from typing import TypeAlias, Union

from gymnasium import Env
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# TODO what if we want the contextID to be a unique UUID and we keep a registry of already existing contextID in the world
# TODO registry object for the world.
ContextID: TypeAlias = str
"""Unique identifier for a context object in the World.

Used to distinguish different context schemas stored in a
:class:`~core.world.world.World` (e.g. ``"quota_violation"``,
``"market_price"``, ``"stock_level"``).  ContextIDs are semantic — they
identify *what* the context represents, not its storage address.
"""

# TODO again what if we want a way to register the Optimizer in a memory object and generate a unique uuid for it ?
OptimizerID: TypeAlias = str
"""Unique identifier for an optimizer (outer-loop) instance.

Used to:

* Namespace contexts inside a :class:`~core.world.world.World`.
* Associate emitted contexts with the producing optimizer.
* Support parent/child or upstream/downstream optimizer graphs.

OptimizerIDs are expected to remain stable for the lifetime of an
experiment run.
"""


# TODO create the WorldEnv
# TODO in ray there are different types of envs : BaseEnv, ExternalEnv, ExternalMultiAgentEnv
# TODO i really dont like any because it is not restricting enough. but I want ability to accomodate other environments in the future
# TODO WorldEnv should be also a gymnasium Env with the added feature to have sub envs

EnvType: TypeAlias = Union[BaseEnv, MultiAgentEnv, Env]
"""Union of accepted RLlib/Gymnasium environment types.

Covers ``BaseEnv``, ``MultiAgentEnv``, ``ExternalEnv``,
``ExternalMultiAgentEnv``, ``VectorEnv``, ``gym.Env``, and Ray
``ActorHandle`` instances wrapping any of the above.
"""


EnvConfigDict: TypeAlias = dict
"""The ``env_config`` sub-dict of an RLlib algorithm config.

Passed verbatim to the environment constructor.  Typically contains
scenario-specific hyperparameters such as population dynamics coefficients,
quota bounds, and agent counts.
"""
