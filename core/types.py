from typing import Any, Hashable, TypeAlias, Union

from gymnasium import Env
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# TODO what if we want the contextID to be a unique UUID and we keep a registry of already existing contextID in the world
# TODO registry object for the world.
ContextID: TypeAlias = str
"""
Unique identifier for a context object.

Used to distinguish different context schemas stored in a World
(e.g., 'quota_violation', 'market_price', 'stock_level').

ContextIDs are semantic, not structural.
"""

# TODO again what if we want a way to register the Optimizer in a memory object and generate a unique uuid for it ?
OptimizerID: TypeAlias = str
"""
Unique identifier for an optimizer instance.

Used to:
- Namespace contexts inside a World
- Associate contexts with producing optimizers
- Support parent/child or upstream/downstream optimizer graphs

OptimizerIDs are expected to be stable for the lifetime of an experiment.
"""


# Represents a gymnasium Env, a MultiAgentEnv, WorldEnv.
# """

EnvType: TypeAlias = Union[BaseEnv, MultiAgentEnv, Env]
"""
Represents a BaseEnv, MultiAgentEnv, ExternalEnv, ExternalMultiAgentEnv,
VectorEnv, gym.Env, or ActorHandle.
"""


EnvConfigDict: TypeAlias = dict
"""
Represents the env_config sub-dict of the algo config that is passed to
the env constructor.
"""

AgentID = Hashable
"""Represents a generic identifier for an agent (e.g., "agent1")."""

MultiAgentDict = dict[AgentID, Any]
"""A dict keyed by agent ids, e.g. {"agent-1": value}."""
