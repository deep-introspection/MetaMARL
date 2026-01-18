from typing import Any, Union

import gym

# TODO create the WorldEnv
# TODO in ray there are different types of envs : BaseEnv, ExternalEnv, ExternalMultiAgentEnv
# TODO i really dont like any because it is not restricting enough. but I want ability to accomodate other environments in the future
# TODO WorldEnv should be also a gym.Env with the added feature to have sub envs

WorldType = Union[Any, gym.Env]
"""
Represents a gym.Env, a MultiAgentEnv, WorldEnv.
"""


ContextID = str
