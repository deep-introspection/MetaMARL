"""Runtime registry mapping string names (from YAML config) to Python classes.

The ``REGISTRY`` dict is structured as::

    {
        "mechanism_space": { "<ClassName>": <class>, ... },
        "mechanism":       { "<ClassName>": <class>, ... },
        "env":             { "<ClassName>": <class>, ... },
    }

Callers look up a class by category and name at runtime, enabling experiment
configs to reference environment or mechanism implementations by string without
hard-coding imports throughout the codebase.

Currently registered components
---------------------------------
mechanism_space
    ``FisheryMechanismSpace``     — v0 (default) fishery mechanism search space.
    ``FisheryMechanismSpaceV1``   — v1 fishery mechanism search space.
mechanism
    ``FisheryMechanism``          — v0 fishery regulatory mechanism.
    ``FisheryMechanismV1``        — v1 fishery regulatory mechanism.
env
    ``FisheryRegulatorEnv``       — outer-loop regulator environment (fishery).
    ``FisheryRegulatedEnv``       — inner-loop regulated fishing environment (v0).
    ``FisheryRegulatedEnvV1``     — inner-loop regulated fishing environment (v1).
"""

from examples.bilevel_fishery.mechanism import FisheryMechanismSpace
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanismSpace as FisheryMechanismSpaceV1
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulated_env_v1 import FisheryRegulatedEnv as FisheryRegulatedEnvV1
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.bilevel_fishery.mechanism import FisheryMechanism as FisheryMechanism
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanism as FisheryMechanismV1

# Water-usage example components
# from examples.water_usage.mechanism import (
#     WaterMechanism,
#     WaterMechanismSpace,
# )
# from examples.water_usage.regulated_env import WaterRegulatedEnv
# from examples.water_usage.regulator_env import WaterRegulatorEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpace": FisheryMechanismSpace,
        "FisheryMechanismSpaceV1": FisheryMechanismSpaceV1
        # "WaterMechanismSpace": WaterMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanism": FisheryMechanism,
        "FisheryMechanismV1": FisheryMechanismV1,
        # "WaterMechanism": WaterMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
        "FisheryRegulatedEnvV1": FisheryRegulatedEnvV1,
        # "WaterRegulatorEnv": WaterRegulatorEnv,
        # "WaterRegulatedEnv": WaterRegulatedEnv,
        # Generic bilevel aliases (map to water example by default)
        # "RegulatorEnv": WaterRegulatorEnv,
        # "RegulatedEnv": WaterRegulatedEnv,
    },
}
