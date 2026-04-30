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
