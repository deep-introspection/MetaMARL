from examples.bilevel_fishery.mechanism import (
    FisheryMechanism,
    FisheryMechanismSpace,
)
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpace": FisheryMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanism": FisheryMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
    },
}
