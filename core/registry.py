from examples.bilevel_fishery.mechanism import (
    FisheryMechanism,
    FisheryMechanismSpace,
)
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# Water-usage example components
from examples.water_usage.mechanism import (
    WaterMechanism,
    WaterMechanismSpace,
)
from examples.water_usage.regulated_env import WaterRegulatedEnv

from examples.water_usage.regulated_env_ed_hs import WaterRegulatedEdHsEnv
from examples.water_usage.regulator_env import WaterRegulatorEnv
from examples.water_usage.regulator_env_raven import WaterRegulatorRavenEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpace": FisheryMechanismSpace,
        "WaterMechanismSpace": WaterMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanism": FisheryMechanism,
        "WaterMechanism": WaterMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
    "WaterRegulatorEnv": WaterRegulatorEnv,
    "WaterRegulatedEnv": WaterRegulatedEnv,
    # Generic bilevel aliases (map to water example by default)
    "RegulatorEnv": WaterRegulatorEnv,
    "WaterRegulatorRavenEnv": WaterRegulatorRavenEnv,
    "RegulatedEnv": WaterRegulatedEnv,
    "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv
    },
}
