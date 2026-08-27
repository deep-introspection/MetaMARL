"""String-to-class registry used by the YAML experiment loaders.

Maps the names used in ``config.yaml`` files to environment and mechanism
classes. Only components that import cleanly on this branch are registered.
"""

from examples.bilevel_fishery.deprecated.mechanism import (
    FisheryMechanism,
    FisheryMechanismSpace,
)
from examples.bilevel_fishery.mechanism_v1 import (
    FisheryMechanism as FisheryMechanismV1,
)
from examples.bilevel_fishery.mechanism_v1 import (
    FisheryMechanismSpace as FisheryMechanismSpaceV1,
)
from examples.bilevel_fishery.regulated_env_shaefer import (
    FisheryRegulatedEnv as FisheryRegulatedEnvSchaefer,
)
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.fresh_water.deprecated.regulator_env import WaterRegulatorEnv
from examples.fresh_water.mechanism import WaterMechanism, WaterMechanismSpace
from examples.fresh_water.regulated_env_ed_hs import WaterRegulatedEdHsEnv
from examples.fresh_water.regulator_env_raven import WaterRegulatorRavenEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpace": FisheryMechanismSpace,
        "FisheryMechanismSpaceV1": FisheryMechanismSpaceV1,
        "WaterMechanismSpace": WaterMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanism": FisheryMechanism,
        "FisheryMechanismV1": FisheryMechanismV1,
        "WaterMechanism": WaterMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnvSchaefer,
        "FisheryRegulatedEnvSchaefer": FisheryRegulatedEnvSchaefer,
        "WaterRegulatorEnv": WaterRegulatorEnv,
        "WaterRegulatorRavenEnv": WaterRegulatorRavenEnv,
        "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv,
    },
}
