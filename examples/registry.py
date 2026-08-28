"""String-to-class registry used by the YAML experiment loaders.

Maps the names used in ``config.yaml`` files to environment and mechanism
classes. It lives under ``examples`` because it refers to example code; the
library (``core``) does not depend on it.
"""

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
from examples.fresh_water.mechanism import WaterMechanism, WaterMechanismSpace
from examples.fresh_water.regulated_env_ed_hs import WaterRegulatedEdHsEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpaceV1": FisheryMechanismSpaceV1,
        "WaterMechanismSpace": WaterMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanismV1": FisheryMechanismV1,
        "WaterMechanism": WaterMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnvSchaefer,
        "FisheryRegulatedEnvSchaefer": FisheryRegulatedEnvSchaefer,
        "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv,
    },
}
