"""String-to-class registry used by the YAML experiment loaders.

Maps the names used in ``config.yaml`` files to environment classes. Only
components that exist on this branch are registered; the pre-mechanism
``mechanism_space`` entries were removed together with that abstraction.
"""

from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.fresh_water.regulated_env_ed_hs import WaterRegulatedEdHsEnv
from examples.fresh_water.regulator_env_raven import WaterRegulatorRavenEnv

REGISTRY = {
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
        "WaterRegulatorRavenEnv": WaterRegulatorRavenEnv,
        "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv,
    },
}
