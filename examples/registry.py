"""String-to-class registry used by the YAML experiment loaders.

Maps the names used in ``config.yaml`` files to environment classes. It lives
under ``examples`` because it refers to example code; the library (``core``)
does not depend on it. The pre-mechanism ``mechanism_space`` entries were
removed together with that abstraction.
"""

from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.fresh_water.regulated_env_ed_hs import WaterRegulatedEdHsEnv

REGISTRY = {
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
        "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv,
    },
}
