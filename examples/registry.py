"""String-to-class registry used by the YAML experiment loaders.

Maps the names used in ``config.yaml`` files to environment classes. It lives
under ``examples`` because it refers to example code; the library (``core``)
does not depend on it. The pre-mechanism ``mechanism_space`` entries were
removed together with that abstraction. The fresh-water example still uses
the pre-mechanism environment API and is not registered on this branch.
"""

from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

REGISTRY = {
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
    },
}
