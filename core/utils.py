from collections.abc import Mapping
import re
import uuid
from typing import AbstractSet, Any, Optional

import numpy as np

# Generic utils

# TODO restrict Any type annotation.


def generate_uuid(registry: AbstractSet[Any]) -> str:
    """
    Generate a UUID that is guaranteed not to exist in the given registry.
    """
    while True:
        id = str(uuid.uuid4())
        if id not in registry:
            return id


def safe_ratio(num: Any, den: Any) -> Optional[float]:
    fnum = finite(num)
    fden = finite(den)
    if fnum is None or fden is None or fden == 0.0:
        return None
    return float(fnum) / float(fden)


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def finite(x: Any) -> Optional[float]:
    fx = to_float(x)
    if fx is None or not np.isfinite(fx):
        return None
    return fx


def sanitize_key(s: str) -> str:
    # keep alnum, underscore, dash; replace everything else with underscore
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(s))


def is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def flatten_numeric(value: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(value)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr.reshape(-1)]
