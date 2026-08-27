import re
import uuid
from collections.abc import Mapping
from typing import AbstractSet, Any, Optional

import numpy as np

# Generic utils

# TODO restrict Any type annotation.
EPS = 1e-8


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


def sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def smooth_positive(x: float, width: float) -> float:
    """Smooth approximation of max(x, 0)."""
    width = max(float(width), EPS)

    return float(
        width
        * np.logaddexp(
            0.0,
            float(x) / width,
        )
    )


def smooth_min(
    a: float,
    b: float,
    width: float,
) -> float:
    """Smooth transition between a and b approximating min(a, b)."""
    a = float(a)
    b = float(b)
    width = max(float(width), EPS)

    weight_on_b = sigmoid((a - b) / width)

    return float((1.0 - weight_on_b) * a + weight_on_b * b)


def smooth_cap_01(x: float) -> float:
    """Smooth saturation of a non-negative value into [0, 1)."""
    x = float(x)
    return float(1.0 - np.exp(-x))


def smooth_positive_zero_at_origin(
    x: float,
    width: float,
) -> float:
    """
    Smooth approximation of max(x, 0) with value 0 at x=0.
    """
    width = max(float(width), EPS)

    value = width * (np.logaddexp(0.0, float(x) / width) - np.log(2.0))

    return float(max(0.0, value))
