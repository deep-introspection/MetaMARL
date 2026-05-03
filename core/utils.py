from collections.abc import Mapping
import re
import uuid
from typing import AbstractSet, Any, Optional

import numpy as np

# Generic utils

# TODO restrict Any type annotation.


def generate_uuid(registry: AbstractSet[Any]) -> str:
    """Generate a UUID that is guaranteed not to exist in the given registry.

    Retries until a collision-free UUID is found.  In practice the probability
    of even a single retry is astronomically small for any realistic registry
    size.

    Parameters
    ----------
    registry : AbstractSet[Any]
        Set of already-used identifiers.  The generated UUID is guaranteed not
        to be a member of this set at the moment of return.

    Returns
    -------
    str
        A new UUID4 string not present in ``registry``.
    """
    while True:
        id = str(uuid.uuid4())
        if id not in registry:
            return id


def safe_ratio(num: Any, den: Any) -> Optional[float]:
    """Safely compute ``num / den``, returning ``None`` on invalid inputs.

    Returns ``None`` when either operand is non-finite or when ``den`` is
    zero, avoiding ``ZeroDivisionError`` and propagation of ``NaN``/``Inf``.

    Parameters
    ----------
    num : Any
        Numerator.  Must be convertible to a finite float.
    den : Any
        Denominator.  Must be convertible to a non-zero finite float.

    Returns
    -------
    float or None
        ``float(num) / float(den)``, or ``None`` if either value is
        non-finite or ``den`` is zero.
    """
    fnum = finite(num)
    fden = finite(den)
    if fnum is None or fden is None or fden == 0.0:
        return None
    return float(fnum) / float(fden)


def to_float(x: Any) -> Optional[float]:
    """Safely cast a value to a Python ``float``.

    Unwraps NumPy scalars via ``.item()`` before casting to avoid
    scalar-wrapping edge cases.

    Parameters
    ----------
    x : Any
        Value to convert.

    Returns
    -------
    float or None
        Converted value, or ``None`` if ``x`` is ``None`` or cannot be cast
        to float.
    """
    if x is None:
        return None
    try:
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def finite(x: Any) -> Optional[float]:
    """Cast to float and return ``None`` for non-finite values.

    Combines :func:`to_float` with a finiteness check, so ``NaN`` and
    ``±Inf`` are treated the same as unconvertible values.

    Parameters
    ----------
    x : Any
        Value to convert and validate.

    Returns
    -------
    float or None
        A finite float, or ``None`` if ``x`` cannot be converted or is
        ``NaN``/``±Inf``.
    """
    fx = to_float(x)
    if fx is None or not np.isfinite(fx):
        return None
    return fx


def sanitize_key(s: str) -> str:
    """Sanitise a string for use as a W&B metric key.

    Replaces any character that is not alphanumeric, an underscore, or a dash
    with an underscore.  Ensures metric keys are safe across logging backends.

    Parameters
    ----------
    s : str
        Raw key string (e.g. an observation name that may contain spaces or
        special characters).

    Returns
    -------
    str
        Sanitised key containing only ``[0-9a-zA-Z_-]``.
    """
    # keep alnum, underscore, dash; replace everything else with underscore
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(s))


def is_mapping(x: Any) -> bool:
    """Return ``True`` if ``x`` is an instance of ``collections.abc.Mapping``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` for dicts and other mapping types; ``False`` otherwise.
    """
    return isinstance(x, Mapping)


def flatten_numeric(value: Any) -> list[float]:
    """Flatten any numeric value or array into a flat list of Python floats.

    Scalars (0-D arrays or plain numbers) are returned as a single-element
    list.  Multi-dimensional arrays are flattened row-major.

    Parameters
    ----------
    value : Any
        Scalar, list, or NumPy array of numeric values.

    Returns
    -------
    list[float]
        Flat list of Python floats.
    """
    import numpy as np

    arr = np.asarray(value)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr.reshape(-1)]
