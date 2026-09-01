"""Enumerations shared by the metric loggers."""

from enum import Enum


class ReduceProtocol(str, Enum):
    """How a windowed series of values is collapsed into one number.

    The members are ``str`` subclasses so a protocol can be written as its
    plain name (``"mean"``, ``"sum"``, ``"max"``, ``"min"``, ``"last"``) in a
    configuration and compared directly to the enum.
    """

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"
