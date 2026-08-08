from enum import Enum


class ReduceProtocol(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"
    EMA = "ema"
    SERIES = "series"
