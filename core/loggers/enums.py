from enum import Enum


class ReduceProtocol(str, Enum):
    """Reduction strategy applied when aggregating a metric across steps or workers.

    Used as metadata on :class:`~loggers.schemas.LoggerSchema` fields to
    indicate how multiple values for the same metric should be combined before
    being written to a logger.

    Values
    ------
    MEAN : str
        Arithmetic mean of all collected values.
    SUM : str
        Sum of all collected values.
    MAX : str
        Maximum of all collected values.
    MIN : str
        Minimum of all collected values.
    LAST : str
        Keep only the most recent value; discard earlier ones.
    """

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"