from abc import ABCMeta
from typing import Optional

from torch import Type


class StatsBase(metaclass=ABCMeta):
    """Abstract base class for statistics containers.

    Subclasses define the schema of statistics that a reporter can collect and
    reduce across steps or episodes.  Concrete implementations are registered
    in a ``stats_cls_lookup`` dict passed to :class:`ResultReporter`.
    """


class ResultReporter:
    """Generic reporter that collects and reduces experiment results.

    Intended to be subclassed for specific backends (e.g. wandb, local CSV).
    Holds a lookup mapping string keys to :class:`StatsBase` subclasses so
    that heterogeneous result types can be reduced and forwarded correctly.

    Parameters
    ----------
    stats_cls_lookup : dict[str, Type[StatsBase]] or None
        Mapping from result-category names (e.g. ``"episode"``, ``"learner"``)
        to their corresponding :class:`StatsBase` subclass.  Pass ``None`` to
        defer registration.
    """

    def __init__(
            self,
            stats_cls_lookup: Optional[
                dict[str, Type[StatsBase]]
            ]
    ):
        # TODO
        pass
