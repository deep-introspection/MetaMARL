from abc import ABCMeta
from typing import Optional

from torch import Type


class StatsBase(metaclass=ABCMeta):
    # TODO
    pass


class ResultReporter:
    """A generic class collecting and reducing results"""

    # TODO how to store data in the results reporter ?

    def __init__(self, stats_cls_lookup: Optional[dict[str, Type[StatsBase]]]):
        # TODO
        pass
