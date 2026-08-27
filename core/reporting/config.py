import copy
from abc import ABC, abstractmethod
from typing import Self, Union

from core.reporting.base import Reporter


class ReporterConfig(ABC):
    def __init__(
        self,
        project: str,
    ):
        self.project_name: str = project
        self._world_name: Union[str | None] = None
        self._outer_iters: Union[int | None] = None

    @property
    def world(self) -> Union[str, None]:
        return self._world_name

    @world.setter
    def world(self, world: str) -> None:
        self._world_name = world

    @property
    def outer_iters(self) -> Union[int, None]:
        return self._outer_iters

    @outer_iters.setter
    def outer_iters(self, outer_iters: int) -> None:
        self._outer_iters = outer_iters

    def copy(self) -> Self:
        """Return a deep copy of this reporter configuration."""
        return copy.deepcopy(self)

    @abstractmethod
    def build(self) -> Reporter:
        """
        Compiles config based on the reporter type and initiates the reporter
        """
        ...
