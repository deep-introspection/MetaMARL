from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Mechanism(Protocol):
    """
    Typed interface for regulator parameters.
    """

    def to_vector(self) -> list[float]: ...

    @classmethod
    def from_vector(cls, x: list[float]) -> "Mechanism": ...


@dataclass(frozen=True)
class VectorMechanism(Mechanism):
    x: np.ndarray

    def to_vector(self) -> list[float]:
        return np.asarray(self.x, dtype=np.float32).ravel().tolist()

    @classmethod
    def from_vector(cls, v: list[float]) -> "VectorMechanism":
        return cls(np.asarray(v, dtype=np.float32))
