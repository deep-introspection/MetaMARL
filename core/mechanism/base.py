from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Mechanism(Protocol):
    """Semantic representation of a regulatory mechanism."""

    def to_vector(self) -> list[float]:
        """Convert semantic mechanism to normalized vector in [0,1]^d."""
        ...


@dataclass(frozen=True)
class VectorMechanism(Mechanism):
    x: np.ndarray

    def to_vector(self) -> list[float]:
        return np.asarray(self.x, dtype=np.float32).ravel().tolist()

    @classmethod
    def from_vector(cls, v: list[float]) -> "VectorMechanism":
        return cls(np.asarray(v, dtype=np.float32))
