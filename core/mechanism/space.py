from abc import abstractmethod
from typing import Protocol

import numpy as np

from core.mechanism.base import Mechanism


# TODO why not have these methods be abstract ?
class MechanismSpace(Protocol):
    """Geometry + constraints over a mechanism manifold."""

    dimension: int

    def _validate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in vector: {x}")

        return x

    @abstractmethod
    def encode(self, m: Mechanism) -> np.ndarray: ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> Mechanism: ...

    def clip(self, m: Mechanism) -> Mechanism: ...

    def sample(self) -> Mechanism: ...
