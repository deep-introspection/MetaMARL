from asyncio import Protocol

import numpy as np

from core.mechanism.base import Mechanism


class MechanismSpace(Protocol):
    dimension: int

    def sample(self) -> Mechanism: ...

    def project(self, x: np.ndarray) -> Mechanism: ...

    def clip(self, x: Mechanism) -> Mechanism: ...

    def from_vector(self, x: np.ndarray) -> Mechanism: ...

    def batch_size(self, action) -> int: ...

    def broadcast(self, scalar: float, batch_size: int) -> np.ndarray: ...
