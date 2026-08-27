from dataclasses import dataclass

import numpy as np
from core.mechanism.space import MechanismSpace
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism


@dataclass(frozen=True)
class DummyMechanism(Mechanism):
    value: float = 0.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array([], dtype=np.float32)


class DummyMechanismSpace(MechanismSpace):
    def __init__(self):
        super().__init__()
        self.dimension = 1
        self.full_dimension = 0

    def default(self) -> DummyMechanism:
        return DummyMechanism(0.0)

    def encode(self, m: DummyMechanism) -> NDArray[np.float32]:
        return np.array([m.value], dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        x = self._validate(x)
        return DummyMechanism(float(x[0]))

    def clip(self, m: DummyMechanism) -> DummyMechanism:
        return DummyMechanism(float(m.value))

    def from_dict(self, cfg: dict) -> DummyMechanism:
        return DummyMechanism(float(cfg.get("value", 0.0)))
