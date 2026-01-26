from typing import Protocol


class Mechanism(Protocol):
    """
    Typed interface for regulator parameters.
    """

    def to_vector(self) -> list[float]: ...

    @classmethod
    def from_vector(cls, x: list[float]) -> "Mechanism": ...
