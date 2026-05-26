"""Ecological state representation.

The state is intentionally immutable (``frozen=True``) so that ``step`` is a
pure function: same inputs always produce the same outputs, no aliasing
across simulation paths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EcologicalState:
    """Immutable snapshot of the predator-prey ecological system.

    Parameters
    ----------
    fish
        Fish biomass (predator).
    algae
        Algae biomass (prey).
    """

    fish: float
    algae: float

    def as_tuple(self) -> tuple[float, float]:
        """Return the state as a ``(fish, algae)`` tuple."""
        return (self.fish, self.algae)
