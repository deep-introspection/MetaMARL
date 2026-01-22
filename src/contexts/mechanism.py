"""Regulatory mechanism parameters and mapping functions.

This module defines the mechanism parameter structure and provides functions
to map between the unit hypercube optimization space and actual parameter values.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MechanismParameters:
    """Regulatory mechanism parameters for the fishery.

    Attributes:
        fixed_quota: Absolute harvest limit (0 to 1)
        prop_quota: Proportional quota factor (0 to 1)
        min_stock: Minimum stock threshold for fishing (0 to 1)
        fine_amount: Penalty per unit over-harvest (0 to 2)
        ban_period: Duration of ban after violation (0 to 10 periods)
    """

    fixed_quota: float
    prop_quota: float
    min_stock: float
    fine_amount: float
    ban_period: int

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert 0.0 <= self.fixed_quota <= 1.0, (
            f"fixed_quota must be in [0,1], got {self.fixed_quota}"
        )
        assert 0.0 <= self.prop_quota <= 1.0, (
            f"prop_quota must be in [0,1], got {self.prop_quota}"
        )
        assert 0.0 <= self.min_stock <= 1.0, (
            f"min_stock must be in [0,1], got {self.min_stock}"
        )
        assert 0.0 <= self.fine_amount <= 2.0, (
            f"fine_amount must be in [0,2], got {self.fine_amount}"
        )
        assert 0 <= self.ban_period <= 10, (
            f"ban_period must be in [0,10], got {self.ban_period}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "fixed_quota": self.fixed_quota,
            "prop_quota": self.prop_quota,
            "min_stock": self.min_stock,
            "fine_amount": self.fine_amount,
            "ban_period": self.ban_period,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MechanismParameters("
            f"fixed_quota={self.fixed_quota:.3f}, "
            f"prop_quota={self.prop_quota:.3f}, "
            f"min_stock={self.min_stock:.3f}, "
            f"fine_amount={self.fine_amount:.3f}, "
            f"ban_period={self.ban_period})"
        )


def map_unit_vector_to_mechanism(
    unit_vector: np.ndarray, use_stochastic_rounding: bool = True
) -> MechanismParameters:
    """Map unit vector from [0,1]^5 to mechanism parameters.

    This function provides the mapping from the ES optimization space
    (unit hypercube) to the actual mechanism parameter space.

    Args:
        unit_vector: 5D vector with components in [0,1]
        use_stochastic_rounding: If True, use stochastic rounding for ban_period

    Returns:
        MechanismParameters object with mapped values

    Parameter Mappings:
        - fixed_quota: [0,1] → [0,1] (identity)
        - prop_quota: [0,1] → [0,1] (identity)
        - min_stock: [0,1] → [0,1] (identity)
        - fine_amount: [0,1] → [0,2] (linear scaling)
        - ban_period: [0,1] → {0,1,2,...,10} (stochastic or deterministic rounding)
    """
    # Ensure input is valid
    u = np.clip(np.asarray(unit_vector, dtype=np.float32), 0.0, 1.0)

    if len(u) != 5:
        raise ValueError(f"Expected 5D vector, got {len(u)}D")

    # Map to parameter ranges
    fixed_quota = float(u[0])  # [0,1] → [0,1]
    prop_quota = float(u[1])  # [0,1] → [0,1]
    min_stock = float(u[2])  # [0,1] → [0,1]
    fine_amount = float(u[3]) * 2.0  # [0,1] → [0,2]

    # Stochastic rounding for ban_period
    ban_period_continuous = u[4] * 10.0  # [0,1] → [0,10]

    if use_stochastic_rounding:
        # Stochastic rounding: round down with prob (1-frac), round up with prob frac
        ban_period_floor = int(np.floor(ban_period_continuous))
        ban_period_frac = ban_period_continuous - ban_period_floor

        # Use a deterministic hash based on all parameters for reproducibility
        # This ensures same parameters always map to same ban_period
        hash_value = hash(
            (float(u[0]), float(u[1]), float(u[2]), float(u[3]), float(u[4]))
        )
        pseudo_random = (hash_value % 10000) / 10000.0

        if pseudo_random < ban_period_frac and ban_period_floor < 10:
            ban_period = ban_period_floor + 1
        else:
            ban_period = ban_period_floor
    else:
        # Deterministic rounding
        ban_period = int(np.clip(np.round(ban_period_continuous), 0, 10))

    ban_period = int(np.clip(ban_period, 0, 10))  # Ensure bounds

    return MechanismParameters(
        fixed_quota=fixed_quota,
        prop_quota=prop_quota,
        min_stock=min_stock,
        fine_amount=fine_amount,
        ban_period=ban_period,
    )


def mechanism_to_unit_vector(params: MechanismParameters) -> np.ndarray:
    """Map mechanism parameters back to unit vector (inverse mapping).

    This is useful for initialization or analysis purposes.

    Args:
        params: MechanismParameters object

    Returns:
        5D unit vector in [0,1]^5
    """
    return np.array(
        [
            params.fixed_quota,  # [0,1] → [0,1]
            params.prop_quota,  # [0,1] → [0,1]
            params.min_stock,  # [0,1] → [0,1]
            params.fine_amount / 2.0,  # [0,2] → [0,1]
            params.ban_period / 10.0,  # {0,...,10} → [0,1]
        ],
        dtype=np.float32,
    )
