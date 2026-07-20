from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace

EPS = 1e-8

@dataclass(frozen=True)
class FisheryMechanism(Mechanism):
    fixed_quota: float
    min_demand_frac: float
    max_demand_frac: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float

    def __post_init__(self) -> None:
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.min_demand_frac <= 1.0
        # TODO rename max demand frac to range
        assert 0.0 <= self.max_demand_frac <= 1.0
        assert self.min_demand_frac <= self.max_demand_frac
        assert 0.0 <= self.fine_amount <= 1.0
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        remaining_range = 1.0 - self.min_demand_frac

        if remaining_range <= EPS:
            max_demand_range = 0.0
        else:
            max_demand_range = (
                self.max_demand_frac - self.min_demand_frac
            ) / remaining_range

        return np.array(
            [
                self.fixed_quota,
                self.min_demand_frac,
                max_demand_range,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "min_demand_frac",
            "max_demand_frac",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
        ]


class FisheryMechanismSpace(MechanismSpace):
    ALL_PARAMS = [
        "fixed_quota",
        "min_demand_frac",
        "max_demand_frac",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
    ]

    def __init__(
        self,
        use_stochastic_roundting: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 1.0,
        default_min_demand_frac: float = 0.05,
        default_max_demand_frac: float = 1.0,
        default_fine_amount: float = 0.5,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
    ):
        super().__init__()

        self.use_stochastic_roundting = use_stochastic_roundting

        self.optimize_params = optimize_params or [
            "fixed_quota",
            "min_demand_frac",
            "max_demand_frac",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
        ]

        self.dimension = len(self.optimize_params)
        self.full_dimension = len(self.ALL_PARAMS)

        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "min_demand_frac": default_min_demand_frac,
            "max_demand_frac": default_max_demand_frac,
            "fine_amount": default_fine_amount,
            "risk_penalty_scale": default_risk_penalty_scale,
            "risk_penalty_power": default_risk_penalty_power,
        }

    def default(self) -> FisheryMechanism:
        return self.clip(
            FisheryMechanism(
                fixed_quota=self.defaults["fixed_quota"],
                min_demand_frac=self.defaults["min_demand_frac"],
                max_demand_frac=self.defaults["max_demand_frac"],
                fine_amount=self.defaults["fine_amount"],
                risk_penalty_scale=self.defaults["risk_penalty_scale"],
                risk_penalty_power=self.defaults["risk_penalty_power"],
            )
        )

    def _denormalize_param(self, name: str, value: float) -> float:
        value = float(value)

        if name == "risk_penalty_power":
            return 1.0 + 4.0 * value

        return value

    def _normalize_param(self, name: str, value: float) -> float:
        value = float(value)

        if name == "risk_penalty_power":
            return (value - 1.0) / 4.0

        return value

    def _denormalize(self, u: np.ndarray) -> dict:
        result = {}

        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]))

        return result

    def encode(
        self,
        m: FisheryMechanism,
    ) -> NDArray[np.float32]:
        """
        Convert a physical mechanism into the normalized ES representation.

        For max_demand_frac, encode its relative position in the interval
        [min_demand_frac, 1.0].
        """
        values = []

        for name in self.optimize_params:
            if name == "max_demand_frac":
                remaining_range = 1.0 - m.min_demand_frac

                if remaining_range <= EPS:
                    normalized_value = 0.0
                else:
                    normalized_value = (
                        m.max_demand_frac
                        - m.min_demand_frac
                    ) / remaining_range

                values.append(
                    float(np.clip(normalized_value, 0.0, 1.0))
                )
                continue

            raw = getattr(m, name)
            normalized_value = self._normalize_param(name, raw)

            values.append(
                float(np.clip(normalized_value, 0.0, 1.0))
            )

        return np.array(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> FisheryMechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = dict(self.defaults)
        decoded_values = self._denormalize(u)

        for name, value in decoded_values.items():
            if name != "max_demand_frac":
                params[name] = value

        # The optimizer's max coordinate is a relative range coordinate.
        if "max_demand_frac" in decoded_values:
            min_demand_frac = float(params["min_demand_frac"])
            max_range = float(decoded_values["max_demand_frac"])

            params["max_demand_frac"] = (
                min_demand_frac
                + max_range * (1.0 - min_demand_frac)
            )

        mech = FisheryMechanism(
            fixed_quota=params["fixed_quota"],
            min_demand_frac=params["min_demand_frac"],
            max_demand_frac=params["max_demand_frac"],
            fine_amount=params["fine_amount"],
            risk_penalty_scale=params["risk_penalty_scale"],
            risk_penalty_power=params["risk_penalty_power"],
        )

        return self.clip(mech)

    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        min_demand_frac = float(np.clip(m.min_demand_frac, 0.0, 1.0))
        max_demand_frac = float(np.clip(m.max_demand_frac, 0.0, 1.0))

        max_demand_frac = max(max_demand_frac, min_demand_frac)

        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0.0, 1.0)),
            min_demand_frac=min_demand_frac,
            max_demand_frac=max_demand_frac,
            fine_amount=float(np.clip(m.fine_amount, 0.0, 1.0)),
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0.0, 1.0)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        cfg = dict(cfg)

        cfg.pop("prop_quota", None)
        cfg.pop("min_stock", None)
        cfg.pop("target_stock", None)

        for name, default_value in self.defaults.items():
            cfg.setdefault(name, default_value)

        mech = FisheryMechanism(
            fixed_quota=cfg["fixed_quota"],
            min_demand_frac=cfg["min_demand_frac"],
            max_demand_frac=cfg["max_demand_frac"],
            fine_amount=cfg["fine_amount"],
            risk_penalty_scale=cfg["risk_penalty_scale"],
            risk_penalty_power=cfg["risk_penalty_power"],
        )

        return self.clip(mech)