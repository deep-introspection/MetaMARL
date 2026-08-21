from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class FisheryMechanism(Mechanism):
    fixed_quota: float
    max_demand_frac: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float
    restoration_subsidy: float

    def __post_init__(self) -> None:
        assert 0.0 <= self.fixed_quota <= 1.0
        # TODO rename max demand frac to range
        assert 0.0 <= self.max_demand_frac <= 1.0
        assert 0.0 <= self.fine_amount <= 1.0
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0
        assert 0.0 <= self.restoration_subsidy <= 0.5

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.max_demand_frac,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,
                self.restoration_subsidy / 0.5,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "max_demand_frac",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
            "restoration_subsidy",
        ]


class FisheryMechanismSpace(MechanismSpace):
    ALL_PARAMS = [
        "fixed_quota",
        "max_demand_frac",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
        "restoration_subsidy",
    ]

    def __init__(
        self,
        use_stochastic_roundting: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 1.0,
        default_max_demand_frac: float = 1.0,
        default_fine_amount: float = 0.5,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
        default_restoration_subsidy: float = 0.10,
    ):
        super().__init__()

        self.use_stochastic_roundting = use_stochastic_roundting

        # TODO fix this
        self.optimize_params = (
            list(self.ALL_PARAMS)
            if optimize_params is None
            else list(optimize_params)
        )

        self.dimension = len(self.optimize_params)
        self.full_dimension = len(self.ALL_PARAMS)

        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "max_demand_frac": default_max_demand_frac,
            "fine_amount": default_fine_amount,
            "risk_penalty_scale": default_risk_penalty_scale,
            "risk_penalty_power": default_risk_penalty_power,
            "restoration_subsidy": default_restoration_subsidy,
        }

    def default(self) -> FisheryMechanism:
        return self.clip(
            FisheryMechanism(
                fixed_quota=self.defaults["fixed_quota"],
                max_demand_frac=self.defaults["max_demand_frac"],
                fine_amount=self.defaults["fine_amount"],
                risk_penalty_scale=self.defaults["risk_penalty_scale"],
                risk_penalty_power=self.defaults["risk_penalty_power"],
                restoration_subsidy=self.defaults["restoration_subsidy"],
            )
        )

    def _denormalize_param(self, name: str, value: float) -> float:
        value = float(value)

        if name == "risk_penalty_power":
            return 1.0 + 4.0 * value
        if name == "restoration_subsidy":
            return value * 0.5

        return value

    def _normalize_param(self, name: str, value: float) -> float:
        value = float(value)

        if name == "risk_penalty_power":
            return (value - 1.0) / 4.0
        if name == "restoration_subsidy":
            return value / 0.5

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
        """
        values = [
            np.clip(
                self._normalize_param(name, getattr(m, name)),
                0.0,
                1.0,
            )
            for name in self.optimize_params
        ]

        return np.asarray(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> FisheryMechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)
        params = dict(self.defaults)

        for i, name in enumerate(self.optimize_params):
            params[name] = self._denormalize_param(name, float(u[i]))

        return self.clip(FisheryMechanism(**params))

    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0.0, 1.0)),
            max_demand_frac=float(np.clip(m.max_demand_frac, 0.0, 1.0)),
            fine_amount=float(np.clip(m.fine_amount, 0.0, 1.0)),
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0.0, 1.0)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
            restoration_subsidy=float(np.clip(m.restoration_subsidy, 0.0, 0.5))
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        cfg = dict(cfg)

        for name, default_value in self.defaults.items():
            cfg.setdefault(name, default_value)

        mech = FisheryMechanism(
            fixed_quota=cfg["fixed_quota"],
            max_demand_frac=cfg["max_demand_frac"],
            fine_amount=cfg["fine_amount"],
            risk_penalty_scale=cfg["risk_penalty_scale"],
            risk_penalty_power=cfg["risk_penalty_power"],
            restoration_subsidy=cfg["restoration_subsidy"],
        )

        return self.clip(mech)