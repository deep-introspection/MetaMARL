from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace

# TODO move this to core/mechanisms and generalize to constraint mechanism
@dataclass(frozen=True)
class WaterMechanism(Mechanism):
    fixed_quota: float
    max_demand_frac: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float

    def __post_init__(self) -> None:
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.max_demand_frac <= 1.0
        assert 0.0 <= self.fine_amount <= 1.0
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.max_demand_frac,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,  # maps [1,5] -> [0,1]
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
        ]


class WaterMechanismSpace(MechanismSpace):
    # CHANGED:
    # prop_quota and min_stock removed.
    # min_demand_frac and max_demand_frac added.
    ALL_PARAMS = [
        "fixed_quota",
        "max_demand_frac",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
    ]

    def __init__(
        self,
        use_stochastic_rounding: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 1.0,
        default_max_demand_frac: float = 1.0,
        default_fine_amount: float = 0.05,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
    ):
        super().__init__()

        self.use_stochastic_rounding = use_stochastic_rounding

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
        }

    def _denormalize_param(self, name: str, value: float, u: np.ndarray) -> float | int:
        # if name == "fixed_quota":
        #     return 0.6 + value * (0.95 - 0.6)
        
        if name == "risk_penalty_power":
            return 1.0 + 4.0 * value

        # CHANGED:
        # min_demand_frac controls the floor of allowed irrigation demand.
        # if name == "min_demand_frac":
        #     return value * 0.35  # maps [0,1] -> [0,0.35]

        # # CHANGED:
        # # max_demand_frac controls the ceiling of allowed irrigation demand.
        # # We enforce max_demand_frac >= min_demand_frac after decoding.
        # if name == "max_demand_frac":
        #     return 0.35 + value * (1.0 - 0.35)  # maps [0,1] -> [0.35,1.0]

        # if name == "fine_amount":
        #     return value * 0.10

        # if name == "risk_penalty_scale":
        #     return value

        # if name == "risk_penalty_power":
        #     return 1.0 + 4.0 * value

        # if name == "under_irrigation_penalty_scale":
        #     return value
        
        # if name == "max_farm_area_m2":
        #     return 100_000 + value * (20_000_000 - 100_000)

        return value

    def _denormalize(self, u: np.ndarray) -> dict:
        result = {}
        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]), u)

        return result

    def _normalize_param(self, name: str, value: float | int) -> float:
        # if name == "fixed_quota":
        #     return (float(value) - 0.6) / (0.95 - 0.6)

        if name == "risk_penalty_power":
            return (value - 1.0) / 4.0

        # CHANGED:
        # Normalize demand-fraction quota parameters.
        # if name == "min_demand_frac":
        #     return float(value) / 0.35

        # if name == "max_demand_frac":
        #     return (float(value) - 0.35) / (1.0 - 0.35)

        # if name == "fine_amount":
        #     return float(value) / 0.10

        # if name == "risk_penalty_scale":
        #     return float(value)

        # if name == "risk_penalty_power":
        #     return (float(value) - 1.0) / 4.0

        # if name == "under_irrigation_penalty_scale":
        #     return float(value)
        
        # if name == "max_farm_area_m2":
        #     return (float(value) - 100_000) / (20_000_000 - 100_000)

        return float(value)

    def default(self) -> WaterMechanism:
        return WaterMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            max_demand_frac=self.defaults["max_demand_frac"],
            fine_amount=self.defaults["fine_amount"],
            risk_penalty_scale=self.defaults["risk_penalty_scale"],
            risk_penalty_power=self.defaults["risk_penalty_power"],
        )

    def encode(self, m: WaterMechanism) -> NDArray[np.float32]:
        values = [
            np.clip(
                self._normalize_param(name, getattr(m, name)),
                0.0,
                1.0,
            )
            for name in self.optimize_params
        ]

        return np.asarray(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = dict(self.defaults)

        for i, name in enumerate(self.optimize_params):
            params[name] = self._denormalize_param(name, float(u[i]), u)

        return self.clip(WaterMechanism(**params))

    def clip(self, m: WaterMechanism) -> WaterMechanism:
        return WaterMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0.0, 1.0)),
            max_demand_frac=float(np.clip(m.max_demand_frac, 0.0, 1.0)),
            fine_amount=float(np.clip(m.fine_amount, 0.0, 1.0)),
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0.0, 1.0)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
        )

    def from_dict(self, cfg: dict) -> WaterMechanism:
        cfg = dict(cfg)
        # CHANGED:
        # Backward-compatible migration for old configs.
        # If old prop_quota/min_stock appear, ignore them and use the new defaults.
        cfg.pop("prop_quota", None)
        cfg.pop("min_stock", None)
        cfg.pop("min_demand_frac", None)
        cfg.pop("under_irrigation_penalty_scale", None)
        cfg.pop("max_farm_area_m2", None)

        for name, default_value in self.defaults.items():  # CHANGED
            cfg.setdefault(name, default_value)

        mech = WaterMechanism(
            fixed_quota=cfg["fixed_quota"],
            max_demand_frac=cfg["max_demand_frac"],
            fine_amount=cfg["fine_amount"],
            risk_penalty_scale=cfg["risk_penalty_scale"],
            risk_penalty_power=cfg["risk_penalty_power"],
        )

        return self.clip(mech)
