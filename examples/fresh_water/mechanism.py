from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class WaterMechanism(Mechanism):
    fixed_quota: float
    prop_quota: float
    min_stock: float
    fine_amount: float

    risk_penalty_scale: float
    risk_penalty_power: float
    under_irrigation_penalty_scale: float

    def __post_init__(self) -> None:
        # max_pull_fraction = 0.0001 -> individual farm / small water user
        # max_pull_fraction = 0.0005 -> large farm / small irrigation system
        # max_pull_fraction = 0.001  -> irrigation district / small utility
        # max_pull_fraction = 0.005  -> municipality / industrial user / large irrigation district
        # max_pull_fraction = 0.01   -> major municipality / large industrial user
        # max_pull_fraction = 0.05   -> unrealistic for one agent; behaves like regional extraction

        # Quota scales:
        # 0.001 -> severe drought restriction
        # 0.003 -> strict regulation
        # 0.005 -> moderate (municipality / irrigation district)
        # 0.01  -> permissive
        # >0.05 -> effectively unconstrained
        
        # fixed_quota = protected reservoir fullness
        assert 0.6 <= self.fixed_quota <= 0.95
        assert 1e-7 <= self.prop_quota <= 1e-4
        assert 0.0 <= self.min_stock <= 1.0
        assert 0.0 <= self.fine_amount <= 0.1
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0
        assert 0.0 <= self.under_irrigation_penalty_scale <= 1.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,  # maps [1,5] -> [0,1]
                self.under_irrigation_penalty_scale,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
            "under_irrigation_penalty_scale",
        ]


class WaterMechanismSpace(MechanismSpace):
    ALL_PARAMS = [
        "fixed_quota",
        "prop_quota",
        "min_stock",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
        "under_irrigation_penalty_scale"
    ]

    def __init__(
        self,
        use_stochastic_rounding: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 0.85,
        default_prop_quota: float = 1e-5,
        default_min_stock: float = 0.15,
        default_fine_amount: float = 0.05,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
        default_under_irrigation_penalty_scale: float = 0.25,
    ):
        super().__init__()

        self.use_stochastic_rounding = use_stochastic_rounding

        self.optimize_params = optimize_params or [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
            "under_irrigation_penalty_scale",
        ]

        self.dimension = len(self.optimize_params)
        self.full_dimension = len(self.ALL_PARAMS)

        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "prop_quota": default_prop_quota,
            "min_stock": default_min_stock,
            "fine_amount": default_fine_amount,
            "risk_penalty_scale": default_risk_penalty_scale,
            "risk_penalty_power": default_risk_penalty_power,
            "under_irrigation_penalty_scale": default_under_irrigation_penalty_scale
        }

    def _denormalize_param(self, name: str, value: float, u: np.ndarray) -> float | int:
        if name == "fixed_quota":
            return 0.6 + value * (0.95 - 0.6)
        if name == "prop_quota":
            return 1e-7 + value * (1e-4 - 1e-7)
        if name == "fine_amount":
            return value * 0.10   # map [0,1] -> [0,0.1]
        if name == "risk_penalty_scale":
            return value
        if name == "risk_penalty_power":
            return 1.0 + 4.0 * value
        if name == "under_irrigation_penalty_scale":
            return value
        return value
    
    def _denormalize(self, u: np.ndarray) -> dict:
        result = {}
        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]), u)
        return result

    def _normalize_param(self, name: str, value: float | int) -> float:
        if name == "fixed_quota":
            return (float(value) - 0.6) / (0.95 - 0.6)
        if name == "prop_quota":
            return (float(value) - 1e-7) / (1e-4 - 1e-7)
        if name == "fine_amount":
            return float(value) / 0.10
        if name == "risk_penalty_scale":
            return value
        if name == "risk_penalty_power":
            return (float(value) - 1.0) / 4.0
        if name == "under_irrigation_penalty_scale":
            return value

        return float(value)

    def default(self) -> WaterMechanism:
        return WaterMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            prop_quota=self.defaults["prop_quota"],
            min_stock=self.defaults["min_stock"],
            fine_amount=self.defaults["fine_amount"],
            risk_penalty_scale=self.defaults["risk_penalty_scale"],
            risk_penalty_power=self.defaults["risk_penalty_power"],
            under_irrigation_penalty_scale=self.defaults["under_irrigation_penalty_scale"]
        )

    def encode(self, m: WaterMechanism) -> NDArray[np.float32]:
        values = []

        for name in self.optimize_params:
            raw = getattr(m, name)
            values.append(self._normalize_param(name, raw))

        return np.array(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = dict(self.defaults)

        for i, name in enumerate(self.optimize_params):
            params[name] = self._denormalize_param(name, float(u[i]), u)

        mech = WaterMechanism(
            fixed_quota=params["fixed_quota"],
            prop_quota=params["prop_quota"],
            min_stock=params["min_stock"],
            fine_amount=params["fine_amount"],
            risk_penalty_scale=params["risk_penalty_scale"],
            risk_penalty_power=params["risk_penalty_power"],
            under_irrigation_penalty_scale=params["under_irrigation_penalty_scale"]
        )

        return self.clip(mech)

    def clip(self, m: WaterMechanism) -> WaterMechanism:
        return WaterMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0.6, 0.95)),
            prop_quota=float(np.clip(m.prop_quota, 1e-7, 1e-4)),
            min_stock=float(np.clip(m.min_stock, 0.0, 1.0)),
            fine_amount=float(np.clip(m.fine_amount, 0.0, 0.1)),
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0, 1)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
            under_irrigation_penalty_scale=float(np.clip(m.under_irrigation_penalty_scale, 0, 1)),
        )

    def from_dict(self, cfg: dict) -> WaterMechanism:
        if "risk_penalty_scale" not in cfg:
            cfg = {**cfg, "risk_penalty_scale": self.defaults["risk_penalty_scale"]}
        if "risk_penalty_power" not in cfg:
            cfg = {**cfg, "risk_penalty_power": self.defaults["risk_penalty_power"]}
        return WaterMechanism(**cfg)