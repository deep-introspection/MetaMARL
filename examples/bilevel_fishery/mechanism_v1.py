from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class FisheryMechanism(Mechanism):
    fixed_quota: float
    prop_quota: float
    min_stock: float
    target_stock: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float

    def __post_init__(self) -> None:
        # Q: review these ranges. its probably not realistic to pull 80% of basin capacity !
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.prop_quota <= 1.0
        assert 0.0 <= self.min_stock <= 1.0
        assert 0.0 <= self.target_stock <= 1.0
        assert 0.0 <= self.fine_amount <= 1.0
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.target_stock,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,  # maps [1,5] -> [0,1]
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "target_stock",
            "fine_amount",
            # CHANGED
            "risk_penalty_scale",
            "risk_penalty_power",
        ]


class FisheryMechanismSpace(MechanismSpace):
    ALL_PARAMS = [
        "fixed_quota",
        "prop_quota",
        "min_stock",
        "target_stock",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
    ]

    def __init__(
        self,
        use_stochastic_roundting: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 1.0,
        default_prop_quota: float = 1.0,
        default_min_stock: float = 0.1,
        default_target_stock: float = 0.2,
        default_fine_amount: float = 0.5,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
    ):
        super().__init__()
        self.use_stochastic_roundting = use_stochastic_roundting

        self.optimize_params = optimize_params or [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "target_stock",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
        ]

        self.dimension = len(self.optimize_params)
        self.full_dimension = len(self.ALL_PARAMS)

        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "prop_quota": default_prop_quota,
            "min_stock": default_min_stock,
            "target_stock": default_target_stock,
            "fine_amount": default_fine_amount,
            "risk_penalty_scale": default_risk_penalty_scale,
            "risk_penalty_power": default_risk_penalty_power,
        }

    def _denormalize_param(self, name: str, value: float) -> float:
        if name in ("fine_amount", "risk_penalty_scale"):
            return value
        elif name == "risk_penalty_power":
            return 1.0 + 4.0 * value
        else:
            return value

    def _denormalize(self, u: np.ndarray) -> dict:
        result = {}
        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]))
        return result

    def default(self) -> FisheryMechanism:
        return FisheryMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            prop_quota=self.defaults["prop_quota"],
            min_stock=self.defaults["min_stock"],
            target_stock=self.defaults["target_stock"],
            fine_amount=self.defaults["fine_amount"],
            risk_penalty_scale=self.defaults["risk_penalty_scale"],
            risk_penalty_power=self.defaults["risk_penalty_power"]
        )

    def _normalize_param(self, name: str, value: float) -> float:
        if name in ("fine_amount", "risk_penalty_scale"):
            return value
        elif name == "risk_penalty_power":
            return (value - 1.0) / 4.0
        else:
            return value

    def encode(self, m: FisheryMechanism) -> NDArray[np.float32]:
        values = []
        for name in self.optimize_params:
            raw = getattr(m, name)
            values.append(self._normalize_param(name, raw))
        return np.array(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = dict(self.defaults)
        optimized = self._denormalize(u)
        params.update(optimized)

        mech = FisheryMechanism(
            fixed_quota=params["fixed_quota"],
            prop_quota=params["prop_quota"],
            min_stock=params["min_stock"],
            target_stock=params["target_stock"],
            fine_amount=params["fine_amount"],
            # CHANGED
            risk_penalty_scale=params["risk_penalty_scale"],
            risk_penalty_power=params["risk_penalty_power"],
        )

        return self.clip(mech)

    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0, 1)),
            prop_quota=float(np.clip(m.prop_quota, 0, 1)),
            min_stock=float(np.clip(m.min_stock, 0, 1)),
            target_stock=float(np.clip(m.target_stock, 0, 1)),
            fine_amount=float(np.clip(m.fine_amount, 0, 1)),
            # CHANGED
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0, 1)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        # CHANGED: add backward-compatible defaults for new params
        if "risk_penalty_scale" not in cfg:
            cfg = {**cfg, "risk_penalty_scale": self.defaults["risk_penalty_scale"]}
        if "risk_penalty_power" not in cfg:
            cfg = {**cfg, "risk_penalty_power": self.defaults["risk_penalty_power"]}

        return FisheryMechanism(**cfg)
