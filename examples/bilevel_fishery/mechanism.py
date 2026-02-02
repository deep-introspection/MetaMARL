from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class FisheryMechanism(Mechanism):
    """Regulatory mechanism parameters for the fishery.

    Attributes:
        fixed_quota: Absolute harvest limit (0 to 1)
        prop_quota: Proportional quota factor (0 to 1)
        min_stock: Minimum stock threshold for fishing (0 to 1)
        fine_amount: Penalty per unit over-harvest (0 to max_fine)
        ban_period: Duration of ban after violation (0 to max_ban periods)
    """

    fixed_quota: float
    prop_quota: float
    min_stock: float
    fine_amount: float
    ban_period: int
    # Scaling parameters (not optimized, used for normalization)
    max_fine: float = 5.0
    max_ban: int = 50

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.prop_quota <= 1.0
        assert 0.0 <= self.min_stock <= 1.0
        assert 0.0 <= self.fine_amount <= self.max_fine
        assert 0 <= self.ban_period <= self.max_ban

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.fine_amount / self.max_fine,
                self.ban_period / self.max_ban,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "ban_period",
        ]


class FisheryMechanismSpace(MechanismSpace):
    def __init__(
        self,
        use_stochastic_roundting: bool = True,
        max_fine: float = 5.0,
        max_ban: int = 50,
    ):
        super().__init__()
        self.use_stochastic_roundting = use_stochastic_roundting
        self.max_fine = max_fine
        self.max_ban = max_ban
        self.dimension = 5

    # private
    def _discretize_ban(self, ban_period_cont: float, u: np.ndarray) -> int:
        if not self.use_stochastic_roundting:
            return int(np.clip(round(ban_period_cont), 0, self.max_ban))

        floor = int(np.floor(ban_period_cont))
        frac = ban_period_cont - floor

        h = hash(tuple(map(float, u)))
        pseudo_random = (h % 10000) / 10000.0

        if pseudo_random < frac and floor < self.max_ban:
            return floor + 1
        return floor

    def _denormalize(self, u: np.ndarray) -> dict:
        return {
            "fixed_quota": float(u[0]),
            "prop_quota": float(u[1]),
            "min_stock": float(u[2]),
            "fine_amount": float(u[3]) * self.max_fine,
            "ban_period_cont": float(u[4]) * self.max_ban,
        }

    def default(self) -> FisheryMechanism:
        # Permissive defaults so PPO learns to fish (not just avoid penalties)
        return FisheryMechanism(
            fixed_quota=0.7,
            prop_quota=0.6,
            min_stock=0.15,
            fine_amount=0.5,
            ban_period=3,
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

    def encode(self, m: FisheryMechanism) -> NDArray[np.float32]:
        return m.to_vector()

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        # insure input is valid
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = self._denormalize(u)
        ban = self._discretize_ban(params.pop("ban_period_cont"), u)

        mech = FisheryMechanism(
            **params,
            ban_period=ban,
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

        return self.clip(mech)

    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0, 1)),
            prop_quota=float(np.clip(m.prop_quota, 0, 1)),
            min_stock=float(np.clip(m.min_stock, 0, 1)),
            fine_amount=float(np.clip(m.fine_amount, 0, self.max_fine)),
            ban_period=int(np.clip(m.ban_period, 0, self.max_ban)),
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        return FisheryMechanism(**cfg, max_fine=self.max_fine, max_ban=self.max_ban)
