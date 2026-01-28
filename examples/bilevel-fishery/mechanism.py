import numpy as np
from beartype import beartype
from jaxtyping import Array, Float

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


class FisheryMechanism(Mechanism):
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
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.prop_quota <= 1.0
        assert 0.0 <= self.min_stock <= 1.0
        assert 0.0 <= self.fine_amount <= 2.0
        assert 0 <= self.ban_period <= 10

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.fine_amount / 2.0,
                self.ban_period / 10.0,
            ],
            dtype=np.float32,
        )


class FisheryMechnaismSpace(MechanismSpace):
    def __init__(self, use_stochastic_roundting: bool = True):
        super().__init__()
        self.use_stochastic_roundting = use_stochastic_roundting
        self.dimension = 5

    # private
    def _discretize_ban(self, ban_period_cont: float, u: np.ndarray) -> int:
        if not self.use_stochastic_roundting:
            return int(np.clip(round(ban_period_cont), 0, 10))

        floor = int(np.floor(ban_period_cont))
        frac = ban_period_cont - floor

        h = hash(tuple(map(float, u)))
        pseudo_random = (h % 10000) / 10000.0

        if pseudo_random < frac and floor < 10:
            return floor + 1
        return floor

    def _denormalize(self, u: np.ndarray) -> dict:
        return {
            "fixed_quota": float(u[0]),
            "prop_quota": float(u[1]),
            "min_stock": float(u[2]),
            "fine_amount": float(u[3]) * 2.0,
            "ban_period_cont": float(u[4]) * 10.0,
        }

    @beartype
    def encode(self, m: FisheryMechanism) -> Float[Array, "5"]:
        return m.to_vector()

    @beartype
    def decode(self, x: Float[Array, "5"]) -> Mechanism:
        # insure input is valid
        u = np.clip(self._validate(x), 0.0, 1.0)

        params = self._denormalize(u)
        ban = self._discretize_ban(params.pop("ban_period_cont"), u)

        mech = FisheryMechanism(
            **params,
            ban_period=ban,
        )

        return self.clip(mech)

    @beartype
    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0, 1)),
            prop_quota=float(np.clip(m.prop_quota, 0, 1)),
            min_stock=float(np.clip(m.min_stock, 0, 1)),
            fine_amount=float(np.clip(m.fine_amount, 0, 2)),
            ban_period=int(np.clip(m.ban_period, 0, 10)),
        )
