from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from examples.bilevel_fishery.mechanism import FisheryMechanism


@dataclass(frozen=True)
class WaterMechanism(Mechanism):
    """Regulatory mechanism parameters for the fishery.

    Attributes:
        water_: Absolute harvest limit (0 to 1)
        prop_quota: Proportional quota factor (0 to 1)
        min_stock: Minimum stock threshold for fishing (0 to 1)
        fine_amount: Penalty per unit over-harvest (0 to max_fine)
        ban_period: Duration of ban after violation (0 to max_ban periods)

        # new water usage is like the fitness
    """

    water_threshold: float

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
        """Encode the mechanism as a normalized float32 vector in ``[0, 1]^5``.

        ``fine_amount`` and ``ban_period`` are divided by their respective
        maximums so that all components lie in ``[0, 1]``, matching the ES
        search space convention.

        Returns
        -------
        np.ndarray
            Shape ``(5,)`` float32 array with elements
            ``[fixed_quota, prop_quota, min_stock, fine_amount/max_fine,
            ban_period/max_ban]``.
        """
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
        """Return human-readable names for each parameter dimension.

        Returns
        -------
        list[str]
            Ordered list of parameter names corresponding to the components
            of :meth:`to_vector`.
        """
        return [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "ban_period",
        ]


class FisheryMechanismSpace(MechanismSpace):
    """Mechanism space for the fresh-water (fishery) regulatory problem.

    Manages encoding, decoding, clipping, and default construction of
    :class:`~examples.bilevel_fishery.mechanism.FisheryMechanism` objects.
    The ES outer loop operates on normalized ``[0, 1]^5`` vectors; this
    class handles the bijection between that space and the physically
    meaningful parameter ranges.

    Parameters
    ----------
    use_stochastic_roundting : bool, optional
        When ``True``, the continuous ``ban_period`` value is rounded
        stochastically (probabilistic rounding) rather than
        deterministically.  Stochastic rounding preserves gradient signal
        through the discretization step.  Default is ``True``.
    max_fine : float, optional
        Maximum allowed fine amount used for denormalization.
        Default is ``5.0``.
    max_ban : int, optional
        Maximum allowed ban duration (number of steps) used for
        denormalization.  Default is ``50``.
    """

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
        """Discretize a continuous ban-period value into an integer.

        When :attr:`use_stochastic_roundting` is ``True``, the floor and
        ceiling values are selected with probabilities proportional to the
        fractional part, using a hash of the full parameter vector as a
        deterministic pseudo-random source.  This avoids introducing a
        non-differentiable hard threshold in the ES landscape.

        Parameters
        ----------
        ban_period_cont : float
            Continuous ban-period value in ``[0, max_ban]``.
        u : np.ndarray
            Full normalized parameter vector used as the hash seed for
            pseudo-random rounding.

        Returns
        -------
        int
            Discretized ban period clamped to ``[0, max_ban]``.
        """
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
        """Convert a normalized ``[0, 1]^5`` vector to physical parameter values.

        Parameters
        ----------
        u : np.ndarray
            Normalized mechanism vector of shape ``(5,)``.

        Returns
        -------
        dict
            Dictionary with keys ``fixed_quota``, ``prop_quota``,
            ``min_stock``, ``fine_amount``, and ``ban_period_cont``.
        """
        return {
            "fixed_quota": float(u[0]),
            "prop_quota": float(u[1]),
            "min_stock": float(u[2]),
            "fine_amount": float(u[3]) * self.max_fine,
            "ban_period_cont": float(u[4]) * self.max_ban,
        }

    def default(self) -> FisheryMechanism:
        """Return a permissive default mechanism that allows meaningful fishing.

        The defaults are chosen so that the PPO inner loop encounters
        positive rewards from the start, avoiding a pathological policy of
        simply never fishing to dodge all penalties.

        Returns
        -------
        FisheryMechanism
            Default mechanism with relaxed quota and low fine/ban parameters.
        """
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
        """Encode a :class:`~examples.bilevel_fishery.mechanism.FisheryMechanism`
        as a normalized float32 vector.

        Parameters
        ----------
        m : FisheryMechanism
            Mechanism to encode.

        Returns
        -------
        NDArray[np.float32]
            Shape ``(5,)`` normalized vector via :meth:`FisheryMechanism.to_vector`.
        """
        return m.to_vector()

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        """Decode a normalized float32 vector into a
        :class:`~examples.bilevel_fishery.mechanism.FisheryMechanism`.

        Clips the input to ``[0, 1]``, denormalizes to physical ranges,
        discretizes the ban period, and clips the resulting mechanism to
        valid bounds.

        Parameters
        ----------
        x : NDArray[np.float32]
            Normalized mechanism vector of shape ``(5,)``.

        Returns
        -------
        Mechanism
            Decoded and clipped :class:`FisheryMechanism`.
        """
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
        """Clip all parameters of a mechanism to their valid ranges.

        Parameters
        ----------
        m : FisheryMechanism
            Mechanism whose parameters may be out of range.

        Returns
        -------
        FisheryMechanism
            New mechanism with each parameter clamped to its valid interval.
        """
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
        """Construct a :class:`~examples.bilevel_fishery.mechanism.FisheryMechanism`
        from a plain configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Mapping of parameter names to values.  Must contain all
            :class:`FisheryMechanism` fields except ``max_fine`` and
            ``max_ban``, which are injected from this space's attributes.

        Returns
        -------
        FisheryMechanism
            Constructed mechanism instance.
        """
        return FisheryMechanism(**cfg, max_fine=self.max_fine, max_ban=self.max_ban)
