"""V0 fishery regulatory mechanism and mechanism space.

Defines the mechanism vector that the outer ES loop optimises.  The V0
mechanism models a simple deterministic quota/fine/ban scheme:

- ``fixed_quota``  — hard cap on per-agent harvest per step.
- ``prop_quota``   — proportional cap relative to current normalised stock.
- ``min_stock``    — minimum stock level below which fishing is prohibited.
- ``fine_amount``  — penalty multiplier applied to the violation magnitude.
- ``ban_period``   — number of steps an agent is barred from fishing after
  a detected violation.
- ``catch_prob``   — probability that a violation is detected (stochastic
  enforcement).

Classes
-------
FisheryMechanism
    Immutable dataclass holding a single realisation of the mechanism.
FisheryMechanismSpace
    Encodes / decodes mechanism vectors for the ES search and manages
    normalisation and stochastic discretisation of ``ban_period``.
"""

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
        catch_prob: Probability of detecting a violation (0 to 1)
    """

    fixed_quota: float
    prop_quota: float
    min_stock: float
    fine_amount: float
    ban_period: int
    catch_prob: float
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
        assert 0.0 <= self.catch_prob <= 1.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        """Serialise the mechanism to a normalised float32 vector in [0, 1]^6.

        Normalisation applied per parameter:

        - ``fixed_quota``, ``prop_quota``, ``min_stock``, ``catch_prob``:
          already in [0, 1] — returned as-is.
        - ``fine_amount``:  divided by ``max_fine``  → [0, 1].
        - ``ban_period``:   divided by ``max_ban``   → [0, 1].

        Returns
        -------
        np.ndarray
            Shape ``(6,)``, dtype ``float32``.
            Order: ``[fixed_quota, prop_quota, min_stock,
            fine_amount/max_fine, ban_period/max_ban, catch_prob]``.
        """
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.fine_amount / self.max_fine,
                self.ban_period / self.max_ban,
                self.catch_prob,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        """Return the ordered list of parameter names matching :meth:`to_vector`.

        Returns
        -------
        list of str
            ``["fixed_quota", "prop_quota", "min_stock",
            "fine_amount", "ban_period", "catch_prob"]``
        """
        return [
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "ban_period",
            "catch_prob",
        ]


class FisheryMechanismSpace(MechanismSpace):
    """Mechanism space for the V0 fishery regulatory mechanism.

    Manages the encoding and decoding of mechanism vectors used by the outer
    Evolution Strategies (ES) optimizer.  The space supports partial
    optimisation: only the parameters listed in ``optimize_params`` are varied
    by ES; all other parameters are held fixed at their defaults.

    The full observation vector exposed to fishing agents always contains all
    six parameters (``full_dimension = 6``), regardless of which subset ES
    is optimising.

    Parameters
    ----------
    use_stochastic_roundting : bool, optional
        If ``True`` (default), the continuous ban-period value produced by ES
        is rounded to an integer using stochastic rounding (i.e. probabilistic
        floor/ceil based on the fractional part).  If ``False``, standard
        round-half-up is used.
    max_fine : float, optional
        Maximum fine amount used for normalisation.  Default is ``5.0``.
    max_ban : int, optional
        Maximum ban duration (steps) used for normalisation.  Default is ``50``.
    optimize_params : list of str or None, optional
        Names of the parameters that ES will optimise.  If ``None``, defaults
        to an empty list (all parameters fixed — useful for debugging).
    default_fixed_quota : float, optional
        Default value for ``fixed_quota`` when not optimised.  Default ``1.0``.
    default_prop_quota : float, optional
        Default value for ``prop_quota`` when not optimised.  Default ``1.0``.
    default_min_stock : float, optional
        Default value for ``min_stock`` when not optimised.  Default ``0.1``.
    default_fine_amount : float, optional
        Default value for ``fine_amount`` when not optimised.  Default ``0.5``.
    default_ban_period : int, optional
        Default value for ``ban_period`` when not optimised.  Default ``0``.
    default_catch_prob : float, optional
        Default value for ``catch_prob`` when not optimised.  Default ``1.0``.

    Attributes
    ----------
    dimension : int
        Number of parameters actively optimised by ES (``len(optimize_params)``).
    full_dimension : int
        Total number of mechanism parameters (always 6 for V0).
    """

    # All optimizable parameter names
    ALL_PARAMS = [
        "fixed_quota",
        "prop_quota",
        "min_stock",
        "fine_amount",
        "ban_period",
        "catch_prob",
    ]

    def __init__(
        self,
        use_stochastic_roundting: bool = True,
        max_fine: float = 5.0,
        max_ban: int = 50,
        # Which parameters ES optimizes (default: only min_stock, fine_amount)
        optimize_params: list[str] | None = None,
        # Default/fixed values for all parameters
        default_fixed_quota: float = 1.0,
        default_prop_quota: float = 1.0,
        default_min_stock: float = 0.1,
        default_fine_amount: float = 0.5,
        default_ban_period: int = 0,
        default_catch_prob: float = 1.0,
    ):
        super().__init__()
        self.use_stochastic_roundting = use_stochastic_roundting
        self.max_fine = max_fine
        self.max_ban = max_ban

        # Parameters to optimize (default: min_stock, fine_amount)
        # TODO ability to toggle this from config
        # TODO ability to freeze mechanism for debugging
        self.optimize_params = []
        # optimize_params or [
        #     "fixed_quota",
        #     "prop_quota",
        #     "min_stock",
        #     "fine_amount",
        #     "ban_period",
        #     "catch_prob",
        # ]
        self.dimension = len(self.optimize_params)
        # Full mechanism dimension (for observation space - agent sees all params)
        self.full_dimension = len(self.ALL_PARAMS)

        # Default values for fixed parameters
        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "prop_quota": default_prop_quota,
            "min_stock": default_min_stock,
            "fine_amount": default_fine_amount,
            "ban_period": default_ban_period,
            "catch_prob": default_catch_prob,
        }

    def _discretize_ban(self, ban_period_cont: float, u: np.ndarray) -> int:
        """Convert a continuous ban-period value to a non-negative integer.

        When ``use_stochastic_roundting=True`` a pseudo-random bit derived
        from the hash of the full mechanism vector ``u`` is used to decide
        between ``floor`` and ``ceil``, so the expected value equals the
        continuous input while the gradient estimator remains unbiased.

        Parameters
        ----------
        ban_period_cont : float
            Continuous ban-period value in ``[0, max_ban]``.
        u : np.ndarray
            Full (un-denormalized) ES parameter vector; used as a hash seed
            for the stochastic rounding.

        Returns
        -------
        int
            Integer ban period in ``[0, max_ban]``.
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

    def _denormalize_param(self, name: str, value: float) -> float:
        """Denormalize a single parameter from [0,1] to its actual range."""
        if name == "fine_amount":
            return value * self.max_fine
        elif name == "ban_period":
            return value * self.max_ban
        else:
            # fixed_quota, prop_quota, min_stock, catch_prob are all [0,1]
            return value

    def _denormalize(self, u: np.ndarray) -> dict:
        """Denormalize only the optimized parameters."""
        result = {}
        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]))
        return result

    def default(self) -> FisheryMechanism:
        """Return a :class:`FisheryMechanism` populated with the default parameter values.

        Returns
        -------
        FisheryMechanism
            Mechanism with all parameters set to their constructor defaults.
        """
        return FisheryMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            prop_quota=self.defaults["prop_quota"],
            min_stock=self.defaults["min_stock"],
            fine_amount=self.defaults["fine_amount"],
            ban_period=int(self.defaults["ban_period"]),
            catch_prob=self.defaults["catch_prob"],
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

    def _normalize_param(self, name: str, value: float) -> float:
        """Normalize a parameter to [0,1] range."""
        if name == "fine_amount":
            return value / self.max_fine
        elif name == "ban_period":
            return value / self.max_ban
        else:
            return value

    def encode(self, m: FisheryMechanism) -> NDArray[np.float32]:
        """Encode a mechanism to the normalised ES search vector.

        Only the parameters in ``optimize_params`` are included; fixed
        parameters are omitted.  The same normalisation as
        :meth:`FisheryMechanism.to_vector` is applied:

        - ``fine_amount``  → divided by ``max_fine``.
        - ``ban_period``   → divided by ``max_ban``.
        - All others       → unchanged (already in [0, 1]).

        Parameters
        ----------
        m : FisheryMechanism
            Mechanism instance to encode.

        Returns
        -------
        NDArray[np.float32]
            Shape ``(dimension,)``, values in ``[0, 1]``.
        """
        values = []
        for name in self.optimize_params:
            raw = getattr(m, name)
            values.append(self._normalize_param(name, raw))
        return np.array(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        """Decode a normalised ES vector into a :class:`FisheryMechanism`.

        The input is first clipped to ``[0, 1]``, then de-normalised
        (reverse of :meth:`encode`).  Non-optimised parameters are filled
        from ``self.defaults``.  ``ban_period`` is discretised to an integer
        using :meth:`_discretize_ban`.

        Parameters
        ----------
        x : NDArray[np.float32]
            Shape ``(dimension,)``, values expected in ``[0, 1]``.

        Returns
        -------
        FisheryMechanism
            A valid, clipped mechanism instance.
        """
        # insure input is valid
        u = np.clip(self._validate(x), 0.0, 1.0)

        # Start with defaults, override with optimized params
        params = dict(self.defaults)
        optimized = self._denormalize(u)
        params.update(optimized)

        # Discretize ban_period if it was optimized
        if "ban_period" in self.optimize_params:
            params["ban_period"] = self._discretize_ban(params["ban_period"], u)
        else:
            params["ban_period"] = int(params["ban_period"])

        mech = FisheryMechanism(
            fixed_quota=params["fixed_quota"],
            prop_quota=params["prop_quota"],
            min_stock=params["min_stock"],
            fine_amount=params["fine_amount"],
            ban_period=params["ban_period"],
            catch_prob=params["catch_prob"],
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

        return self.clip(mech)

    def clip(self, m: FisheryMechanism) -> FisheryMechanism:
        """Return a copy of *m* with all parameters clamped to their valid ranges.

        Parameters
        ----------
        m : FisheryMechanism
            Mechanism instance to clip.

        Returns
        -------
        FisheryMechanism
            New mechanism with:
            ``fixed_quota, prop_quota, min_stock, catch_prob`` ∈ [0, 1];
            ``fine_amount`` ∈ [0, max_fine];
            ``ban_period`` ∈ [0, max_ban].
        """
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0, 1)),
            prop_quota=float(np.clip(m.prop_quota, 0, 1)),
            min_stock=float(np.clip(m.min_stock, 0, 1)),
            fine_amount=float(np.clip(m.fine_amount, 0, self.max_fine)),
            ban_period=int(np.clip(m.ban_period, 0, self.max_ban)),
            catch_prob=float(np.clip(m.catch_prob, 0, 1)),
            max_fine=self.max_fine,
            max_ban=self.max_ban,
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        """Build a :class:`FisheryMechanism` from a plain parameter dictionary.

        Provides backward compatibility: if ``catch_prob`` is absent from
        *cfg*, it is filled in with ``1.0`` (always-catch behaviour).

        Parameters
        ----------
        cfg : dict
            Mapping of parameter name → value.  Must contain at minimum:
            ``fixed_quota``, ``prop_quota``, ``min_stock``, ``fine_amount``,
            ``ban_period``.  ``catch_prob`` is optional.

        Returns
        -------
        FisheryMechanism
            Mechanism constructed from *cfg*, with ``max_fine`` and ``max_ban``
            injected from ``self``.
        """
        # Default catch_prob to 1.0 if not specified (backwards compatible)
        if "catch_prob" not in cfg:
            cfg = {**cfg, "catch_prob": 1.0}
        return FisheryMechanism(**cfg, max_fine=self.max_fine, max_ban=self.max_ban)
