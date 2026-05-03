"""V1 fishery regulatory mechanism and mechanism space.

Extends the V0 mechanism by replacing the binary fine/ban enforcement model
with a *risk-sensitive continuous penalty* scheme.  Key differences from V0:

- ``target_stock`` — desired fish-stock level; agents are penalised for
  predicted shortfalls below this target.
- ``risk_penalty_scale`` — overall magnitude of the risk penalty (replaces
  ``fine_amount`` as the primary enforcement signal).
- ``risk_penalty_power`` — exponent of the shortfall term, allowing
  sub-linear (``< 2``) or super-linear (``> 2``) penalty curvature.
- ``ban_period`` and ``catch_prob`` are **removed** in V1; enforcement is
  purely through the continuous risk penalty.

Classes
-------
FisheryMechanism
    Immutable dataclass for a single V1 mechanism realisation.
FisheryMechanismSpace
    Encodes / decodes V1 mechanism vectors for the ES search.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class FisheryMechanism(Mechanism):
    """Regulatory mechanism parameters for the V1 fishery (risk-penalty variant).

    Replaces the binary fine/ban enforcement of V0 with a smooth,
    risk-sensitive penalty that discourages fishing when the stock approaches
    or drops below ``target_stock``.

    Attributes
    ----------
    fixed_quota : float
        Hard harvest cap per agent per step, in [0, 1] (normalised by
        ``max_fish``).
    prop_quota : float
        Proportional quota: the effective quota is
        ``min(fixed_quota, prop_quota * fish_norm)``.  In [0, 1].
    min_stock : float
        Stock level (normalised) below which fishing is prohibited.
        In [0, 1].
    target_stock : float
        Desired fish-stock level (normalised).  Agents are penalised for
        predicted shortfalls below this target.  In [0, 1].
    fine_amount : float
        Base fine applied to quota violations, in [0, max_fine].
    risk_penalty_scale : float
        Overall scale of the predictive collapse penalty, in [0, max_fine].
    risk_penalty_power : float
        Exponent applied to the normalised stock shortfall, in [1, 5].
        ``power=2`` gives a quadratic penalty; higher values concentrate
        the penalty near collapse.
    max_fine : float, optional
        Normalisation constant for ``fine_amount`` and ``risk_penalty_scale``.
        Default is ``5.0``.
    """

    fixed_quota: float
    prop_quota: float
    min_stock: float
    target_stock: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float
    max_fine: float = 5.0

    def __post_init__(self) -> None:
        """Validate all parameter ranges."""
        assert 0.0 <= self.fixed_quota <= 1.0
        assert 0.0 <= self.prop_quota <= 1.0
        assert 0.0 <= self.min_stock <= 1.0
        assert 0.0 <= self.target_stock <= 1.0
        assert 0.0 <= self.fine_amount <= self.max_fine
        assert 0.0 <= self.risk_penalty_scale <= self.max_fine
        assert 1.0 <= self.risk_penalty_power <= 5.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        """Serialise the mechanism to a normalised float32 vector in [0, 1]^7.

        Normalisation applied per parameter:

        - ``fixed_quota``, ``prop_quota``, ``min_stock``, ``target_stock``:
          already in [0, 1] — returned as-is.
        - ``fine_amount``:        divided by ``max_fine``  → [0, 1].
        - ``risk_penalty_scale``: divided by ``max_fine``  → [0, 1].
        - ``risk_penalty_power``: ``(value - 1) / 4``      → [0, 1]
          (maps the [1, 5] range to [0, 1]).

        Returns
        -------
        np.ndarray
            Shape ``(7,)``, dtype ``float32``.
            Order: ``[fixed_quota, prop_quota, min_stock, target_stock,
            fine_amount/max_fine, risk_penalty_scale/max_fine,
            (risk_penalty_power - 1) / 4]``.
        """
        return np.array(
            [
                self.fixed_quota,
                self.prop_quota,
                self.min_stock,
                self.target_stock,
                self.fine_amount / self.max_fine,
                self.risk_penalty_scale / self.max_fine,
                (self.risk_penalty_power - 1.0) / 4.0,  # maps [1,5] -> [0,1]
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        """Return the ordered list of parameter names matching :meth:`to_vector`.

        Returns
        -------
        list of str
            ``["fixed_quota", "prop_quota", "min_stock", "target_stock",
            "fine_amount", "risk_penalty_scale", "risk_penalty_power"]``
        """
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
    """Mechanism space for the V1 fishery regulatory mechanism.

    Manages the encoding and decoding of the seven-dimensional V1 mechanism
    vector used by the outer ES optimizer.

    Parameters
    ----------
    use_stochastic_roundting : bool, optional
        Retained for API compatibility (V1 has no integer parameter).
        Default is ``True``.
    max_fine : float, optional
        Upper bound for ``fine_amount`` and ``risk_penalty_scale``.
        Used for normalisation.  Default is ``5.0``.
    optimize_params : list of str or None, optional
        Parameters that ES actively optimises.  If ``None``, all seven
        parameters are optimised.
    default_fixed_quota : float, optional
        Default ``fixed_quota``.  Default is ``1.0``.
    default_prop_quota : float, optional
        Default ``prop_quota``.  Default is ``1.0``.
    default_min_stock : float, optional
        Default ``min_stock``.  Default is ``0.1``.
    default_target_stock : float, optional
        Default ``target_stock``.  Default is ``0.2``.
    default_fine_amount : float, optional
        Default ``fine_amount``.  Default is ``0.5``.
    default_risk_penalty_scale : float, optional
        Default ``risk_penalty_scale``.  Default is ``2.0``.
    default_risk_penalty_power : float, optional
        Default ``risk_penalty_power``.  Default is ``2.0``.

    Attributes
    ----------
    dimension : int
        Number of parameters ES optimises (``len(optimize_params)``).
    full_dimension : int
        Total number of V1 mechanism parameters (always 7).
    """

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
        max_fine: float = 5.0,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 1.0,
        default_prop_quota: float = 1.0,
        default_min_stock: float = 0.1,
        default_target_stock: float = 0.2,
        default_fine_amount: float = 0.5,
        default_risk_penalty_scale: float = 2.0,
        default_risk_penalty_power: float = 2.0,
    ):
        super().__init__()
        self.use_stochastic_roundting = use_stochastic_roundting
        self.max_fine = max_fine

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
        """De-normalise a single parameter from [0, 1] to its native range.

        Parameters
        ----------
        name : str
            Parameter name (must be one of :attr:`ALL_PARAMS`).
        value : float
            Normalised value in [0, 1].

        Returns
        -------
        float
            De-normalised value:
            ``fine_amount``, ``risk_penalty_scale`` → ``value * max_fine``;
            ``risk_penalty_power``                  → ``1 + 4 * value``;
            all others                              → ``value`` unchanged.
        """
        if name in ("fine_amount", "risk_penalty_scale"):
            return value * self.max_fine
        elif name == "risk_penalty_power":
            return 1.0 + 4.0 * value
        else:
            return value

    def _denormalize(self, u: np.ndarray) -> dict:
        """De-normalise the optimised subset of parameters.

        Parameters
        ----------
        u : np.ndarray
            Normalised ES vector of length ``dimension``.

        Returns
        -------
        dict
            Mapping of parameter name → de-normalised value, covering only
            the parameters listed in ``optimize_params``.
        """
        result = {}
        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]))
        return result

    def default(self) -> FisheryMechanism:
        """Return a :class:`FisheryMechanism` with the default parameter values.

        Returns
        -------
        FisheryMechanism
            Mechanism with all seven parameters set to their constructor
            defaults.
        """
        return FisheryMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            prop_quota=self.defaults["prop_quota"],
            min_stock=self.defaults["min_stock"],
            target_stock=self.defaults["target_stock"],
            fine_amount=self.defaults["fine_amount"],
            risk_penalty_scale=self.defaults["risk_penalty_scale"],
            risk_penalty_power=self.defaults["risk_penalty_power"],
            max_fine=self.max_fine,
        )

    def _normalize_param(self, name: str, value: float) -> float:
        """Normalise a single parameter to [0, 1].

        Inverse of :meth:`_denormalize_param`.

        Parameters
        ----------
        name : str
            Parameter name.
        value : float
            Native-range value.

        Returns
        -------
        float
            Normalised value in [0, 1]:
            ``fine_amount``, ``risk_penalty_scale`` → ``value / max_fine``;
            ``risk_penalty_power``                  → ``(value - 1) / 4``;
            all others                              → ``value`` unchanged.
        """
        if name in ("fine_amount", "risk_penalty_scale"):
            return value / self.max_fine
        elif name == "risk_penalty_power":
            return (value - 1.0) / 4.0
        else:
            return value

    def encode(self, m: FisheryMechanism) -> NDArray[np.float32]:
        """Encode a V1 mechanism to the normalised ES search vector.

        Only the parameters in ``optimize_params`` are included.  See
        :meth:`_normalize_param` for the normalisation applied to each field.

        Parameters
        ----------
        m : FisheryMechanism
            V1 mechanism instance to encode.

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
        """Decode a normalised ES vector into a :class:`FisheryMechanism` (V1).

        The input vector is clipped to ``[0, 1]``, de-normalised via
        :meth:`_denormalize_param`, merged with the fixed defaults, and
        finally clipped via :meth:`clip`.

        Parameters
        ----------
        x : NDArray[np.float32]
            Shape ``(dimension,)``, values expected in ``[0, 1]``.

        Returns
        -------
        FisheryMechanism
            Valid, clipped V1 mechanism.
        """
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
            max_fine=self.max_fine,
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
            ``fixed_quota, prop_quota, min_stock, target_stock`` ∈ [0, 1];
            ``fine_amount``, ``risk_penalty_scale`` ∈ [0, max_fine];
            ``risk_penalty_power`` ∈ [1, 5].
        """
        return FisheryMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0, 1)),
            prop_quota=float(np.clip(m.prop_quota, 0, 1)),
            min_stock=float(np.clip(m.min_stock, 0, 1)),
            target_stock=float(np.clip(m.target_stock, 0, 1)),
            fine_amount=float(np.clip(m.fine_amount, 0, self.max_fine)),
            # CHANGED
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0, self.max_fine)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
            max_fine=self.max_fine,
        )

    def from_dict(self, cfg: dict) -> FisheryMechanism:
        """Build a V1 :class:`FisheryMechanism` from a plain parameter dictionary.

        Provides backward compatibility: if ``risk_penalty_scale`` or
        ``risk_penalty_power`` are absent, they are filled in from
        ``self.defaults``.

        Parameters
        ----------
        cfg : dict
            Mapping of parameter name → value.  Must contain at minimum:
            ``fixed_quota``, ``prop_quota``, ``min_stock``, ``target_stock``,
            ``fine_amount``.  The risk-penalty fields are optional.

        Returns
        -------
        FisheryMechanism
            V1 mechanism constructed from *cfg* with ``max_fine`` injected
            from ``self``.
        """
        # CHANGED: add backward-compatible defaults for new params
        if "risk_penalty_scale" not in cfg:
            cfg = {**cfg, "risk_penalty_scale": self.defaults["risk_penalty_scale"]}
        if "risk_penalty_power" not in cfg:
            cfg = {**cfg, "risk_penalty_power": self.defaults["risk_penalty_power"]}

        return FisheryMechanism(**cfg, max_fine=self.max_fine)