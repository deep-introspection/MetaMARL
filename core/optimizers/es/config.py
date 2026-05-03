from typing import Optional, Self

from core.annotations import override
from core.optimizers.config import OptimizerConfig
from core.optimizers.es.optimizer import ESOptimizer


class ESConfig(OptimizerConfig):
    """Configuration for the separable Natural Evolution Strategies (sNES) outer optimizer.

    ``ESConfig`` configures a diagonal-covariance ES that operates in logit
    space to naturally respect the ``[0, 1]`` bounds of the mechanism parameter
    space.  Antithetic (mirrored) sampling is used by default to reduce gradient
    variance at zero extra cost (Brockhoff et al., 2010).

    The search distribution is a factorised Gaussian
    :math:`\\mathcal{N}(\\mu, \\sigma^2 I)` maintained in logit space.
    The mean update follows an NES-style natural gradient:

    .. math::

        g = \\frac{1}{N} \\sum_{i=1}^{N} \\hat{f}_i \\, \\tilde{\\varepsilon}_i

        \\mu_{\\text{logit}} \\leftarrow \\mu_{\\text{logit}} + \\alpha_\\mu \\, g

    where :math:`\\hat{f}_i` are whitened fitness scores,
    :math:`\\tilde{\\varepsilon}_i = (z_i^{\\text{logit}} - \\mu_{\\text{logit}}) / \\sigma`
    are approximate noise vectors, and :math:`\\alpha_\\mu` is ``mean_lr``.
    The step-size :math:`\\sigma` is adapted with a 1/5-success rule
    (Rechenberg, 1973):

    .. math::

        \\sigma \\leftarrow \\text{clip}\\!\\left(
            \\sigma \\cdot e^{\\alpha_\\sigma (p_s - 0.2)},\\,
            \\sigma_{\\min},\\, \\sigma_{\\max}
        \\right)

    where :math:`p_s = \\mathbb{E}[\\hat{f}_i > 0]` is the empirical success
    rate and :math:`\\alpha_\\sigma` is ``sigma_lr``.

    References
    ----------
    Wierstra, D. et al. (2014) "Natural Evolution Strategies"
    *Journal of Machine Learning Research*, 15, pp. 949-980.

    Rechenberg, I. (1973) *Evolutionsstrategie*.  Frommann-Holzboog.

    Brockhoff, D. et al. (2010) "Mirrored Sampling and Sequential Selection
    for Evolution Strategies" *PPSN XI*, LNCS 6238.

    Attributes
    ----------
    dimension : int or None
        Dimensionality of the mechanism parameter vector.  Set automatically
        from ``MechanismSpace.dimension`` by :class:`~core.optimizers.bilevel.BilevelConfig`.
    sigma : float
        Initial standard deviation of the isotropic Gaussian search
        distribution in logit space.  Defaults to ``0.15``.
    mean_lr : float
        Learning rate :math:`\\alpha_\\mu` for the mean parameter update.
        Defaults to ``0.1``.
    sigma_lr : float
        Learning rate :math:`\\alpha_\\sigma` for the step-size adaptation.
        Defaults to ``0.05``.
    min_sigma : float
        Lower bound on :math:`\\sigma` to prevent premature convergence.
        Defaults to ``1e-3``.
    max_sigma : float
        Upper bound on :math:`\\sigma` to prevent excessively large steps.
        Defaults to ``0.5``.
    break_symmetry : bool
        When ``True``, the last sample of an antithetic pair is replaced
        by an independent draw, breaking strict mirror symmetry.  Useful
        when the population size is fixed and cannot be made even.
        Defaults to ``False``.
    convergence_eps : float
        Minimum fitness improvement required to reset the patience counter.
        Defaults to ``1e-4``.
    convergence_patience : int
        Number of consecutive generations without improvement before the
        optimizer reports convergence.  Defaults to ``10``.
    initial_mean : list[float] or None
        Optional initial mean vector in ``[0, 1]^d``.  If ``None``, the mean
        is initialised to ``0.5`` in every dimension (centre of the unit
        hypercube).
    """

    def __init__(self, opt_class=None):
        """Initialise ``ESConfig`` with default hyperparameters.

        Parameters
        ----------
        opt_class : type[ESOptimizer], optional
            Concrete optimizer class to instantiate via :meth:`build_optimizer`.
            Defaults to :class:`~core.optimizers.es.optimizer.ESOptimizer`.
        """
        super().__init__(opt_class=opt_class or ESOptimizer)

        # Add default or from default
        # ES training hyperparameters
        self.dimension: int = None
        self.sigma: int = 0.15
        self.mean_lr: float = 0.1
        self.sigma_lr: float = 0.05
        self.min_sigma: float = 1e-3
        self.max_sigma: float = 0.5
        self.break_symmetry: bool = False

        self.convergence_eps: float = 1e-4
        self.convergence_patience: int = 10
        self.initial_mean: Optional[list[float]] = None

    @override(OptimizerConfig)
    def training(
        self,
        *,
        sigma: Optional[int] = None,
        mean_lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        generation: Optional[int] = None,
        break_symmetry: Optional[bool] = None,
        convergence_eps: Optional[float] = None,
        convergence_patience: Optional[int] = None,
        initial_mean: Optional[list[float]] = None,
        **kwargs,
    ) -> Self:
        """Set ES-specific training hyperparameters.

        All parameters are optional; unset parameters leave the corresponding
        attribute unchanged.  Unknown keyword arguments are forwarded to
        :meth:`~core.optimizers.config.OptimizerConfig.training` (e.g.
        ``seed``).

        Parameters
        ----------
        sigma : float, optional
            Initial standard deviation of the isotropic Gaussian search
            distribution in logit space.  Larger values increase exploration
            at the cost of noisier gradient estimates.
        mean_lr : float, optional
            Learning rate :math:`\\alpha_\\mu` for the mean parameter update.
            Higher values shift the distribution centre more aggressively
            toward high-fitness regions.
        sigma_lr : float, optional
            Learning rate :math:`\\alpha_\\sigma` for the 1/5-success step-size
            adaptation rule.  Higher values make the step size respond faster
            to the current success rate.
        min_sigma : float, optional
            Lower bound on :math:`\\sigma`.  Prevents premature convergence by
            ensuring a minimum level of exploration is always maintained.
        max_sigma : float, optional
            Upper bound on :math:`\\sigma`.  Prevents excessively large
            exploration steps that could destabilise training.
        generation : int, optional
            Starting generation counter (useful when resuming a run from a
            checkpoint).
        break_symmetry : bool, optional
            When ``True``, replace the last antithetic mirror sample with an
            independent draw, breaking strict antithetic symmetry.  Required
            when the population size is odd and cannot be changed.
        convergence_eps : float, optional
            Minimum absolute fitness improvement across generations required
            to reset the patience counter.
        convergence_patience : int, optional
            Number of consecutive generations without a ``convergence_eps``
            improvement before the optimizer reports convergence and triggers
            early stopping.
        initial_mean : list[float], optional
            Initial mean vector in ``[0, 1]^d``.  Must have length equal to
            ``dimension``.  Defaults to an all-``0.5`` vector.
        **kwargs : Any
            Additional keyword arguments forwarded to
            :meth:`~core.optimizers.config.OptimizerConfig.training`.

        Returns
        -------
        Self
            This ``ESConfig`` instance (for method chaining).
        """
        super().training(**kwargs)

        if sigma is not None:
            self.sigma = sigma
        if mean_lr is not None:
            self.mean_lr = mean_lr
        if sigma_lr is not None:
            self.sigma_lr = sigma_lr
        if min_sigma is not None:
            self.min_sigma = min_sigma
        if max_sigma is not None:
            self.max_sigma = max_sigma
        if generation is not None:
            self.generation = generation
        if break_symmetry is not None:
            self.break_symmetry = break_symmetry
        if convergence_eps is not None:
            self.convergence_eps = convergence_eps
        if convergence_patience is not None:
            self.convergence_patience = convergence_patience
        if initial_mean is not None:
            self.initial_mean = initial_mean

        return self

    # @override(OptimizerConfig)
    # def evaluation(
    #     self, *, evaluation_best_fitness, evaluation_best_candidate, **kwargs
    # ) -> Self:
    #     raise NotImplementedError

    # # TODO this is where the random seed goes
    # # TODO do we put rng here ?
    # @override(OptimizerConfig)
    # def fault_tolerance(self, *, rng: Optional[float] = None, **kwargs) -> Self:
    #     return super().fault_tolerance()
