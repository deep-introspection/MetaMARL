import logging
from abc import abstractmethod
from typing import Optional, SupportsFloat

import ray

from core.annotations import override
from core.envs.base import BaseEnv
from core.types import OptimizerID
from core.world.base import World
from core.world.context import MechanismContext, MechanismStatus

logger = logging.getLogger(__name__)


class RegulatedEnv(BaseEnv):
    """Inner-loop environment that operates under an externally imposed mechanism.

    Extends :class:`~core.envs.base.BaseEnv` with mechanism-fetching logic and
    a penalised reward override.  At the start of each episode :meth:`_pre_reset`
    attempts to fetch a freshly published :class:`~core.world.context.MechanismContext`
    from the shared ``World`` actor.  If the mechanism queue is empty the
    previously active mechanism is reused; if no mechanism has ever been set a
    fallback default is instantiated from :attr:`m_space`.

    The penalised reward is

    .. math::

        r(t) = r_{\\text{raw}}(t) - \\lambda \\cdot v(r_{\\text{raw}}(t))

    where :math:`\\lambda` is provided by :meth:`penalty` and :math:`v` by
    :meth:`violation_signal`.

    Parameters
    ----------
    world : World
        Ray remote actor serving as the shared runtime state container.
    opt_id : OptimizerID or None, optional
        Identifier of the outer-loop optimizer that owns this environment.
    **kwargs
        Forwarded to :class:`~core.envs.base.BaseEnv`.
    """

    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        **kwargs,
    ) -> None:
        super().__init__(world=world, opt_id=opt_id, **kwargs)

    @override(BaseEnv)
    def _pre_reset(self):
        """Fetch the next published mechanism from the World before episode reset.

        Resolution order:

        1. Try :pymeth:`World.try_get_mechanism` for a non-blocking poll.  If a
           ``published`` mechanism exists it is atomically marked ``assigned``
           and stored in :attr:`m_ctx` / :attr:`m`.
        2. If no mechanism was obtained *and* none is currently active, fall back
           to a blocking :pymeth:`World.get_mechanism` call.
        3. If even that fails (``RuntimeError``), instantiate a default mechanism
           from :attr:`m_space` with ``index=-1`` as a last resort.
        """
        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        try:
            new_ctx = ray.get(self.world.try_get_mechanism.remote())
        except Exception:
            new_ctx = None

        if new_ctx is not None:
            self.m_ctx = new_ctx
            self.m = self.m_ctx.mechanism
        # Fallback: if no mechanism yet, we must get one

        # TODO better fallback mechanism since this leads to silent bugs!
        if self.m_ctx is None or self.m is None:
            try:
                self.m_ctx = ray.get(self.world.get_mechanism.remote())
                self.m = self.m_ctx.mechanism
            except RuntimeError:
                # fallback baseline mechanism (must exist locally)
                self.m_ctx = MechanismContext(
                    index=-1,
                    env_id=None,
                    mechanism=self.m_space.default(),
                    status=MechanismStatus.init,
                    job=None,
                    metrics=None,
                )
                self.m = self.m_ctx.mechanism

    @abstractmethod
    def violation_signal(self, reward: Optional[SupportsFloat] = None) -> float:
        """Return the regulatory violation signal for the current timestep.

        Concrete implementations must inspect the agent's action and/or reward
        against the active mechanism constraints (quota, ban period, stock
        threshold) and return a non-negative scalar indicating the degree of
        violation.

        Parameters
        ----------
        reward : SupportsFloat or None, optional
            Raw reward from :meth:`_step`, provided as a convenience signal
            for implementations that measure violation relative to utility.

        Returns
        -------
        float
            Violation signal :math:`v \\geq 0`.  Zero means no violation.
        """
        raise NotImplementedError

    @abstractmethod
    def penalty(self, reward: Optional[SupportsFloat] = None) -> float:
        """Return the penalty scale :math:`\\lambda` for the current mechanism.

        Concrete implementations should extract the relevant fine or penalty
        parameter from the active mechanism :attr:`m` and return it as a scalar
        multiplier applied to :meth:`violation_signal`.

        Parameters
        ----------
        reward : SupportsFloat or None, optional
            Raw reward from :meth:`_step`.  Provided for implementations that
            compute adaptive penalties relative to utility magnitude.

        Returns
        -------
        float
            Penalty scale :math:`\\lambda \\geq 0`.
        """
        raise NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Return the mechanism-penalised reward.

        Applies the regulatory penalty to the raw step reward:

        .. math::

            r_{\\text{penalised}} = r_{\\text{raw}} - \\lambda(r) \\cdot v(r)

        where :math:`\\lambda` is returned by :meth:`penalty` and :math:`v` by
        :meth:`violation_signal`.

        Parameters
        ----------
        reward : SupportsFloat
            Raw scalar reward from :meth:`_step`.

        Returns
        -------
        SupportsFloat
            Penalised reward.
        """
        return reward - self.penalty(reward) * self.violation_signal(reward)
