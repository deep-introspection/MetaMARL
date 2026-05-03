from abc import abstractmethod
from typing import Any, Optional, SupportsFloat
import uuid

import numpy as np
import ray
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.world.base import World
from core.world.context import Context, ContextSchema, EnvStepContext, MechanismContext


class BaseEnv(Env):
    """Base environment that directly interacts with the World.

    Abstract base class for all fishery environments in the bilevel framework.
    Wraps a Gymnasium ``Env`` and wires it to a Ray-remote ``World`` actor for
    context publishing, mechanism retrieval, and cross-optimizer communication.

    Concrete subclasses must implement :meth:`_step`, :meth:`_reset`, and
    :meth:`_pre_reset`.  The public :meth:`step` / :meth:`reset` methods add
    observation/reward post-processing and publish :class:`~core.world.context.EnvStepContext`
    objects to the shared ``World``.

    Parameters
    ----------
    world : World
        Ray remote actor that serves as the shared runtime state container.
    opt_id : OptimizerID or None, optional
        Identifier of the outer-loop optimizer that owns this environment.
        Can be set later via :meth:`set_opt_id`.
    horizon : int or None, optional
        Maximum number of steps per episode.  ``None`` means no hard limit.
    mechanism_space : MechanismSpace or type, optional
        Geometry over the mechanism manifold.  Accepts either an instance or a
        class (in which case it is instantiated with no arguments).
    **kwargs
        Forwarded to ``gymnasium.Env.__init__``.
    """

    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        horizon: Optional[int] = None,
        mechanism_space: MechanismSpace = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.world = world
        self._opt_id = opt_id
        self.horizon = horizon
        self._t = 0
        # mechanism_space can be a class or an instance
        if isinstance(mechanism_space, type):
            self.m_space: MechanismSpace = mechanism_space()
        else:
            self.m_space: MechanismSpace = mechanism_space
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self.env_id = uuid.uuid4().hex[:8]

        # observation map
        self.obs_map: dict[int, str] = None

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        """Set or update the optimizer ID associated with this environment.

        Parameters
        ----------
        opt_id : OptimizerID
            Identifier of the outer-loop optimizer that owns this environment.
        """
        self._opt_id = opt_id

    # private methods
    def _publish(self, payload: ContextSchema):
        """Publish a context payload to the shared World actor.

        Wraps the payload in a :class:`~core.world.context.Context` carrying
        the current optimizer ID, timestep, and environment class name, then
        calls :pymeth:`World.append_context` on the Ray remote actor.

        Parameters
        ----------
        payload : ContextSchema
            The typed payload to publish (e.g. :class:`~core.world.context.EnvStepContext`
            or :class:`~core.world.context.MechanismContext`).
        """
        ctx = Context(
            id=None,
            opt_id=self._opt_id,
            step=self._t,
            env=self.__class__.__name__,
            payload=payload,
        )
        ray.get(self.world.append_context.remote(ctx))

    @abstractmethod
    def _step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Concrete implementations must apply the given action to the internal
        state and return the standard Gymnasium 5-tuple.  The action has already
        been transformed by :meth:`action` before this method is called.

        Parameters
        ----------
        action : ActType, optional
            The (already post-processed) action selected by the agent.

        Returns
        -------
        obs : ObsType
            The raw next observation (before :meth:`observation` post-processing).
        reward : SupportsFloat
            The raw scalar reward (before :meth:`reward` post-processing).
        terminated : bool
            ``True`` if the episode ended due to a terminal condition.
        truncated : bool
            ``True`` if the episode ended due to horizon/time-limit truncation.
        info : dict[str, Any]
            Auxiliary diagnostic information.
        """
        raise NotImplementedError

    def _base_reset(self, *, seed=None):
        """Perform shared reset bookkeeping common to all subclasses.

        Initialises the random number generator, resets the internal timestep
        counter to zero, and delegates to :meth:`_pre_reset` for any
        subclass-specific pre-reset logic (e.g. mechanism fetching).

        Parameters
        ----------
        seed : int or None, optional
            Seed for :class:`numpy.random.Generator`.  ``None`` produces a
            randomly seeded generator.
        """
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._pre_reset()

    @abstractmethod
    def _pre_reset(self) -> None:
        """Execute subclass-specific logic that must run before :meth:`_reset`.

        Concrete implementations should use this hook to fetch a new mechanism
        from the ``World``, update internal episode counters, or perform any
        other pre-episode initialisation.  Called automatically by
        :meth:`_base_reset`.
        """
        pass

    @abstractmethod
    def _reset(self):
        """Reset the environment to its initial state and return the first observation.

        Called by :meth:`reset` after :meth:`_base_reset` has run.  Concrete
        implementations must reinitialise all episode-level state (e.g. fish
        stock, agent inventories) and return the initial observation.

        Returns
        -------
        ObsType
            The initial observation for the new episode.
        """
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance the environment by one timestep.

        Applies :meth:`action` to pre-process the raw action, delegates to
        :meth:`_step` for dynamics, then applies :meth:`observation` and
        :meth:`reward` post-processing.  Publishes an
        :class:`~core.world.context.EnvStepContext` to the ``World`` and
        increments the internal timestep counter.

        Parameters
        ----------
        action : ActType, optional
            Raw action from the agent.

        Returns
        -------
        obs : ObsType
            Post-processed observation.
        reward : SupportsFloat
            Post-processed scalar reward.
        terminated : bool
            ``True`` if a terminal state was reached.
        truncated : bool
            ``True`` if the episode was cut short by the horizon.
        info : dict[str, Any]
            Auxiliary diagnostic information from :meth:`_step`.
        """
        raw_obs, raw_reward, terminated, truncated, info = self._step(
            self.action(action)
        )

        obs = self.observation(raw_obs)
        reward = self.reward(raw_reward)

        m_idx = self.m_ctx.index if self.m_ctx is not None else None

        # Publish env context to World
        self._publish(
            EnvStepContext(
                mechanism=m_idx,  # TODO link with mechanismID rather than mechanism
                observation=obs,
                observation_map=self.obs_map,
                reward=reward,
                action=action,
                info=info,
            )
        )
        self._t += 1
        return obs, reward, terminated, truncated, info

    @override(Env)
    def reset(self, *, seed=None, options=None):
        """Reset the environment and publish the initial observation to the World.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the random number generator.
        options : dict or None, optional
            Additional reset options (currently unused).

        Returns
        -------
        ObsType
            The initial post-processed observation.
        """
        self._base_reset(seed=seed)
        obs = self._reset()
        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index,
                observation=obs,
                observation_map=self.obs_map,
                reward=0.0,
                action=None,
                info={},
            )
        )
        return obs

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Return a (optionally modified) observation.

        Identity by default.  Subclasses may override to normalise, stack, or
        augment observations before they are returned to the agent.

        Parameters
        ----------
        observation : ObsType
            The raw observation produced by :meth:`_step` or :meth:`_reset`.

        Returns
        -------
        WrapperObsType
            The (potentially transformed) observation.
        """
        return observation

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Return a (optionally modified) scalar reward.

        Identity by default.  Subclasses (e.g. :class:`~core.envs.regulated.RegulatedEnv`)
        override this to apply penalty terms from the active mechanism.

        Parameters
        ----------
        reward : SupportsFloat
            The raw reward returned by :meth:`_step`.

        Returns
        -------
        SupportsFloat
            The (potentially transformed) reward signal.
        """
        return reward

    def action(self, action: WrapperActType) -> ActType:
        """Return a (optionally modified) action before :meth:`_step` is called.

        Identity by default.  Subclasses (e.g.
        :class:`~core.envs.regulator.RegulatorEnv`) override this to decode raw
        ES parameter vectors into typed :class:`~core.mechanism.base.Mechanism`
        objects.

        Parameters
        ----------
        action : WrapperActType
            The raw action provided by the agent or outer-loop optimiser.

        Returns
        -------
        ActType
            The (potentially transformed) action forwarded to :meth:`_step`.
        """
        return action
