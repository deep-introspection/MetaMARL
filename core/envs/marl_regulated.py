import logging
from abc import abstractmethod
from typing import Any, SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.types import OptimizerID
from core.world.base import World
from core.world.context import EnvStepContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type

# TODO remove inheritance from regulatedEnv completley
# class SingleAgentBaseEnv(gym.Env): ...
# class MultiAgentBaseEnv(MultiAgentEnv): ...

# class RegulatedEnv(SingleAgentBaseEnv): ...
# class MultiAgentRegulatedEnv(MultiAgentBaseEnv): ...


class MultiAgentRegulatedEnv(RegulatedEnv, MultiAgentEnv):
    """Multi-agent inner-loop environment for regulated fishery agents.

    Combines :class:`~core.envs.regulated.RegulatedEnv` (mechanism-fetching and
    penalty application) with Ray RLlib's :class:`~ray.rllib.env.multi_agent_env.MultiAgentEnv`
    interface.  Each of the ``N`` fishing agents acts independently and receives
    a per-agent observation that includes both ecological state and the currently
    active regulatory mechanism vector.

    Concretely, the reward for agent ``i`` at time ``t`` is

    .. math::

        r_i(t) = u_i(a_i, S_t) - \\lambda(M) \\cdot v_i(a_i, S_t, M)

    where :math:`u_i` is the intrinsic fishing utility, :math:`\\lambda` the
    mechanism-defined penalty scale, and :math:`v_i` the violation signal.

    Subclasses must implement:
    :meth:`_step`, :meth:`_reset`, :meth:`transition_kernel`,
    :meth:`intrinsic_utility`, :meth:`violation_signal`, :meth:`penalty`,
    :meth:`_observation`, :meth:`_is_truncated`, and :meth:`aggregate_rewards`.

    Parameters
    ----------
    world : World
        Ray remote actor serving as the shared runtime state container.
    opt_id : OptimizerID
        Identifier of the outer-loop optimizer that owns this environment.
    agents : list[AgentID]
        Ordered list of agent identifiers (e.g. ``["agent_0", "agent_1", ...]``).
    **kwargs
        Additional keyword arguments forwarded to :class:`~core.envs.regulated.RegulatedEnv`.
        Recognised keys include ``action_spaces`` and ``observation_spaces``
        (both ``dict[AgentID, Space]``).
    """

    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID,
        agents: list[AgentID],
        **kwargs,
    ):
        super().__init__(world=world, opt_id=opt_id, **kwargs)
        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.observation_spaces = kwargs.get("observation_spaces", {})

        # TODO move this to baseenv later
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

    @abstractmethod
    def _step(
        self, action_dict: MultiAgentDict = None
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Run one timestep of the multi-agent environment dynamics.

        Concrete implementations must apply each agent's action, advance the
        ecological state via :meth:`transition_kernel`, compute per-agent
        rewards (combining :meth:`intrinsic_utility` and penalty terms), and
        build per-agent observations and termination flags.

        Parameters
        ----------
        action_dict : MultiAgentDict, optional
            Mapping ``{agent_id: action}`` for all active agents.

        Returns
        -------
        obs : MultiAgentDict
            Per-agent next observations.
        rewards : MultiAgentDict
            Per-agent scalar rewards.
        terminated : MultiAgentDict
            Per-agent (and ``"__all__"``) termination flags.
        truncated : MultiAgentDict
            Per-agent (and ``"__all__"``) truncation flags.
        infos : MultiAgentDict
            Per-agent auxiliary diagnostic information.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> MultiAgentDict:
        """Reset the environment and return per-agent initial observations.

        Returns
        -------
        MultiAgentDict
            Mapping ``{agent_id: initial_obs}`` for all agents.
        """
        raise NotImplementedError

    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset the multi-agent environment for a new episode.

        Calls shared bookkeeping in :meth:`~core.envs.base.BaseEnv._base_reset`
        (mechanism fetch, RNG init, timestep reset) then delegates to
        :meth:`_reset` for environment-specific initialisation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the random number generator.
        options : dict or None, optional
            Additional reset options (currently unused).

        Returns
        -------
        obs : MultiAgentDict
            Per-agent initial observations.
        infos : MultiAgentDict
            Per-agent empty info dictionaries.
        """
        self._base_reset(seed=seed)
        obs = self._reset()
        infos = {agent_id: {} for agent_id in self.agents}
        return obs, infos

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Advance all agents by one timestep and publish context to the World.

        Pre-processes actions via :meth:`~core.envs.base.BaseEnv.action`,
        delegates to :meth:`_step`, publishes an
        :class:`~core.world.context.EnvStepContext` to the ``World``, and
        increments the internal timestep counter.

        Parameters
        ----------
        action_dict : MultiAgentDict
            Mapping ``{agent_id: action}`` for all active agents.

        Returns
        -------
        obs : MultiAgentDict
            Per-agent next observations.
        rewards : MultiAgentDict
            Per-agent scalar rewards.
        terminated : MultiAgentDict
            Per-agent (and ``"__all__"``) termination flags.
        truncated : MultiAgentDict
            Per-agent (and ``"__all__"``) truncation flags.
        infos : MultiAgentDict
            Per-agent auxiliary diagnostic information.
        """
        actions = self.action(action_dict)
        obs, rewards, terminated, truncated, infos = self._step(actions)

        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index if self.m_ctx else None,
                observation=obs,
                observation_map=self.obs_map,
                reward=rewards,
                action=actions,
                info=infos,
            )
        )

        self._t += 1
        return obs, rewards, terminated, truncated, infos

    @abstractmethod
    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
        **kwargs,
    ) -> dict[str, float]:
        """Compute the next ecological state given current state and joint actions.

        Implements the stochastic transition :math:`S_{t+1} = T(S_t, A_t)`.
        In the fishery setting this typically integrates Lotka-Volterra
        predator-prey dynamics after aggregating agent harvests.

        Parameters
        ----------
        A_t : MultiAgentDict
            Joint action mapping ``{agent_id: action}`` at time ``t``.
        S_t : dict[str, MultiAgentDict]
            Current state of the world (e.g. fish stock, algae biomass).
        **kwargs
            Additional keyword arguments forwarded by concrete subclasses.

        Returns
        -------
        dict[str, float]
            Next-state variables (e.g. ``{"fish": float, "algae": float}``).
        """
        ...

    @abstractmethod
    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Compute agent ``i``'s intrinsic (pre-penalty) utility.

        Implements :math:`u_i = U(a_i, S_t)`, typically a profit signal based
        on the quantity harvested and the current fish stock.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the agent whose utility is being computed.
        action : ActType
            The action taken by agent ``i`` at time ``t``.
        S_t : dict[str, MultiAgentDict]
            Current world state at time ``t``.

        Returns
        -------
        SupportsFloat
            Scalar intrinsic utility :math:`u_i`.
        """
        ...

    @abstractmethod
    @override(RegulatedEnv)
    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Compute the regulatory violation signal for agent ``i``.

        Implements :math:`v_i = V(a_i, S_t, M)`.  The signal is typically
        non-zero when the agent violates a mechanism constraint (e.g. exceeds
        the quota or harvests during a ban period) and zero otherwise.  The
        combined penalised reward is

        .. math::

            r_i = u_i - \\lambda(M) \\cdot v_i

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the agent being evaluated.
        u_i : SupportsFloat
            Intrinsic utility of agent ``i`` (output of :meth:`intrinsic_utility`).
        S_t : dict[str, MultiAgentDict]
            Current world state at time ``t``.

        Returns
        -------
        SupportsFloat
            Scalar violation signal :math:`v_i \\geq 0`.
        """
        ...

    @abstractmethod
    @override(RegulatedEnv)
    def penalty(self) -> np.ndarray:
        """Return the penalty scale vector derived from the active mechanism.

        Implements :math:`\\lambda = \\lambda(M)`.  The penalty scale is
        typically a scalar or per-agent array extracted from the mechanism
        parameters (e.g. fine amount, risk penalty scale).

        Returns
        -------
        np.ndarray
            Penalty scale :math:`\\lambda \\geq 0`, shape ``(N,)`` or scalar.
        """
        ...

    @abstractmethod
    def _observation(
        self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]
    ) -> ObsType:
        """Compute the base observation for agent ``i`` from world state.

        Implements :math:`o_i = O_i(S_t)`, returning the ecological and
        agent-specific features *without* the mechanism vector appended.
        The full observation (including mechanism parameters) is assembled by
        :meth:`observation`.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the agent being observed.
        S_t : dict[str, MultiAgentDict]
            Current world state at time ``t``.

        Returns
        -------
        ObsType
            Base observation array for agent ``i``.
        """
        ...

    # TODO Restrict Any Type
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> Any:
        """Construct the full observation for agent ``i`` including mechanism parameters.

        Augments the base ecological observation from :meth:`_observation` with
        the current mechanism vector, so agents can condition their policy on the
        active regulatory parameters:

        .. math::

            \\tilde{o}_i = \\bigl[O_i(S_t) \\;\\|\\; \\theta\\bigr]

        where :math:`\\theta = M.\\text{to\\_vector}()` is the flattened mechanism.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the agent being observed.
        S_t : dict[str, MultiAgentDict]
            Current world state at time ``t``.

        Returns
        -------
        Any
            Concatenated observation array of shape
            ``(|O_i| + |\\theta|,)``.
        """
        # TODO may wanna normalize base_obs later
        base_obs = self._observation(agent_id=agent_id, S_t=S_t)
        theta = self.m.to_vector()
        return np.concatenate([base_obs, theta], axis=0)

    @abstractmethod
    def _is_truncated(self) -> bool:
        """Return ``True`` if the current episode should be truncated.

        Concrete implementations should check horizon limits, external signals,
        or any time-based stopping criterion.  This is separate from termination
        (which is triggered by reaching an absorbing state).

        Returns
        -------
        bool
            ``True`` if the episode should end due to truncation.
        """
        ...

    @abstractmethod
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        """Aggregate or transform per-agent raw rewards before returning them.

        Called after individual rewards are computed.  Concrete implementations
        may normalise, clip, or redistribute rewards across agents (e.g. social
        welfare pooling).

        Parameters
        ----------
        rewards : MultiAgentDict
            Mapping ``{agent_id: raw_reward}`` produced by the step dynamics.

        Returns
        -------
        MultiAgentDict
            Mapping ``{agent_id: final_reward}`` after aggregation.
        """
        ...

    # @override(BaseEnv)
    # def reward(self, agent_id: AgentID, action: ActType) -> SupportsFloat:
    #     o_i = self.observation(agent_id=agent_id, S_t=self.S_t)
    #     u_i = self.intrinsic_utility(agent_id=agent_id, action=action, observation=o_i)
    #     return u_i - self.penalty() * self.violation_signal(
    #         agent_id=agent_id, reward=u_i, observation=o_i
    #     )