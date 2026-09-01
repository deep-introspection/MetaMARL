"""Multi-agent environment regulated by an explicit mechanism.

``MultiAgentRegulatedEnv`` is the RLlib-facing environment of the inner
optimization level. A concrete benchmark subclasses it and declares its
dynamics through the decorators in :mod:`core.envs.hooks`; the base class owns
the step lifecycle and the mechanism dispatch::

    policy output
        -> normalize action               (sigmoid squashing)
        -> benchmark @action hook          (optional)
        -> mechanism.action                M^A
        -> benchmark @reward hook          (intrinsic / base reward)
        -> benchmark @transition hook      S_{t+1} = T(S_t, A_t)
        -> mechanism.reward                M^R
        -> benchmark @observation hook     o_i = O_i(S_{t+1})
        -> append mechanism.to_vector()    theta visible to agents
        -> mechanism.observation           M^O
        -> publish EnvStepContext

The mechanism in force is fetched from the ``World`` actor at ``reset`` by
``mechanism_id`` (the candidate index published by the outer optimizer). Until
one is published the environment falls back to the ``mechanism`` template it
was constructed with and returns zero rewards, so RLlib's environment checks
can run before training starts.
"""

import logging
from typing import Any, ClassVar, Optional, SupportsFloat

import numpy as np
import ray
from gymnasium import spaces
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import AgentID, MultiAgentDict, OptimizerID
from core.utils import sigmoid
from core.world.base import World
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

logger = logging.getLogger(__name__)

HOOK_NAMES = ("reset", "action", "reward", "observation", "transition")


class MultiAgentRegulatedEnv(MultiAgentEnv):
    """Base class for mechanism-regulated multi-agent benchmarks.

    Parameters
    ----------
    world : World
        Ray actor handle of the shared blackboard.
    mechanism_id : int
        Index of the candidate mechanism this env instance trains against.
    mechanism : Mechanism, optional
        Template used until a candidate is published (also defines the
        observation size through ``to_vector``).
    agents : list[AgentID]
        Agent identifiers; also fixes the peer ordering seen by mechanisms.
    horizon : int, optional
        Episode length; ``truncated["__all__"]`` is raised at ``horizon``.
    seed, policy_seed : int, optional
        Environment RNG seed and the seed of the policy trained on this env.
    mode : {"train", "eval"}
        Lifecycle status requested from the World when fetching a mechanism.
    action_temperature : float
        Temperature of the sigmoid squashing raw policy outputs to ``[0, 1]``.
    action_spaces, observation_spaces : dict, optional
        Per-agent gymnasium spaces (forwarded by the RLlib env creator).
    """

    _reset: ClassVar[str | None] = None
    _action: ClassVar[str | None] = None
    _reward: ClassVar[str | None] = None
    _observation: ClassVar[str | None] = None
    _transition: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        world: World,
        mechanism_id: int,
        agents: list[AgentID],
        opt_id: Optional[OptimizerID] = None,
        horizon: Optional[int] = None,
        mechanism: Optional[Mechanism] = None,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        action_temperature: float = 4.0,
        action_spaces: Optional[dict] = None,
        observation_spaces: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.world = world
        self._opt_id = opt_id

        # training
        self.horizon = horizon
        self._t = 0
        self.env_id = None

        # seeding
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self.mode = MechanismStatus(mode)

        # mechanism
        self.mechanism_id = mechanism_id
        self.mechanism_template: Optional[Mechanism] = mechanism
        self.m_ctx: Optional[MechanismContext] = None
        self.m: Optional[Mechanism] = None
        self._using_default_mechanism = True

        if action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        self.action_temperature = float(action_temperature)

        # multi-agent bookkeeping
        self.agents = list(agents)
        self.possible_agents = list(self.agents)
        self.observation_spaces = observation_spaces or {}
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_spaces = action_spaces or {}
        self.action_space = spaces.Dict(self.action_spaces)
        self.obs_map: Optional[list[str]] = None
        self._infos: MultiAgentDict = {agent_id: {} for agent_id in self.agents}

        # benchmark state and last delivered actions (exposed to mechanisms)
        self.S_t: dict[str, Any] = {}
        self.previous_actions: MultiAgentDict = self._zero_actions()

    def __init_subclass__(cls, **kwargs):
        """Discover the benchmark hooks declared with :mod:`core.envs.hooks`.

        Exactly one method per hook type may be declared on a class; a second
        one raises rather than silently overriding the first.
        """
        super().__init_subclass__(**kwargs)
        for hook in HOOK_NAMES:
            found = [
                name for name, f in cls.__dict__.items() if getattr(f, hook, False)
            ]
            if len(found) > 1:
                raise TypeError(
                    f"{cls.__name__} declares several @{hook} hooks: {found}. "
                    "Declare exactly one."
                )
            if found:
                setattr(cls, f"_{hook}", found[0])

    # --- accessors ---------------------------------------------------------------

    @property
    def mechanism(self) -> Mechanism:
        """Mechanism in force: the published candidate, else the template."""
        if self.m is not None:
            return self.m
        if self.mechanism_template is None:
            raise RuntimeError(
                "No mechanism available: none published for "
                f"mechanism_id={self.mechanism_id} and no template configured."
            )
        return self.mechanism_template

    @property
    def published_mechanism_assigned(self) -> bool:
        """Whether a candidate fetched from the ``World`` (not the template) is in force."""
        return self.m is not None and not self._using_default_mechanism

    def set_opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""
        self._opt_id = opt_id

    # --- helpers -------------------------------------------------------------------

    def _zero_actions(self) -> MultiAgentDict:
        """Zero action per agent, shaped like the declared action space."""
        zeros = {}
        for agent_id in self.agents:
            space = self.action_spaces.get(agent_id)
            shape = getattr(space, "shape", None) or (0,)
            zeros[agent_id] = np.zeros(shape, dtype=np.float32)
        return zeros

    def _publish(self, payload: ContextSchema) -> None:
        ctx = Context(
            id=None,
            opt_id=self._opt_id,
            step=self._t,
            env=self.__class__.__name__,
            payload=payload,
        )
        ray.get(self.world.append_context.remote(ctx))

    def _update_infos(self, key: str, values: MultiAgentDict | SupportsFloat) -> None:
        """Record a per-agent (or broadcast scalar) diagnostic in ``infos``."""
        values = (
            values
            if isinstance(values, dict)
            else {agent_id: values for agent_id in self._infos}
        )
        for agent_id, value in values.items():
            self._infos[agent_id][key] = value

    def _normalize_action(self, action: ActType) -> np.ndarray:
        """Squash raw policy outputs to ``[0, 1]`` component-wise."""
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.asarray(
            [sigmoid(float(value) / self.action_temperature) for value in z],
            dtype=np.float32,
        )

    def _fetch_published_mechanism(self) -> None:
        if self.mechanism_id is None:
            raise RuntimeError(
                "RegulatedEnv has no mechanism_id. mechanism_id must be injected at env creation."
            )
        try:
            new_ctx = ray.get(
                self.world.get_mechanism_by_id.remote(
                    mechanism_id=self.mechanism_id,
                    seed=self.policy_seed,
                    mode=self.mode,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch mechanism_id={self.mechanism_id} from World."
            ) from e

        if new_ctx is not None:
            self.m_ctx = new_ctx
            self.m = new_ctx.mechanism
            self._using_default_mechanism = False

    def _all_agents(self, value: Any) -> MultiAgentDict:
        out = {agent_id: value for agent_id in self.agents}
        out["__all__"] = value
        return out

    # --- gymnasium API ------------------------------------------------------------

    @override(MultiAgentEnv)
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Start a new episode under the mechanism currently in force.

        The step counter restarts. If no published candidate has been assigned
        yet, one is fetched from the ``World`` by ``mechanism_id``; otherwise
        the current mechanism is kept for the whole run. The benchmark
        ``@reset`` hook then produces the initial state ``S_t``, per-agent
        infos and previous actions are cleared, and the initial observations
        go through :meth:`observation`. Both ``seed`` and ``options`` are
        ignored: the environment seed is fixed at construction.

        Returns
        -------
        tuple[MultiAgentDict, MultiAgentDict]
            Per-agent initial observations and per-agent (empty) infos.
        """
        # The env seed is fixed at construction; RLlib's per-reset seed is ignored.
        self._t = 0

        # Try to fetch a new mechanism if one is available (published);
        # otherwise keep the current one for subsequent episodes.
        if not self.published_mechanism_assigned:
            self._fetch_published_mechanism()

        if self._reset is not None:
            self.S_t = getattr(self, self._reset)()
        self._infos = {agent_id: {} for agent_id in self.agents}
        self.previous_actions = self._zero_actions()

        return self.observation({}), self._infos

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Run one regulated step of the benchmark (see the module docstring).

        Raw policy outputs go through :meth:`action`; the benchmark ``@reward``
        hook computes the intrinsic rewards on the current state and delivered
        actions, the ``@transition`` hook advances ``S_t``, :meth:`reward`
        applies the mechanism's reward transform (with ``action_after`` set to
        the delivered actions) and :meth:`observation` builds the next
        observations. The intrinsic rewards are recorded in the infos as
        ``intrinsic_utility`` and an ``EnvStepContext`` is published to the
        ``World``. Episodes never terminate; ``truncated["__all__"]`` becomes
        ``True`` at ``horizon``.

        Before any candidate mechanism has been published (RLlib's environment
        checks), the step is inert: zero rewards, no dynamics, nothing
        published.

        Returns
        -------
        tuple
            ``(obs, rewards, terminated, truncated, infos)``, each keyed by
            agent ID; ``terminated`` and ``truncated`` also carry ``"__all__"``.
        """
        # policy outputs -> normalized, benchmark-adjusted, mechanism-regulated actions
        actions = self.action(action_dict)

        if not self.published_mechanism_assigned:
            # No candidate published yet (e.g. RLlib env checking): inert step.
            self.previous_actions = actions
            self._t += 1
            return (
                self.observation({}),
                {agent_id: 0.0 for agent_id in self.agents},
                self._all_agents(False),
                self._all_agents(False),
                self._infos,
            )

        # benchmark intrinsic reward on the current state and delivered actions
        if self._reward is not None:
            intrinsic_rewards = getattr(self, self._reward)(actions)
        else:
            intrinsic_rewards = {agent_id: 0.0 for agent_id in self.agents}

        # dynamics
        if self._transition is not None:
            self.S_t = getattr(self, self._transition)(A_t=actions, S_t=dict(self.S_t))

        rewards = self.reward(intrinsic_rewards, action_after=actions)
        self.previous_actions = actions
        obs = self.observation({})

        time_limit = self.horizon is not None and (self._t + 1) >= self.horizon
        terminated = self._all_agents(False)
        truncated = self._all_agents(bool(time_limit))

        self._update_infos(key="intrinsic_utility", values=intrinsic_rewards)
        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=self.mechanism_id,
                observation=obs,
                observation_map=self.obs_map,
                reward=rewards,
                action=actions,
                info=self._infos,
            )
        )
        self._t += 1
        return obs, rewards, terminated, truncated, self._infos

    # --- pipeline methods ------------------------------------------------------------

    def action(self, action_dict: MultiAgentDict) -> MultiAgentDict:
        """``a* = M^A(s, a)``: normalize, apply the benchmark hook, then the mechanism."""
        action_dict = {
            agent_id: self._normalize_action(action)
            for agent_id, action in action_dict.items()
        }
        if self._action is not None:
            action_dict = getattr(self, self._action)(action_dict)
        mechanism = self.mechanism
        return mechanism.action(action_dict, env=self, **mechanism.resolve(self))

    def reward(self, reward_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """``r* = M^R(r, s, a*, s')``: mechanism reward transform of the base reward."""
        mechanism = self.mechanism
        return mechanism.reward(
            reward_dict, env=self, **{**kwargs, **mechanism.resolve(self)}
        )

    def observation(self, observation_dict: MultiAgentDict) -> MultiAgentDict:
        """``o* = M^O(s, [o, theta])``: benchmark observation, mechanism vector, transform."""
        if self._observation is not None:
            observation_dict = getattr(self, self._observation)(observation_dict)

        theta = np.asarray(self.mechanism.to_vector(), dtype=np.float32).reshape(-1)
        obs_with_theta = {
            agent_id: np.concatenate(
                [np.asarray(observation, dtype=np.float32).reshape(-1), theta]
            ).astype(np.float32, copy=False)
            for agent_id, observation in observation_dict.items()
        }
        mechanism = self.mechanism
        return mechanism.observation(
            obs_with_theta, env=self, **mechanism.resolve(self)
        )
