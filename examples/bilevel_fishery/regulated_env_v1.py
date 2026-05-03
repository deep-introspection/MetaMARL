"""V1 multi-agent regulated fishery environment (risk-sensitive penalties).

Extends the V0 environment by replacing binary fine/ban enforcement with a
smooth, *predictive* risk penalty that discourages fishing when the stock is
below or approaching ``target_stock``.

Ecology model (modified Lotka-Volterra, Euler integration)
----------------------------------------------------------
Fish dynamics:

    X_{t+1} = X_t + dt * (delta * Y_t * X_t * (1 - X_t/X_max) - gamma * X_t - H_t)

Algae dynamics:

    Y_{t+1} = Y_t + dt * (alpha * Y_t * (1 - Y_t/Y_max) - beta * Y_t * X_t)

Note the logistic growth terms ``(1 - X/X_max)`` and ``(1 - Y/Y_max)``, which
bound the populations independently of the harvest.

Intrinsic utility (V1)
----------------------
Sustainability-modulated harvest reward:

    u_i = a_i * fish_norm * min(1, fish_norm / target_stock)

This gives zero utility at zero stock and full utility only when the stock
is at or above ``target_stock``.

Penalty (V1 — predictive collapse)
-----------------------------------
A continuous risk penalty based on current *and predicted* next-step stock
shortfall below ``target_stock``:

    penalty = scale * (shortfall_now^power + shortfall_next^power)

where shortfall is normalised by ``max_fish``.
"""

import logging
from typing import SupportsFloat, Tuple

import numpy as np
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

EPS = 1e-8


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    """V1 multi-agent fishery environment with risk-sensitive continuous penalties.

    Key differences from V0 (:class:`~regulated_env.FisheryRegulatedEnv`):

    - **No bans or stochastic enforcement**: penalties are applied every step,
      proportionally to the predicted stock shortfall.
    - **Sustainability-modulated utility**: the agent's intrinsic reward is
      dampened when the stock is below ``target_stock``.
    - **Modified ecology**: fish and algae growth include logistic
      density-dependent terms.

    Parameters
    ----------
    ecology_cfg : dict
        Dictionary with keys:

        - ``algae_init``, ``fish_init``: initial biomasses.
        - ``max_fish``, ``max_algae``: carrying capacities.
        - ``alpha``, ``beta``, ``delta``, ``gamma``: Lotka-Volterra parameters.
        - ``dt``: Euler integration step size.
    **kwargs
        Forwarded to :class:`~core.envs.marl_regulated.MultiAgentRegulatedEnv`.
    """

    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.algae_init = ecology_cfg["algae_init"]
        self.fish_init = ecology_cfg["fish_init"]
        self.max_fish = ecology_cfg["max_fish"]
        self.max_algae = ecology_cfg["max_algae"]
        self.alpha = ecology_cfg["alpha"]
        self.beta = ecology_cfg["beta"]
        self.delta = ecology_cfg["delta"]
        self.gamma = ecology_cfg["gamma"]
        self.dt = ecology_cfg["dt"]

        self.obs_map = [
            "fish_norm",
            "algae_norm",
            "effective_quota",
            "no_fish_zone",
            "fixed_quota",
            "prop_quota",
            "min_stock",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
        ]

    def _reset(self) -> dict[str, np.ndarray]:
        """Reset the environment state for a new episode.

        Initialises fish and algae with log-normal noise (sigma=0.05) around
        their nominal initial values and returns initial observations.

        Returns
        -------
        dict of str → np.ndarray
            Initial observation for each agent (shape: ``(4 + mechanism_dim,)``).
        """
        self.S_t = {
            "fish": max(EPS, self.rng.lognormal(np.log(self.fish_init), 0.05)),
            "algae": max(EPS, self.rng.lognormal(np.log(self.algae_init), 0.05)),
        }

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        return obs

    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Advance the V1 environment by one step.

        For each agent:
        1. Compute sustainability-modulated intrinsic utility
           via :meth:`intrinsic_utility`.
        2. Compute the composite violation (quota + predictive collapse
           penalty) via :meth:`violation_signal`.
        3. Set ``reward_i = u_i - v_i``; no stochastic enforcement.
        4. Broadcast mean reward across agents via :meth:`aggregate_rewards`.
        5. Advance ecology via :meth:`transition_kernel`.

        Parameters
        ----------
        action_dict : dict of AgentID → ActType
            Harvest fraction actions for all agents.

        Returns
        -------
        obs : dict of AgentID → np.ndarray
        rewards : dict of AgentID → float
        terminated : dict of AgentID → bool
            Always ``False``.
        truncated : dict of AgentID → bool
            ``True`` when horizon is reached.
        infos : dict of AgentID → dict
            Per-agent diagnostics including ``harvest``, ``intrinsic_utility``,
            ``violation_signal``, ``quota_violation``, ``preventive_penalty``,
            ``fine``, ``harvest_scale``, ``H_total``, ``below_target_zone``,
            ``target_shortfall``.
        """
        rewards = {}
        fines = {}
        utilities = {}
        violations = {}
        quota_violations = {}
        preventive_penalties = {}

        effective_actions = dict(action_dict)

        for agent_id in self.agents:
            u = float(
                self.intrinsic_utility(agent_id, effective_actions[agent_id], self.S_t)
            )

            v_dict = self.violation_signal(
                agent_id=agent_id,
                u_i=u,
                S_t=self.S_t,
                A_t=effective_actions,
            )

            quota_violation = float(v_dict["quota"])
            preventive_penalty = float(v_dict["preventive"])
            v = float(v_dict["total"])

            utilities[agent_id] = u
            quota_violations[agent_id] = quota_violation
            preventive_penalties[agent_id] = preventive_penalty
            violations[agent_id] = v
            rewards[agent_id] = u - v
            fines[agent_id] = preventive_penalty

        rewards = self.aggregate_rewards(rewards)

        realized_harvest, H_total, harvest_scale = self._compute_harvest_metrics(
            effective_actions, self.S_t
        )
        S_t = self.S_t.copy()

        self.S_t = self.transition_kernel(A_t=effective_actions, S_t=S_t)

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

        time_limit = self._is_truncated()

        terminated = {aid: False for aid in self.agents}
        terminated["__all__"] = False

        truncated = {aid: time_limit for aid in self.agents}
        truncated["__all__"] = time_limit

        infos = {
            agent_id: {
                "harvest": realized_harvest.get(agent_id, 0.0),
                "intrinsic_utility": utilities.get(agent_id, 0.0),
                "violation_signal": violations.get(agent_id, 0.0),
                "quota_violation": quota_violations.get(agent_id, 0.0),
                "preventive_penalty": preventive_penalties.get(agent_id, 0.0),
                "fine": fines.get(agent_id, 0.0),
                "harvest_scale": harvest_scale,
                "H_total": H_total,
                "below_target_zone": float(S_t["fish"] / self.max_fish < self.m.target_stock),
                "target_shortfall": float(
                    max(0.0, self.m.target_stock - (S_t["fish"] / self.max_fish))
                ),
            }
            for agent_id in self.agents
        }
        return obs, rewards, terminated, truncated, infos

    def _is_truncated(self) -> bool:
        """Return ``True`` when the episode horizon has been reached.

        Returns
        -------
        bool
            ``True`` iff ``self._t >= self.horizon``.
        """
        return self._t >= self.horizon

    def desired_harvest_signal(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Return the agent's desired (unconstrained) harvest as a normalised fraction.

        .. math::

            d_i = a_i \\cdot \\frac{X_t}{X_{\\max}}

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the acting agent.
        action : ActType
            Harvest fraction in [0, 1].
        S_t : dict of str → float
            Current ecological state.

        Returns
        -------
        SupportsFloat
            Desired harvest in normalised units (same scale as ``fish_norm``).
        """
        action = float(np.asarray(action).item())
        fish_norm = S_t["fish"] / self.max_fish
        return action * fish_norm

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        """Compute the sustainability-modulated intrinsic harvest utility.

        Unlike V0, the utility is reduced when the stock is below
        ``target_stock``:

        .. math::

            u_i = a_i \\cdot \\frac{X_t}{X_{\\max}} \\cdot
                \\min\\!\\left(1,\\; \\frac{X_t / X_{\\max}}{\\theta_{\\text{target}}}
                \\right)

        This means the agent earns zero utility at zero stock and full utility
        only when the normalised stock is at or above ``target_stock``.

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the acting agent (unused).
        action : ActType
            Harvest fraction in [0, 1].
        S_t : dict of str → float
            Current ecological state.

        Returns
        -------
        SupportsFloat
            Non-negative utility value.
        """
        action = float(np.asarray(action).item())
        fish_norm = S_t["fish"] / self.max_fish
        target_stock = max(EPS, self.m.target_stock)

        sustainability_factor = min(1.0, fish_norm / target_stock)
        return action * fish_norm * sustainability_factor

    def violation_signal(
        self,
        agent_id: AgentID,
        u_i: SupportsFloat,
        S_t: dict[str, MultiAgentDict],
        A_t: dict[AgentID, ActType],
    ) -> dict[str, float]:
        """Compute the composite violation signal for one agent (V1).

        Two components:

        **Quota violation** (same as V0):

        .. math::

            v_{\\text{quota}} = \\max\\!\\left(0,\\; d_i -
                \\min(q_{\\text{fixed}},\\; q_{\\text{prop}} \\cdot
                \\text{fish\\_norm})\\right)

        **Predictive collapse penalty** (V1-specific):

        .. math::

            v_{\\text{prev}} = s \\left(
                \\left(\\frac{\\theta - X_t / X_{\\max}}{X_{\\max}}
                \\right)_+^p
                + \\left(\\frac{\\theta - \\hat{X}_{t+1} / X_{\\max}}{X_{\\max}}
                \\right)_+^p
            \\right)

        where :math:`s` = ``risk_penalty_scale``, :math:`p` =
        ``risk_penalty_power``, :math:`\\theta` = ``target_stock``, and
        :math:`\\hat{X}_{t+1}` is a one-step look-ahead fish stock prediction
        (computed ignoring density-dependence for simplicity).

        Parameters
        ----------
        agent_id : AgentID
            Identifier of the acting agent.
        u_i : SupportsFloat
            Intrinsic utility from :meth:`intrinsic_utility`.
        S_t : dict of str → float
            Current ecological state.
        A_t : dict of AgentID → ActType
            All agents' current actions (needed for the total harvest in the
            look-ahead prediction).

        Returns
        -------
        dict of str → float
            Keys: ``"quota"`` (float), ``"preventive"`` (float),
            ``"total"`` (float).
        """
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish

        raw_harvest_signal = float(
            self.desired_harvest_signal(
                agent_id=agent_id,
                action=A_t[agent_id],
                S_t=S_t,
            )
        )

        quota = max(
            0.0,
            raw_harvest_signal - min(self.m.fixed_quota, self.m.prop_quota * fish_norm),
        )
        preventive = self._predictive_collapse_penalty(A_t=A_t, S_t=S_t)

        total = float(quota + preventive)
        return {
            "quota": float(quota),
            "preventive": float(preventive),
            "total": total,
        }

    def penalty(self) -> SupportsFloat:
        """Return the base fine multiplier for quota violations.

        Returns
        -------
        SupportsFloat
            ``self.m.fine_amount``.
        """
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        """Advance the V1 ecology by one Euler step (logistic Lotka-Volterra).

        V1 adds density-dependent growth terms compared to the V0 Lotka-Volterra:

        .. math::

            X_{t+1} = X_t + \\Delta t \\left(
                \\delta \\, Y_t \\, X_t \\left(1 - \\frac{X_t}{X_{\\max}}\\right)
                - \\gamma \\, X_t - H_t
            \\right)

        .. math::

            Y_{t+1} = Y_t + \\Delta t \\left(
                \\alpha \\, Y_t \\left(1 - \\frac{Y_t}{Y_{\\max}}\\right)
                - \\beta \\, Y_t \\, X_t
            \\right)

        Both values are clipped to ``[0, max_fish]`` / ``[0, max_algae]``.

        Parameters
        ----------
        A_t : dict of AgentID → ActType
            Agent actions for the current step.
        S_t : dict of str → float
            Current ecological state with keys ``"fish"`` and ``"algae"``.

        Returns
        -------
        dict of str → float
            Next ecological state.
        """
        fish = S_t["fish"]
        algae = S_t["algae"]

        _, H, _ = self._compute_harvest_metrics(A_t, S_t)

        fish_next = fish + self.dt * (
            self.delta * algae * fish * (1 - fish / self.max_fish)
            - self.gamma * fish
            - H
        )
        algae_next = algae + self.dt * (
            self.alpha * algae * (1 - algae / self.max_algae)
            - self.beta * algae * fish
        )

        fish_next = np.clip(fish_next, 0.0, self.max_fish)
        algae_next = np.clip(algae_next, 0.0, self.max_algae)

        return {"fish": fish_next, "algae": algae_next}

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        """Broadcast the mean reward to all agents (equalitarian redistribution).

        Parameters
        ----------
        rewards : MultiAgentDict
            Per-agent rewards before aggregation.

        Returns
        -------
        MultiAgentDict
            Mapping of agent ID → mean reward (same value for all agents).
        """
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}

    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> np.ndarray:
        """Build the V1 observation vector (4 base features).

        V1 removes the ban-related feature present in V0 and replaces it with
        no additional feature (ban enforcement is gone in V1).

        ======  =================================================
        Index   Feature
        ======  =================================================
        0       ``fish / max_fish`` (normalised fish stock)
        1       ``algae / max_algae`` (normalised algae)
        2       Effective quota ``min(fixed_quota, prop_quota * fish_norm)``
        3       No-fish-zone indicator (``fish_norm < min_stock``)
        ======  =================================================

        The mechanism vector from :meth:`~FisheryMechanism.to_vector` is
        appended by the parent class, giving total length ``4 + mechanism_dim``.

        Parameters
        ----------
        agent_id : AgentID
            Agent identifier (unused).
        S_t : dict of str → float
            Current ecological state.

        Returns
        -------
        np.ndarray
            Shape ``(4,)``, dtype ``float32``.
        """
        fish_norm = S_t["fish"] / self.max_fish
        algae_norm = S_t["algae"] / self.max_algae

        effective_quota = min(self.m.fixed_quota, self.m.prop_quota * fish_norm)
        no_fish_zone = float(fish_norm < self.m.min_stock)

        return np.array(
            [
                fish_norm,
                algae_norm,
                effective_quota,
                no_fish_zone,
            ],
            dtype=np.float32,
        )

    def _compute_harvest_metrics(
        self, A_t: dict[AgentID, ActType], S_t: dict[str, float]
    ) -> tuple[dict[AgentID, float], float, float]:
        """Compute realised per-agent harvests with proportional scarcity scaling.

        Identical in structure to the V0 method; uses
        :meth:`desired_harvest_signal` (rather than ``intrinsic_utility``)
        to compute per-agent desired harvests.

        Parameters
        ----------
        A_t : dict of AgentID → ActType
            Harvest fraction actions for all agents.
        S_t : dict of str → float
            Current ecological state.

        Returns
        -------
        realized_harvest : dict of AgentID → float
            Per-agent realised harvest in physical units.
        H_total : float
            Sum of all realised harvests.
        scale : float
            Scarcity scaling factor in ``(0, 1]``.
        """
        fish = S_t["fish"]
        fish_norm = fish / self.max_fish

        desired = {
            agent_id: float(
                self.desired_harvest_signal(
                    agent_id=agent_id,
                    action=A_t[agent_id],
                    S_t=S_t,
                )
            )
            for agent_id in self.agents
        }

        total_desired = float(sum(desired.values()))
        scale = min(1.0, fish_norm / max(EPS, total_desired))

        realized_harvest = {
            agent_id: self.max_fish * desired[agent_id] * scale
            for agent_id in self.agents
        }
        H_total = float(sum(realized_harvest.values()))

        return realized_harvest, H_total, scale

    def _predictive_collapse_penalty(
        self, A_t: dict[AgentID, ActType], S_t: dict[str, float]
    ) -> float:
        """Compute a forward-looking collapse penalty based on stock shortfall.

        Predicts the fish stock one step ahead (using a simplified linear
        Euler step, ignoring density-dependent growth) and computes a penalty
        proportional to the normalised shortfall below ``target_stock`` at
        both the current and predicted next step:

        .. math::

            P = s \\left(
                \\left(\\frac{\\max(0, \\theta X_{\\max} - X_t)}{X_{\\max}}
                \\right)^p
                + \\left(\\frac{\\max(0, \\theta X_{\\max} - \\hat{X}_{t+1})}
                {X_{\\max}}\\right)^p
            \\right)

        where :math:`s` = ``risk_penalty_scale``, :math:`p` =
        ``risk_penalty_power``, :math:`\\theta` = ``target_stock``.

        Parameters
        ----------
        A_t : dict of AgentID → ActType
            Current agent actions (used to compute total harvest for the
            one-step prediction).
        S_t : dict of str → float
            Current ecological state with keys ``"fish"`` and ``"algae"``.

        Returns
        -------
        float
            Non-negative penalty value.
        """
        fish = S_t["fish"]
        algae = S_t["algae"]

        _, H_total, _ = self._compute_harvest_metrics(A_t, S_t)

        fish_next_pred = fish + self.dt * (
            self.delta * algae * fish - self.gamma * fish - H_total
        )

        safe_fish = self.m.target_stock * self.max_fish

        current_shortage = max(0.0, safe_fish - fish)
        predicted_shortage = max(0.0, safe_fish - fish_next_pred)

        current_shortage_norm = current_shortage / self.max_fish
        predicted_shortage_norm = predicted_shortage / self.max_fish

        penalty_scale = self.m.risk_penalty_scale
        penalty_power = self.m.risk_penalty_power

        return float(
            penalty_scale
            * (
                current_shortage_norm**penalty_power
                + predicted_shortage_norm**penalty_power
            )
        )