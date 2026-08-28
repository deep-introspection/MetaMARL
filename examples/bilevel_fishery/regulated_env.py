"""Single-stock fishery benchmark regulated by explicit mechanisms.

``N`` fishers share a stock ``B_t`` (biomass) with Pella-Tomlinson growth

    B_{t+1} = B_t + (r / p) * B_t * (1 - (B_t / K)^p) + noise + restoration - H_t

where ``K`` is the carrying capacity, ``r`` the intrinsic growth rate, ``p`` the
shape parameter (``p = 1`` gives the Schaefer logistic model), ``noise`` is
multiplicative Gaussian process noise and ``H_t`` the realized total harvest.
Each agent's action has two components in ``[0, 1]``:

- ``action[0]``: harvest fraction of its maximal request
  ``full_required_harvest = m * F_msy * B_t / N`` (``m`` the unregulated
  fishing-mortality multiplier);
- ``action[1]``: restoration effort, converted to biomass through
  ``restoration_effectiveness`` (ecology) and rewarded/costed by a
  ``SubsidyMechanism`` (incentive), keeping the two roles separate.

The intrinsic reward is the delivered harvest normalized by the maximal
request, as in the pre-mechanism benchmark. Mechanisms (quota, subsidy,
penalty, social observation) plug in through ``MultiAgentRegulatedEnv``.

When the env is built with a metric ``schema`` (``FisheryMetricSchema`` or a
subclass), the hooks push one value per step for every field of that schema
that the fishery produces (stock, growth, harvests, reference points, quota
allowance) and per-agent harvest requests, so that the env-level reporter and
the regulator's fitness aggregation have data; without a schema nothing is
logged.

References
----------
Pella, J. J., & Tomlinson, P. K. (1969). A generalized stock production
model. Inter-American Tropical Tuna Commission Bulletin, 13(3), 416-497.
"""

import logging

import numpy as np

from core.envs.hooks import observation, reset, reward, transition
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.types import MultiAgentDict

logger = logging.getLogger(__name__)

EPS = 1e-8

HARVEST_COMPONENT = 0
RESTORATION_COMPONENT = 1


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    """Pella-Tomlinson fishery (see module docstring).

    Parameters
    ----------
    ecology_cfg : dict
        ``r`` (growth rate, default 0.3), ``K`` (carrying capacity, biomass
        units), ``p`` (shape, default 1.0), ``fish_init``/``B0`` (initial
        biomass, default ``K``), ``sigma`` (process-noise std as a fraction of
        biomass, default 0.05), ``initial_stock_log_sigma`` (log-normal spread
        of the initial biomass, default 0.05), ``unregulated_f_multiplier``
        (default 2.0) and ``restoration_effectiveness`` (biomass added per unit
        of total restoration effort, default 0.0 -- restoration is inert unless
        set).
    """

    def __init__(self, *, ecology_cfg: dict, **kwargs):
        super().__init__(**kwargs)

        self.r = float(ecology_cfg.get("r", 0.3))
        self.K = max(
            float(ecology_cfg.get("K", ecology_cfg.get("max_fish", 1000.0))), EPS
        )
        self.p = max(float(ecology_cfg.get("p", 1.0)), EPS)
        self.fish_init = float(
            ecology_cfg.get("fish_init", ecology_cfg.get("B0", self.K))
        )

        # stochasticity
        self.sigma = float(ecology_cfg.get("sigma", 0.05))
        self.initial_stock_log_sigma = float(
            ecology_cfg.get("initial_stock_log_sigma", 0.05)
        )

        # reference points
        self.B_msy = max(self.K * (1.0 / (self.p + 1.0)) ** (1.0 / self.p), EPS)
        self.MSY = self.r * self.K / (self.p + 1.0) ** ((self.p + 1.0) / self.p)
        self.F_msy = self.MSY / max(self.B_msy, EPS)

        self.unregulated_f_multiplier = float(
            ecology_cfg.get("unregulated_f_multiplier", 2.0)
        )
        self.restoration_effectiveness = float(
            ecology_cfg.get("restoration_effectiveness", 0.0)
        )

        self.obs_map = ["fish_norm", "total_usage_norm"]

    # --- helpers -------------------------------------------------------------------

    def _full_required_harvest(self, fish: float) -> float:
        """Maximal per-agent harvest request at biomass ``fish``."""
        return self.unregulated_f_multiplier * self.F_msy * fish / len(self.agents)

    @staticmethod
    def _components(action) -> tuple[float, float]:
        """Split a delivered action into ``(harvest_fraction, restoration_effort)``."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        harvest = float(a[HARVEST_COMPONENT])
        restoration = (
            float(a[RESTORATION_COMPONENT])
            if a.shape[0] > RESTORATION_COMPONENT
            else 0.0
        )
        return harvest, restoration

    def _quota_allowed_fraction(self, fish_norm: float) -> float:
        """Allowed harvest fraction of the quota in force, ``1.0`` without a quota.

        Walks the mechanism (and the children of a composition) for the first
        one exposing ``allowed_fraction`` and evaluates it at ``fish_norm``.
        Used only for logging (``quota_stress``, ``allowed_harvest``); the
        regulation itself happens in ``mechanism.action``.
        """
        pending = [self.mechanism]
        while pending:
            candidate = pending.pop(0)
            allowed_fraction = getattr(candidate, "allowed_fraction", None)
            if callable(allowed_fraction):
                return float(allowed_fraction(fish_norm))
            pending.extend(getattr(candidate, "children", ()))
        return 1.0

    # --- hooks -----------------------------------------------------------------------

    @reset
    def reset_fishery(self) -> dict[str, float]:
        if self.initial_stock_log_sigma == 0.0:
            initial_fish = self.fish_init
        else:
            initial_fish = self.rng.lognormal(
                mean=np.log(max(self.fish_init, EPS)),
                sigma=self.initial_stock_log_sigma,
            )
        return {"fish": float(np.clip(initial_fish, EPS, self.K)), "last_usage": 0.0}

    @reward
    def harvest_utility(self, A_t: MultiAgentDict) -> MultiAgentDict:
        """Intrinsic utility: delivered harvest fraction (revenue proxy)."""
        fish = float(self.S_t["fish"])
        full_required_harvest = self._full_required_harvest(fish)
        utilities = {}
        for agent_id, action in A_t.items():
            harvest_frac, _ = self._components(action)
            utilities[agent_id] = float(harvest_frac)
            requested_harvest = harvest_frac * full_required_harvest
            self._infos[agent_id]["requested_harvest"] = requested_harvest
            self._infos[agent_id]["requested_frac"] = harvest_frac
            self._log(("by_agent", agent_id, "requested_harvest"), requested_harvest)
            self._log(("by_agent", agent_id, "requested_frac"), harvest_frac)
        self._update_infos(key="full_required_harvest", values=full_required_harvest)
        return utilities

    @transition
    def pella_tomlinson(
        self, *, A_t: MultiAgentDict, S_t: dict, **kwargs
    ) -> dict[str, float]:
        fish = float(S_t["fish"])
        full_required_harvest = self._full_required_harvest(fish)

        harvest_fracs = {}
        efforts = {}
        for agent_id, action in A_t.items():
            harvest_fracs[agent_id], efforts[agent_id] = self._components(action)

        delivered_harvest = {
            agent_id: frac * full_required_harvest
            for agent_id, frac in harvest_fracs.items()
        }
        H = float(sum(delivered_harvest.values()))
        B = max(fish, EPS)

        noise = self.sigma * self.rng.normal() * fish
        restoration = self.restoration_effectiveness * float(sum(efforts.values()))

        biological_growth = (self.r / self.p) * B * (1.0 - (B / self.K) ** self.p)
        growth = biological_growth + noise + restoration
        available = max(B + growth, 0.0)

        H_realized = min(H, available)
        fish_next = float(np.clip(available - H_realized, 0.0, self.K))

        new_state = {"fish": fish_next, "last_usage": H_realized}

        fish_norm = fish / self.K
        fish_norm_next = fish_next / self.K
        total_usage_norm = H_realized / self.K
        # quota allowance at the pre-transition stock (what the action hook saw)
        allowed_frac = self._quota_allowed_fraction(fish_norm)
        allowed_harvest = allowed_frac * full_required_harvest

        for agent_id in A_t:
            self._infos[agent_id]["delivered_harvest"] = delivered_harvest[agent_id]
            self._infos[agent_id]["restoration_effort"] = efforts[agent_id]
            self._log(
                ("by_agent", agent_id, "delivered_harvest"), delivered_harvest[agent_id]
            )
        self._update_infos(key="fish", values=fish)
        self._update_infos(key="fish_next", values=fish_next)
        self._update_infos(key="fish_norm", values=fish_norm)
        self._update_infos(key="fish_norm_next", values=fish_norm_next)
        self._update_infos(key="growth", values=growth)
        self._update_infos(key="growth_noise", values=noise)
        self._update_infos(key="restoration", values=restoration)
        self._update_infos(key="H_attempted", values=H)
        self._update_infos(key="H_realized", values=H_realized)
        self._update_infos(key="harvest_to_msy", values=H_realized / max(self.MSY, EPS))
        self._update_infos(key="total_usage_norm", values=total_usage_norm)
        self._update_infos(key="allowed_harvest", values=allowed_harvest)
        self._update_infos(key="B_msy", values=self.B_msy)
        self._update_infos(key="MSY", values=self.MSY)
        self._update_infos(key="F_msy", values=self.F_msy)

        # env-level series (one value per step, aligned with ``iter``); the
        # field names are those of ``FisheryMetricSchema``
        self._log(("fish_stock",), fish)
        self._log(("fish_stock_next",), fish_next)
        self._log(("fish_norm",), fish_norm)
        self._log(("fish_norm_next",), fish_norm_next)
        self._log(("growth",), growth)
        self._log(("growth_noise",), noise)
        self._log(("H_attempted",), H)
        self._log(("H_realized",), H_realized)
        self._log(("total_usage_norm",), total_usage_norm)
        self._log(("quota_stress",), allowed_frac)
        self._log(("allowed_harvest",), allowed_harvest)
        self._log(("B_msy",), self.B_msy)
        self._log(("MSY",), self.MSY)
        self._log(("F_msy",), self.F_msy)

        return new_state

    @observation
    def fishery_observation(self, observation_dict: MultiAgentDict) -> MultiAgentDict:
        fish_norm = float(self.S_t["fish"]) / self.K
        total_usage_norm = float(self.S_t.get("last_usage", 0.0)) / self.K
        observation = np.array([fish_norm, total_usage_norm], dtype=np.float32)
        return {agent_id: observation.copy() for agent_id in self.agents}
