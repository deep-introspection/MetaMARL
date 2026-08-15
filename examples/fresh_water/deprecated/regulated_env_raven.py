import logging
from time import time
from typing import SupportsFloat

import numpy as np
import subprocess
import csv
import os
from typing import Optional
import shutil
import tempfile
from typing import Dict
import hashlib
import json
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


# Numerical stability constant
EPS = 1e-8


class WaterRegulatedRavenEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict | None = None,
        # When True, run Raven and read its CSV outputs to drive the env state.
        use_raven: bool = False,
        # Raw command to invoke Raven; if None a sensible default is used.
        raven_cmd: Optional[str] = None,
        # Working directory where the `raven` invocation should run (relative to repo root).
        raven_cwd: str = "raven",
        # How often (in env steps) to call Raven. 1 => every step. 0 disables automatic runs.
        raven_freq: int = 1,
        # Which column name from ReservoirStages CSV to use as the 'water' state.
        raven_stage_col: str = "Belwood_Lake",
        **kwargs,
    ):
        super().__init__(**kwargs)

        env_cfg = ecology_cfg or {}
        # Ecology params
        self.water_init = env_cfg.get("water_init", 80.0)
        self.max_water = env_cfg.get("max_water", 100.0)
        self.inflow_rate = env_cfg.get("inflow_rate", 1.0)
        # time-step used in transition kernel (mirror fishery `dt` semantics)
        self.dt = env_cfg.get("dt", 1.0)

        # Restriction tracking per agent
        self._agent_restrictions = {}

        # Raven integration options
        self.use_raven = use_raven
        self.raven_cmd = raven_cmd
        self.raven_cwd = raven_cwd
        self.raven_freq = max(0, raven_freq)
        self.raven_stage_col = raven_stage_col
        # Parse initial reservoir IDs and values from the model .rvc so we can
        # map actions to allowed overrides (InitialReservoirStage) without
        # editing the simulator directly.
        self._initial_reservoirs = self._parse_rvc_initial_reservoirs()
        # Fractional range around initial stage allowed for action-driven changes
        self.raven_stage_delta_fraction = env_cfg.get("raven_stage_delta_fraction", 0.1)
        # time() returns a float; hashlib expects bytes, so encode a string
        self.key = hashlib.sha256(str(time()).encode("utf-8")).hexdigest()
        self.run_root = None

    def _parse_rvc_initial_reservoirs(self) -> Dict[str, float]:
        """Parse `:InitialReservoirStage <id> <value>` from the model .rvc and
        return mapping id->value as strings."""
        rvc = os.path.join(self.raven_cwd, "2_Raven", "ohms_canshield.rvc")
        if not os.path.exists(rvc):
            rvc = os.path.join(self.raven_cwd, "ohms_canshield.rvc")

        out: Dict[str, float] = {}
        try:
            with open(rvc, "r", encoding="utf-8") as fh:
                for ln in fh:
                    if ln.strip().startswith(":InitialReservoirStage"):
                        parts = ln.split()
                        if len(parts) >= 3:
                            rid = parts[1]
                            try:
                                val = float(parts[2])
                            except Exception:
                                continue
                            out[rid] = val
        except Exception:
            logger.exception("Failed parsing rvc for initial reservoir stages: %s", rvc)
        return out

    def _actions_to_overrides(self, A_t: MultiAgentDict) -> Dict[str, Dict[str, float]]:
        """Map agent actions to allowed InitialReservoirStage overrides.

        We respect the allowed delta fraction and map each reservoir id to a new
        absolute value. Currently we map all agents to the same reservoir set.
        Returns: {'InitialReservoirStage': {rid: new_value, ...}}
        """
        if not self._initial_reservoirs:
            return {}

        # For now aggregate agent actions into a single scalar in [0,1]
        vals = [float(np.asarray(a).item()) for a in A_t.values()]
        mean_action = float(np.mean(vals)) if vals else 0.0

        overrides: Dict[str, Dict[str, float]] = {"InitialReservoirStage": {}}
        for rid, base in self._initial_reservoirs.items():
            delta = self.raven_stage_delta_fraction * base
            # map mean_action in [0,1] to [base - delta, base + delta]
            new_val = base - delta + mean_action * (2 * delta)
            overrides["InitialReservoirStage"][rid] = float(new_val)

        return overrides

    def _reset(self):
        # Reset restriction counters for all agents
        self._agent_restrictions = {agent_id: 0 for agent_id in self.agents}

        self.S_t = {
            "water": max(EPS, self.rng.lognormal(np.log(self.water_init), 0.05)),
        }

        obs = {agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents}
        return obs

    def _is_terminated(self) -> bool:
        return self._t >= self.horizon

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        action = float(np.asarray(action).item())
        water_norm = S_t["water"] / self.max_water
        u = action * water_norm
        return u

    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        # quota-like violation and resource-level ban
        quota = max(0.0, u_i - min(self.m.fixed_quota, self.m.prop_quota * S_t["water"] / self.max_water))
        restriction = float(S_t["water"] / self.max_water < self.m.min_stock) * u_i
        v = float(quota + restriction)
        return v

    def penalty(self) -> SupportsFloat:
        return self.m.fine_amount

    def transition_kernel(
        self, A_t: MultiAgentDict, S_t: dict[str, float]
    ) -> dict[str, float]:
        water = self.S_t["water"]

        # compute agents' desired usage (same as before)
        desired = {
            agent_id: self.intrinsic_utility(agent_id=agent_id, action=A_t[agent_id], S_t=S_t)
            for agent_id in self.agents
        }
        total_desired = sum(desired.values())
        scale = min(1.0, water / max(EPS, total_desired))
        usage = self.max_water * sum(desired[agent_id] * scale for agent_id in self.agents)

        # If Raven integration is enabled, try to run Raven (periodically) and read the
        # reservoir stage CSV to obtain a model-driven water state. If anything fails,
        # fall back to the simple replenishment used previously.
        water_next: float | None = None
        if self.use_raven and self._t > 0 and (self._t % self.raven_freq == 0):
            try:
                overrides = self._actions_to_overrides(A_t)
                overrides["usage"] = usage
                self._run_raven(key=self.key, overrides=overrides)
                lake_level = self._read_raven_reservoir_stage(self.raven_stage_col)
                if lake_level is not None:
                    # Keep value within bounds
                    water_next = float(np.clip(lake_level, 0.0, self.max_water))
            except Exception:
                logger.exception("Raven integration failed; falling back to internal dynamics")

        # fallback if Raven wasn't used or failed
        if water_next is None:
            water_next = water + self.dt * (self.inflow_rate) - usage
            water_next = float(np.clip(water_next, 0.0, self.max_water))

        return {"water": water_next}

    def _run_raven(self, key: str, overrides: Dict[str, Dict[str, float]] | None = None) -> None:
        """Wrapper that forwards overrides into the prepared run and executes Raven."""
        # No-op if Raven integration disabled
        if not self.use_raven:
            return

        # Prepare run with overrides
        run_dir = self._prepare_raven_run(key=key, overrides=overrides)

        # Build command pointing to the local copy
        model_base = os.path.join(run_dir, "ohms_canshield")
        out_dir = os.path.join(run_dir, "3_Model_output")
        os.makedirs(out_dir, exist_ok=True)

        self.raven_cmd = self.raven_cmd or "raven"
        cmd = f"{self.raven_cmd} {model_base} -o {out_dir}"
        logger.info("Running Raven: %s (cwd=%s)", cmd, run_dir)
        # subprocess.run(cmd, shell=True, check=True, cwd=run_dir)

    def _prepare_raven_run(self, overrides: Dict[str, Dict[str, float]] | None = None) -> str:
        """Copy the Raven input folder into a temporary directory and apply simple overrides.

        Currently supports overrides for ':InitialReservoirStage' via
        overrides={'InitialReservoirStage': {'29012877': 480.0, ...}}
        Returns the path to the run directory containing the model files.
        """
        src = os.path.abspath(os.path.join(self.raven_cwd, "2_Raven"))
        if not os.path.isdir(src):
            # fall back to raven_cwd itself if layout differs
            src = os.path.abspath(self.raven_cwd)

        # If overrides provided, use a cache directory keyed by the overrides to
        # avoid repeatedly copying and editing inputs.
        if overrides:
            cache_root = os.path.abspath(os.path.join(self.raven_cwd, ".cache", "prepared_runs"))
            os.makedirs(cache_root, exist_ok=True)
            # create a stable key from overrides dict
            try:
                key = hashlib.sha256(json.dumps(overrides, sort_keys=True).encode("utf-8")).hexdigest()
            except Exception:
                # fallback to repr-based key
                key = hashlib.sha256(repr(overrides).encode("utf-8")).hexdigest()

            cached = os.path.join(cache_root, key)
            if os.path.isdir(cached):
                return cached

            run_root = os.path.join(cache_root, key)
            os.makedirs(run_root, exist_ok=True)
        else:
            run_root = tempfile.mkdtemp(prefix="raven_run_")

        # copy contents into run_root
        try:
            for entry in os.listdir(src):
                s = os.path.join(src, entry)
                d = os.path.join(run_root, entry)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        except Exception:
            logger.exception("Failed copying Raven inputs from %s", src)
            raise

        # Apply overrides to .rvc if requested
        if overrides:
            rvc_path = os.path.join(run_root, "ohms_canshield.rvc")
            if os.path.exists(rvc_path) and "InitialReservoirStage" in overrides:
                try:
                    with open(rvc_path, "r", encoding="utf-8") as fh:
                        lines = fh.readlines()

                    new_lines = []
                    for ln in lines:
                        if ln.strip().startswith(":InitialReservoirStage"):
                            parts = ln.split()
                            if len(parts) >= 3:
                                rid = parts[1]
                                if rid in overrides["InitialReservoirStage"]:
                                    val = overrides["InitialReservoirStage"][rid]
                                    new_lines.append(f":InitialReservoirStage {rid} {val}\n")
                                    continue
                        new_lines.append(ln)

                    with open(rvc_path, "w", encoding="utf-8") as fh:
                        fh.writelines(new_lines)
                except Exception:
                    logger.exception("Failed to apply overrides to %s", rvc_path)

        return run_root

    def _read_raven_reservoir_stage(self, column_name: str) -> Optional[float]:
        """Read the ReservoirStages CSV produced by Raven and return the value for
        the configured time index (self._t). Returns None on failure."""
        csv_path = os.path.join(self.raven_cwd, "3_Model_output", "ohms_canshield_ReservoirStages.csv")
        if not os.path.exists(csv_path):
            logger.warning("Raven output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline='') as fh:
                reader = csv.DictReader(fh)
                # DictReader keeps original fieldnames; normalize them by stripping
                rows = []
                for r in reader:
                    rows.append({k.strip(): v for k, v in r.items()})

            if not rows:
                return None

            # select row by time index; clamp to last available row
            idx = int(min(max(0, self._t), len(rows) - 1))
            row = rows[idx]
            if column_name not in row:
                # Try to find a close match if spacing or trailing chars exist
                keys = {k: k for k in row.keys()}
                match = next((k for k in keys if k.startswith(column_name)), None)
                if match is None:
                    logger.warning("ReservoirStages CSV missing column '%s' (available: %s)", column_name, list(row.keys()))
                    return None
                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw != "" else None
        except Exception:
            logger.exception("Failed to read Raven ReservoirStages CSV")
            return None


    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}

    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        water_norm = S_t["water"] / self.max_water

        # Restriction status normalized to [0,1]
        restriction_remaining = 0.0
        if self.m.ban_period > 0:
            restriction_remaining = self._agent_restrictions.get(agent_id, 0) / self.m.ban_period

        effective_quota = min(self.m.fixed_quota, self.m.prop_quota * water_norm)
        no_water_zone = float(water_norm < self.m.min_stock)

        return np.array([
            water_norm, 0.0, restriction_remaining, effective_quota, no_water_zone
        ], dtype=np.float32)

    def _is_restricted(self, agent_id: AgentID) -> bool:
        return self._agent_restrictions.get(agent_id, 0) > 0


def read_raven_stage_at(index: int, column_name: str = "Belwood_Lake", raven_root: str = "raven") -> Optional[float]:
    """Module-level helper: read the ReservoirStages CSV at a given row index.

    This avoids constructing the environment or re-running Raven and is useful
    for small-step checks or unit tests.
    """
    csv_path = os.path.join(raven_root, "3_Model_output", "ohms_canshield_ReservoirStages.csv")
    if not os.path.exists(csv_path):
        logger.warning("Raven output not found: %s", csv_path)
        return None

    try:
        with open(csv_path, newline='') as fh:
            reader = csv.DictReader(fh)
            rows = [ {k.strip(): v for k, v in r.items()} for r in reader ]

        if not rows:
            return None

        idx = int(min(max(0, index), len(rows) - 1))
        row = rows[idx]
        if column_name not in row:
            match = next((k for k in row.keys() if k.startswith(column_name)), None)
            if match is None:
                logger.warning("ReservoirStages CSV missing column '%s' (available: %s)", column_name, list(row.keys()))
                return None
            column_name = match

        raw = row.get(column_name, "")
        return float(raw) if raw != "" else None
    except Exception:
        logger.exception("Failed to read Raven ReservoirStages CSV at index %s", index)
        return None

    def _decrement_restriction(self, agent_id: AgentID) -> None:
        if self._agent_restrictions.get(agent_id, 0) > 0:
            self._agent_restrictions[agent_id] -= 1

    def _apply_restriction(self, agent_id: AgentID) -> None:
        if self.m.ban_period > 0:
            self._agent_restrictions[agent_id] = self.m.ban_period
