"""Water regulated environment.

Implements conservative RVI and Extraction updates and a hydro-based reward.
Provides a single `WaterRegulatedEnv` class (water-only, no fish/algae).
"""

from __future__ import annotations

import logging
import os
import re
import csv
import subprocess
from datetime import datetime, timedelta
from typing import SupportsFloat

import numpy as np
from gymnasium.core import ActType
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

EPS = 1e-8


class WaterRegulatedEnv(MultiAgentRegulatedEnv):
	"""Water-only regulated environment with optional Raven execution."""

	def __init__(self, *, ecology_cfg: dict | None = None, **kwargs):
		super().__init__(**kwargs)
		self.ecology_cfg = ecology_cfg or {}

		# configurable paths
		self.rvi_path = kwargs.get("rvi_path", "raven/2_Raven/ohms_canshield.rvi")
		self.extraction_rvt = kwargs.get("extraction_rvt", "raven/input/Extraction.rvt")
		# optional template to reset extraction file each timestep (prevents cumulative edits)
		self.extraction_template = kwargs.get("extraction_template", self.extraction_rvt + ".template")
		self.hydro_csv = kwargs.get("hydro_csv", "raven/3_Model_output/hydrographs.csv")

		# params
		self.dt = float(self.ecology_cfg.get("dt", 1.0))
		self.initial_water_level = float(self.ecology_cfg.get("initial_water_level", self.ecology_cfg.get("water_init", 1.0)))
		self.demand = float(self.ecology_cfg.get("demand", 0.1))

		self._agent_bans = {agent_id: 0 for agent_id in getattr(self, "agents", [])}

		# Raven options
		self.raven_conda_env = kwargs.get("raven_conda_env", "bilevel-2")
		self.auto_run_raven = bool(kwargs.get("auto_run_raven", False))
		self.raven_output_dir = kwargs.get("raven_output_dir", "results/raven_run_output")
		# watershed storage CSV (tracks total storage)
		self.watershed_storage_csv = kwargs.get("watershed_storage_csv", "raven/3_Model_output/ohms_canshield_WatershedStorage.csv")

		# Try to update RVI end date conservatively during init
		try:
			self._update_rvi_end_date(self.rvi_path)
		except Exception:
			logger.debug("_update_rvi_end_date failed during init", exc_info=True)

		# Optionally run Raven once after RVI update (if explicitly requested)
		if self.auto_run_raven and kwargs.get("run_after_rvi", False):
			try:
				self._run_raven(self.raven_output_dir)
			except Exception:
				logger.debug("_run_raven failed during init", exc_info=True)

	def _reset(self):
		self._agent_bans = {agent_id: 0 for agent_id in self.agents}
		self.S_t = {
			"water_available": max(EPS, float(self.ecology_cfg.get("water_init", 1.0))),
			"flow_rate": float(self.ecology_cfg.get("flow_rate_init", 0.0)),
		}
		# Return the full observation (base observation concatenated with mechanism vector)
		# by using the public observation() method so shapes match the configured
		# observation_space (base_dim + mechanism_dim).
		return {agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents}

	def _is_terminated(self) -> bool:
		return getattr(self, "_t", 0) >= int(getattr(self, "horizon", 0) or 0)

	def intrinsic_utility(self, agent_id: AgentID, action: ActType, S_t: dict) -> SupportsFloat:
		# Simple demand-based utility (can be expanded later)
		return float(self.demand)

	def violation_signal(self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict) -> SupportsFloat:
		# quota violation + low-stock ban signal
		quota_val = min(getattr(self.m, "fixed_quota", 0.0), getattr(self.m, "prop_quota", 0.0) * S_t.get("water_available", 0.0) / max(self.initial_water_level, EPS))
		quota = max(0.0, float(u_i) - quota_val)
		ban = float(S_t.get("water_available", 0.0) / max(self.initial_water_level, EPS) < getattr(self.m, "min_stock", 0.0)) * float(u_i)
		return float(quota + ban)

	def penalty(self) -> SupportsFloat:
		return float(getattr(self.m, "fine_amount", 0.0))

	def transition_kernel(self, *, A_t: MultiAgentDict, S_t: dict, **kwargs) -> dict:
		# compute available water and flow
		water_available = float(S_t.get("water_available", self.S_t.get("water_available", 0.0)))
		flow_rate = float(S_t.get("flow_rate", self.S_t.get("flow_rate", 0.0)))

		initial_level = getattr(self, "initial_water_level", water_available)
		water_norm = water_available / max(initial_level, EPS)

		# desired extraction per agent (currently intrinsic_utility)
		desired = {agent_id: float(self.intrinsic_utility(agent_id=agent_id, action=A_t.get(agent_id, 0.0), S_t=S_t)) for agent_id in self.agents}
		total_desired = sum(desired.values())
		scale = min(1.0, water_norm / max(EPS, total_desired)) if total_desired > 0 else 0.0 #TODO: remove the scale as this doesn't make sense here for the agents
		H = sum(desired[a] * scale for a in self.agents)

		# simple physical dynamics
		water_available_next = max(0.0, water_available + flow_rate - H)
		flow_rate_next = max(0.0, 0.9 * flow_rate) #TODO: this might not make sense, would need to double check


		# update extraction RVT (conservative edit of a single numeric line)
		updated = False
		try:
			updated = self._update_extraction_file(self.extraction_rvt, H)
		except Exception:
			logger.debug("_update_extraction_file failed", exc_info=True)

		# optionally run Raven when extraction file was updated
		if updated and getattr(self, "auto_run_raven", False):
			try:
				self._run_raven(self.raven_output_dir)
			except Exception:
				logger.debug("_run_raven failed after extraction update", exc_info=True)

		# If Raven produced a WatershedStorage.csv, update water_available from its
		# 'total' column at the current timestep. This is conservative: only
		# override if the file exists and the value parses correctly.
		try:
			storage_val = self._read_watershed_storage(self.watershed_storage_csv)
			if storage_val is not None:
				# storage_val may be in same units as water_available; set directly
				water_available_next = max(0.0, float(storage_val))
		except Exception:
			logger.debug("_read_watershed_storage failed", exc_info=True)

		return {
			"water_available": water_available_next,
			"flow_rate": flow_rate_next,
		}

	def _run_raven(self, output_dir: str) -> None:
		if not os.path.exists(self.rvi_path):
			raise FileNotFoundError(self.rvi_path)
		rvi_dir = os.path.dirname(os.path.abspath(self.rvi_path))
		rvi_basename = os.path.splitext(os.path.basename(self.rvi_path))[0]
		os.makedirs(output_dir, exist_ok=True)

		cmd = [
			"conda",
			"run",
			"-n",
			str(self.raven_conda_env),
			"raven",
			rvi_basename,
			"-o",
			os.path.abspath(output_dir),
		]

		logger.info("Running Raven: %s (cwd=%s)", " ".join(cmd), rvi_dir)
		proc = subprocess.run(cmd, cwd=rvi_dir, capture_output=True, text=True)
		if proc.returncode != 0:
			logger.error("Raven run failed: %s", proc.stderr)
			raise RuntimeError(f"Raven failed (rc={proc.returncode})")
		logger.info("Raven run finished: %s", proc.stdout)

	def _update_rvi_end_date(self, rvi_path: str) -> None:
		"""Conservatively replace only the :EndDate line to StartDate + 2 years + horizon."""
		if not os.path.exists(rvi_path):
			return
		with open(rvi_path, "r", encoding="utf-8") as fh:
			lines = fh.readlines()

		start_date = None
		end_idx = None
		for i, line in enumerate(lines):
			s = line.strip()
			if s.startswith(":StartDate"):
				parts = s.split(None, 1)
				if len(parts) > 1:
					try:
						start_date = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
					except Exception:
						try:
							start_date = datetime.strptime(parts[-1].strip(), "%Y-%m-%d %H:%M:%S")
						except Exception:
							start_date = None
			if s.startswith(":EndDate"):
				end_idx = i

		if start_date is None or end_idx is None:
			return

		extra_days = int(getattr(self, "horizon", 0) or 0)
		new_end = start_date + timedelta(days=365 * 2 + extra_days)
		new_end_str = new_end.strftime("%Y-%m-%d 00:00:00")
		lines[end_idx] = f":EndDate         {new_end_str}\n"

		with open(rvi_path, "w", encoding="utf-8") as fh:
			fh.writelines(lines)

	def _update_extraction_file(self, extraction_path: str, total_extraction: float) -> bool:
		"""Reset extraction from a template (if available) then apply the single-step change.

		Behavior:
		- If an `extraction_template` exists, read from that template, modify only the
		  numeric line for the current timestep, and write the resulting file to
		  `extraction_path` (this prevents cumulative edits over multiple timesteps).
		- If no template is available but `extraction_path` exists, behave like the
		  previous conservative in-place edit.
		- Returns True if the target file was written, False otherwise.
		"""

		# Always use the explicit template to initiate the extraction file. If the
		# template is missing, do nothing (do not perform in-place cumulative edits).
		template_path = getattr(self, "extraction_template", None)
		if not template_path or not os.path.exists(template_path):
			# Template required
			return False
		base_path = template_path

		with open(base_path, "r", encoding="utf-8") as fh:
			content = fh.read()

		m = re.search(r"(:BasinInflowHydrograph.*?)(:EndObservationData)", content, flags=re.S | re.I)
		if not m:
			return False
		block = m.group(1)
		lines = block.splitlines()
		if len(lines) < 3:
			return False

		numeric_lines = lines[2:]
		idx = int(getattr(self, "_t", 0) or 0)
		if idx < 0:
			idx = 0
		if idx >= len(numeric_lines):
			idx = len(numeric_lines) - 1

		try:
			old_val = float(numeric_lines[idx].strip())
			new_val = old_val - float(total_extraction)
			# preserve formatting but ensure a reasonable float format
			numeric_lines[idx] = f"\t{new_val:.6f}"
		except Exception:
			return False

		new_block_lines = lines[:2] + numeric_lines
		new_block = "\n".join(new_block_lines)
		# Build new file content from the base content (template or existing file)
		new_content = content[: m.start(1)] + new_block + content[m.end(1) :]

		# Ensure destination directory exists
		os.makedirs(os.path.dirname(os.path.abspath(extraction_path)), exist_ok=True)
		with open(extraction_path, "w", encoding="utf-8") as fh:
			fh.write(new_content)
		return True

	def _compute_hydro_reward(self, hydro_csv_path: str) -> float:
		"""Compute reward = W - 0.9 * W* using hydrographs CSV at current timestep.

		Use the provided `hydro_csv_path` as the baseline (W*), and the file with
	
the same basename inside `self.raven_output_dir` as the simulated output (W).
		Finds columns that match 'west.*montros' (case-insensitive) in each file.
		The first matching column in the baseline file is W*, the first matching
		column in the simulated file is W. If either file or column is missing,
		returns 0.0 conservatively.
		"""
		# Require pandas for hydro reward computation (no csv fallback).
		import pandas as pd  # type: ignore[import]

		# Baseline (W*) is the provided hydro_csv_path. Simulated (W) is the file
		# with the same basename inside the raven output dir.
		baseline_path = hydro_csv_path
		simulated_path = os.path.join(getattr(self, "raven_output_dir", ""), os.path.basename(hydro_csv_path))

		# If either file is missing, return 0.0 conservatively.
		if not os.path.exists(baseline_path) or not os.path.exists(simulated_path):
			return 0.0

		# Read both CSVs with pandas; let pandas errors surface (no fallback).
		df_base = pd.read_csv(baseline_path)
		df_sim = pd.read_csv(simulated_path)
		cols_base = [c for c in df_base.columns if re.search(r"west.*montros", c, flags=re.I)]
		cols_sim = [c for c in df_sim.columns if re.search(r"west.*montros", c, flags=re.I)]
		if not cols_base or not cols_sim:
			return 0.0
		idx = min(int(getattr(self, "_t", 0) or 0), len(df_sim) - 1, len(df_base) - 1)
		W_star = float(df_base[cols_base[0]].iloc[idx])
		W = float(df_sim[cols_sim[0]].iloc[idx])
		return float(W - 0.9 * W_star)

	def _read_watershed_storage(self, storage_csv_path: str):
		"""Read 'total' column from WatershedStorage CSV for current timestep.

		Returns the numeric total value or None if not available/parseable.
		"""
		if not os.path.exists(storage_csv_path):
			return None
		try:
			import pandas as pd  # type: ignore[import]
		except Exception:
			pd = None

		try:
			if pd is not None:
				df = pd.read_csv(storage_csv_path)
				if 'total' not in df.columns:
					# be permissive: try lower-case or variants
					cols = [c for c in df.columns if c.lower() == 'total']
					if not cols:
						return None
					col = cols[0]
				else:
					col = 'total'
				idx = min(int(getattr(self, '_t', 0) or 0), len(df) - 1)
				val = df[col].iloc[idx]
				return float(val)
			else:
				with open(storage_csv_path, newline='') as fh:
					reader = csv.DictReader(fh)
					rows = list(reader)
					if not rows:
						return None
					keys = [k for k in rows[0].keys() if k.lower() == 'total']
					if not keys:
						return None
					idx = min(int(getattr(self, '_t', 0) or 0), len(rows) - 1)
					return float(rows[idx][keys[0]])
		except Exception:
			return None

	@override(MultiAgentRegulatedEnv)
	def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
		mean_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
		hydro_reward = 0.0
		try:
			hydro_reward = float(self._compute_hydro_reward(self.hydro_csv))
		except Exception:
			logger.debug("_compute_hydro_reward failed", exc_info=True)
		total = mean_reward + hydro_reward
		return {agent_id: total for agent_id in self.agents}

	def _observation(self, agent_id: AgentID, S_t: dict):
		"""Return base observation normalized to [0,1]-like ranges.
		Matches the fishery example shape and semantics.
		"""
		water_norm = S_t.get("water_available", 0.0) / max(self.initial_water_level, EPS)
		flow_norm = S_t.get("flow_rate", 0.0) / max(self.initial_water_level, EPS)

		# Ban status: normalized remaining ban steps (0 = not banned, 1 = just banned)
		ban_remaining = 0.0
		if getattr(self.m, "ban_period", 0) > 0:
			ban_remaining = self._agent_bans.get(agent_id, 0) / max(1, getattr(self.m, "ban_period", 1))

		# Computed signals to help learning
		effective_quota = min(getattr(self.m, "fixed_quota", 0.0), getattr(self.m, "prop_quota", 0.0) * water_norm)
		no_water_zone = float(water_norm < getattr(self.m, "min_stock", 0.0))

		observations = np.array([
			water_norm, flow_norm, ban_remaining,
			effective_quota, no_water_zone,
		], dtype=np.float32)

		return observations

	def _is_banned(self, agent_id: AgentID) -> bool:
		return self._agent_bans.get(agent_id, 0) > 0

	def _decrement_ban(self, agent_id: AgentID) -> None:
		if self._agent_bans.get(agent_id, 0) > 0:
			self._agent_bans[agent_id] -= 1

	def _apply_ban(self, agent_id: AgentID) -> None:
		if getattr(self.m, "ban_period", 0) > 0:
			self._agent_bans[agent_id] = getattr(self.m, "ban_period", 0)