import csv
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from time import time
from typing import Optional, SupportsFloat

import numpy as np
from gymnasium.core import ActType
from ray.rllib.utils.typing import AgentID, MultiAgentDict
from datetime import datetime, timedelta

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

logger = logging.getLogger(__name__)

EPS = 1e-8


class WaterRegulatedEdHsEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        use_raven: bool = True,
        raven_cmd: Optional[str] = None,
        raven_cwd: str = "raven",
        raven_freq: int = 1,
        raven_stage_col: str = "Belwood_Lake",
        raven_streamflow_col: str = "Belwood_Lake (res. inflow) [m3/s]",
        **kwargs,
    ):
        super().__init__(**kwargs)

        
        self.inflow_rate = ecology_cfg.get("inflow_rate", 1.0)
        self.dt = ecology_cfg.get("dt", 1.0)

        # max_pull_fraction is physical extraction capacity, not regulation.
        # It approximates the pump / intake capacity of one agent as a fraction
        # of current streamflow.
        #
        # Rough interpretation:
        # 0.0001 -> very small pump / individual farm
        # 0.0005 -> large farm / small irrigation setup
        # 0.001  -> irrigation user / small utility
        # 0.005  -> irrigation district / municipality / industrial user
        # 0.01   -> large municipal or industrial intake
        # 0.05   -> regional-scale extraction; usually unrealistic for one agent
        #
        # This is not the legal quota. The mechanism parameters fixed_quota and
        # prop_quota define regulation. max_pull_fraction only defines how much
        # the agent could physically request before regulation is applied.
        # large scale farmer - 
        # increase water pull when there is percipitation otherwise consume water
        self.max_pull_fraction = ecology_cfg.get("max_pull_fraction", 0.005)

        # TODO move this to automatic parsing from raven files (default vals for Belwood Lake)
        # self.streamflow_init = ecology_cfg.get("streamflow_init", 124.724) # ohms_canshield.rvh
        self.streamflow_init = ecology_cfg.get("streamflow_init", 12.4724)
        self.streamflow_init_sigma = ecology_cfg.get("streamflow_init_sigma", 0.05)

        # Ontario requires a Permit to Take Water above 50,000 L/day.
        # 50,000 L/day = 50 m3/day ≈ 0.00058 m3/s.
        # 0.01 m³/s = 10 L/s      → small user
        # 0.05 m³/s = 50 L/s      → moderate
        # 0.1  m³/s = 100 L/s     → substantial
        # 0.3  m³/s = 300 L/s     → very large demand
        # precip [mm/day]

        # West_Montrose [m3/s]
        # West_Montrose (observed) [m3/s]
        
        self.min_required_demand_m3s = ecology_cfg.get("min_required_demand_m3s", 0.02)
        self.lake_elevation = ecology_cfg.get("lake_elevation", 420.41)
        self.max_depth = ecology_cfg.get("max_depth", 11.0)

        # TODO move this to the mechanism
        self.underuse_penalty_scale = ecology_cfg.get("underuse_penalty_scale", 1.0)
        self.underuse_penalty_power = ecology_cfg.get("underuse_penalty_power", 2.0)

        self.use_raven = use_raven
        self.raven_cmd = raven_cmd or "raven"
        self.raven_cwd = raven_cwd
        self.raven_freq = max(0, raven_freq)
        self.raven_stage_col = raven_stage_col
        self.raven_streamflow_col = raven_streamflow_col

        self.key = hashlib.sha256(str(time()).encode("utf-8")).hexdigest()
        self.run_root: Optional[str] = None

        self.obs_map = [
            "reservoir_level_norm",
            "usage_norm",
            "effective_quota",
            "total_usage_norm",
        ]

    @override(MultiAgentRegulatedEnv)
    def _reset(self):
        streamflow0 = max(
            EPS,
            self.rng.lognormal(
                mean=np.log(self.streamflow_init),
                sigma=self.streamflow_init_sigma,
            ),
        )

        self.S_t = {
            "reservoir_level_norm": 1.0,
            "streamflow": streamflow0,
            "last_usage": 0.0,
            "total_usage": 0.0,
        }

        obs = {
            agent_id: self.observation(agent_id, self.S_t)
            for agent_id in self.agents
        }
        return obs

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        return self.horizon is not None and (self._t + 1) >= self.horizon

    @override(MultiAgentRegulatedEnv)
    def intrinsic_utility(self, A_t: dict[AgentID, ActType]) -> MultiAgentDict:
        # Q: Water allocation is indexed here as percentage of the total active storage capacity.
        # Q: The volume of water pumps depends on the max capacity
        # Q: perhaps we should integrate the maximum pump capacity ?
            # Example solution :
            # max_pump_capacity = 0.05
            # max_step_extraction = self.max_water * max_pump_capacity
            # return {
            #     agent_id: (action.item() * max_step_extraction) / self.max_water
            #     for agent_id, action in A_t.items()
            # }

        # fixed level - dependent on the tater level - hyperparameter to slide between both
        # lake height relative to sea level
        # water_lake = lake_level - elevation (.rvh) / 11 (.rvh)

        # Action is the requested fraction of max allowable streamflow extraction.
        # constant value - constant demand > threshold * stream flow. then limit by ability to pull 
        # water. 

        return {
            agent_id: float(action.item()) * self.max_pull_fraction
            for agent_id, action in A_t.items()
        }

    @override(MultiAgentRegulatedEnv)
    def violation_signal(
        self,
        u_i: SupportsFloat,
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:
        reservoir_level_norm = self.S_t["reservoir_level_norm"]
        current_streamflow = self.S_t["streamflow"]
        
        # What the agent wants based on stable depand capacity
        # cap it and then if it goes 
        requested_m3s = u_i * current_streamflow #m3s = cubic meters / second
        

        # requested = need_crop_value - percipitation
        # read stream flows but do not use them in multiplication


        # allowed_fraction must be normalized by number of agents !
        allowed_fraction = min(
            self.mechanism.fixed_quota,
            self.mechanism.prop_quota * reservoir_level_norm,
        )
        # but do not multiply by reservoir.if the allows is below reservoir quoa level, no allowd
        allowed_m3s = allowed_fraction * current_streamflow

        # TODO : issue if the total pull of agents together exceeds the allowed m3s
        # Q : normalize the allowed_m3s by the number of agents 
        # crop yield - for wtv square meter of corn if the - if you water the crop wtv 
        # if the crop is watered - positive reinforement, if its not watered enough - negative
        # if its overwatered 
        # reward is based on crop yield.
        # corn, soybeans, potatoes

        quota_violation_m3s = max(0.0, requested_m3s - allowed_m3s)
        quota_penalty = min(1.0, self.mechanism.fine_amount * (quota_violation_m3s / max(EPS, allowed_m3s)))
        # quota_penalty = 0
        # Q: No water zone creates a binary cliff of good and bad behaviour - messes up gradients
        # Q : If the basin is below the safe stock then choosing an action of 0 creates no penalty
        # no_water_zone = float(water_norm < self.mechanism.min_stock)
        # stock_penalty = no_water_zone * float(u_i)

        # penalty on not pulling what you need - huge
        # stock penalties fine
        # farmer and power generation station : hydroelectric generation station - different penalty

        # do not need this.
        # For logging purposes only
        # underuse_severity_m3s = max(0.0, self.min_required_demand_m3s - requested_m3s)

        # underuse_ratio = (
        #     underuse_severity_m3s
        #     / max(EPS, self.min_required_demand_m3s)
        # )

        # underuse_penalty = min(
        #     1.0,
        #     self.underuse_penalty_scale
        #     * (
        #         underuse_ratio
        #         ** self.underuse_penalty_power
        #     ),
        # )

        # # Ecological shortage is diagnostic only for now.
        # stock_shortage_severity = max(
        #     0.0,
        #     self.mechanism.min_stock - reservoir_level_norm,
        # )


        total_penalty = min(
            1.0,
            quota_penalty + underuse_penalty,
        )

        self._update_infos(
            key="requested_m3s",
            values={agent_id: requested_m3s},
        )
        self._update_infos(
            key="allowed_m3s",
            values={agent_id: allowed_m3s},
        )
        self._update_infos(
            key="quota_violation_m3s",
            values={agent_id: quota_violation_m3s},
        )
        self._update_infos(
            key="quota_penalty",
            values={agent_id: quota_penalty},
        )
        self._update_infos(
            key="underuse_severity_m3s",
            values={agent_id: underuse_severity_m3s},
        )
        self._update_infos(
            key="underuse_penalty",
            values={agent_id: underuse_penalty},
        )
        self._update_infos(
            key="stock_shortage_severity",
            values={agent_id: stock_shortage_severity},
        )
        self._update_infos(
            key="total_penalty",
            values={agent_id: total_penalty},
        )

        return total_penalty

    @override(MultiAgentRegulatedEnv)
    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, float],
    ) -> dict[str, float]:
        reservoir_level_norm = S_t["reservoir_level_norm"]
        streamflow_next = S_t["streamflow"]

        realized_usage, total_usage, usage_scale = self._compute_usage_metrics(A_t=A_t)
        total_usage_norm = sum(realized_usage.values())

        self._update_infos(key="usage", values=realized_usage)
        self._update_infos(key="total_usage", values=total_usage)
        self._update_infos(key="usage_scale", values=usage_scale)

        reservoir_level_next_norm = None

        if self.use_raven and self.raven_freq > 0 and (self._t % self.raven_freq == 0):
            try:
                # Q : what is the total_usage unit ?
                # Q : what does the raven model output - what is the unit of the lake_level ?
                self._run_raven(usage=total_usage)
                lake_level = self._read_raven_reservoir_stage(self.raven_stage_col)
                streamflow = self._read_raven_streamflow(self.raven_streamflow_col)

                print(
                    f"[RAVEN OUTPUT] t={self._t}",
                    f"lake_level={lake_level}",
                    f"streamflow={streamflow}",
                )
                # Q : Problem - we are clipping lake_level to normalized capacity percentage
                # Q : what is the unit of lake_level ? can we establish a maximum quantity ? 
                    # like max elevation level ?
                # stream flow - normalize reservoir storage 
                # rvc file - profile of the r
                # max_depth - 
                # reservoir stage vs time
                # stream flow vs time (discharge) vs time 
                # optional 
                # stream temperature 
                # water quality
                if lake_level is not None:
                    # reservoir_level_next_norm = (
                    #     lake_level - self.lake_elevation
                    # ) / max(EPS, self.max_depth)
                    # reservoir_level_next_norm = float(np.clip(reservoir_level_next_norm, 0.0, 1.0))
                    reservoir_stage_m = float(lake_level)

                    reservoir_depth_m = max(
                        0.0,
                        reservoir_stage_m - self.lake_elevation,
                    )

                    reservoir_level_next_norm = (
                        reservoir_depth_m
                        / max(EPS, self.max_depth)
                    )

                    reservoir_level_next_norm = float(
                        np.clip(
                            reservoir_level_next_norm,
                            0.0,
                            1.0,
                        )
                    )
                    self._update_infos(
                        key="reservoir_stage_m",
                        values=reservoir_stage_m,
                    )

                    self._update_infos(
                        key="reservoir_depth_m",
                        values=reservoir_depth_m,
                    )

                    self._update_infos(
                        key="reservoir_level_norm",
                        values=reservoir_level_next_norm,
                    )

                    self._update_infos(
                        key="max_depth_m",
                        values=self.max_depth,
                    )

                    self._update_infos(
                        key="total_usage_m3s",
                        values=total_usage,
                    )
                if streamflow is not None:
                    streamflow_next = max(EPS, streamflow)

                    self._update_infos(
                        key="streamflow_m3s",
                        values=streamflow_next,
                    )

            except Exception:
                logger.exception("Raven integration failed; falling back to internal dynamics")

        if reservoir_level_next_norm is None:
            reservoir_level_next_norm = (
                reservoir_level_norm
                + self.dt * (self.inflow_rate / self.streamflow_init)
                - (total_usage / self.streamflow_init)
            )
            reservoir_level_next_norm = float(np.clip(reservoir_level_next_norm, 0.0, 1.0))

        new_state = {
            "reservoir_level_norm": max(EPS, reservoir_level_next_norm),
            "streamflow": streamflow_next,
            "last_usage": total_usage_norm,
            "total_usage": total_usage_norm,
        }

        print(
            f"[STATE UPDATE] t={self._t}",
            f"old_streamflow={self.S_t['streamflow']:.4f}",
            f"new_streamflow={new_state['streamflow']:.4f}",
            f"reservoir={new_state['reservoir_level_norm']:.4f}",
        )

        self.S_t = new_state
        return self.S_t

    def _observation(self, agent_id: AgentID, S_t: dict[str, float]):
        reservoir_level_norm = S_t["reservoir_level_norm"]
        usage_norm = S_t.get("last_usage", 0.0)
        total_usage_norm = S_t.get("total_usage", 0.0)

        effective_quota = min(
            self.mechanism.fixed_quota,
            self.mechanism.prop_quota * reservoir_level_norm,
        )
        return np.array(
            [
                reservoir_level_norm,
                usage_norm,
                effective_quota,
                total_usage_norm,
            ],
            dtype=np.float32,
        )

    def _compute_usage_metrics(
        self,
        A_t: dict[AgentID, ActType],
    ) -> tuple[dict[AgentID, float], float, float]:
        streamflow = self.S_t["streamflow"]

        desired_norm = self.intrinsic_utility(A_t=A_t)
        
        # Q : the pull of several agents at the same time will affect each other ?
        # max_water = maximum flow rate 
        # water = stream - cubic meters/ s 
        # what we are pulling is a stream rate
        # amount of flow rate
        # precipitation

        total_desired_norm = sum(desired_norm.values())
        scale = min(1.0, 1.0 / max(EPS, total_desired_norm))

        realized_usage = {
            agent_id: desired_norm[agent_id] * scale
            for agent_id in self.agents
        }

        total_usage_norm = sum(realized_usage.values())
        total_usage = total_usage_norm * streamflow

        print(
            f"\n[USAGE] t={self._t}",
            f"streamflow={streamflow:.4f}",
            f"actions={[float(a.item()) for a in A_t.values()]}",
            f"desired_norm_by_agent={desired_norm}",
            f"desired_norm_total={total_desired_norm:.4f}",
            f"scale={scale:.4f}",
            f"total_usage_norm={total_usage_norm:.4f}",
            f"total_usage_m3s={total_usage:.4f}",
        )
        return realized_usage, total_usage, scale

    def _prepare_raven_run(self) -> str:
        src = os.path.abspath(self.raven_cwd)

        if self.run_root is not None:
            return self.run_root

        cache_root = os.path.abspath(
            os.path.join(self.raven_cwd, ".cache", "prepared_runs")
        )
        os.makedirs(cache_root, exist_ok=True)

        run_root = os.path.join(cache_root, self.key)
        os.makedirs(run_root, exist_ok=True)

        for entry in os.listdir(src):
            if entry == ".cache":
                continue
            s = os.path.join(src, entry)
            d = os.path.join(run_root, entry)

            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        self.run_root = run_root
        return run_root

    def _append_extraction_to_rvt(self, run_dir: str, usage: float) -> int:
        rvt_path = Path(run_dir) / "input" / "Extraction.rvt"
        usage = -1.0 * float(f"{usage:.6f}")
        print(
            f"[RAVEN INPUT] t={self._t}",
            f"usage_m3s={usage:.6f}",
        )

        lines = rvt_path.read_text(encoding="utf-8").splitlines()

        header_idx = next(
            i for i, line in enumerate(lines)
            if line.strip().startswith("1980-01-01 00:00:00")
        )

        end_idx = next(
            i for i in range(header_idx + 1, len(lines))
            if lines[i].strip() in [":EndObservationData", ":EndData"]
        )

        old_values = [
            line for line in lines[header_idx + 1:end_idx]
            if line.strip() and not line.strip().startswith("#")
        ]

        values = old_values + [f"\t{usage}"]
        n_values = len(values)

        lines[header_idx] = f"\t1980-01-01 00:00:00\t1\t{n_values}"

        new_lines = (
            lines[: header_idx + 1]
            + values
            + [":EndData"]
        )

        rvt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        return n_values


    def _patch_raven_end_date(self, run_dir: str, n_days: int) -> None:
        rvi_path = Path(run_dir) / "2_Raven" / "ohms_canshield.rvi"

        lines = rvi_path.read_text(encoding="utf-8").splitlines()

        start = datetime(1980, 1, 1)
        end = start + timedelta(days=n_days)
        end_str = end.strftime("%Y-%m-%d 00:00:00")

        patched = []
        for line in lines:
            if line.strip().startswith(":EndDate"):
                patched.append(f":EndDate         {end_str}")
            else:
                patched.append(line)

        rvi_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


    def _run_raven(self, usage: float) -> None:
        if not self.use_raven:
            return

        run_dir = self._prepare_raven_run()

        n_days = self._append_extraction_to_rvt(run_dir=run_dir, usage=usage)
        self._patch_raven_end_date(run_dir=run_dir, n_days=n_days)

        extraction_path = Path(run_dir) / "input" / "Extraction.rvt"

        # Debugging
        print("=" * 80)
        print("Raven cwd:", run_dir)
        print("RVI exists:", (Path(run_dir) / "2_Raven" / "ohms_canshield.rvi").exists())
        print("Extraction exists:", extraction_path.exists())
        print("Raven duration days:", n_days)

        # Debugging
        # if extraction_path.exists():
        #     print("\n--- Extraction.rvt contents ---")
        #     print(extraction_path.read_text())
        #     print("--- End Extraction.rvt ---\n")

        out_dir = "3_Model_output"
        os.makedirs(Path(run_dir) / out_dir, exist_ok=True)

        subprocess.run(
            [self.raven_cmd, "2_Raven/ohms_canshield", "-o", out_dir],
            cwd=run_dir,
            check=False,
        )

    def _read_raven_reservoir_stage(self, column_name: str) -> Optional[float]:
        if self.run_root is None:
            return None

        csv_path = os.path.join(
            self.run_root,
            "3_Model_output",
            "ohms_canshield_ReservoirStages.csv",
        )

        if not os.path.exists(csv_path):
            logger.warning("Raven output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k.strip(): v for k, v in row.items()} for row in reader]

            if not rows:
                return None

            idx = int(min(max(0, self._t), len(rows) - 1))
            row = rows[idx]

            if column_name not in row:
                match = next((k for k in row.keys() if k.startswith(column_name)), None)

                if match is None:
                    logger.warning(
                        "ReservoirStages CSV missing column '%s'. Available: %s",
                        column_name,
                        list(row.keys()),
                    )
                    return None

                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw != "" else None

        except Exception:
            logger.exception("Failed to read Raven ReservoirStages CSV")
            return None
        
    def _read_raven_streamflow(self, column_name: str) -> Optional[float]:
        if self.run_root is None:
            return None

        csv_path = os.path.join(
            self.run_root,
            "3_Model_output",
            "ohms_canshield_Hydrographs.csv",
        )

        if not os.path.exists(csv_path):
            logger.warning("Raven hydrograph output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k.strip(): v for k, v in row.items()} for row in reader]

            if not rows:
                return None

            idx = int(min(max(0, self._t), len(rows) - 1))
            row = rows[idx]

            print(
                "READ STREAMFLOW",
                "t=", self._t,
                "idx=", idx,
                "col=", column_name,
                "raw=", row.get(column_name),
            )

            if column_name not in row:
                match = next(
                    (k for k in row.keys() if k.strip() == column_name.strip()),
                    None,
                )

                if match is None:
                    logger.warning(
                        "Hydrographs CSV missing column '%s'. Available: %s",
                        column_name,
                        list(row.keys()),
                    )
                    return None

                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw not in ["", "---"] else None

        except Exception:
            logger.exception("Failed to read Raven Hydrographs CSV")
            return None