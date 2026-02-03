import logging
from pathlib import Path
from typing import SupportsFloat
from core.annotations import override

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
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from examples.water_usage.regulated_env_raven import WaterRegulatedRavenEnv
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# Numerical stability constant
EPS = 1e-8


class WaterRegulatedEdHsEnv(WaterRegulatedRavenEnv):
    def _prepare_raven_run(self, key, overrides: Dict[str, Dict[str, float]] | None = None) -> str:
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
        # Need to create new run_root every horizon.
        if self.run_root is None:
            cache_root = os.path.abspath(os.path.join(self.raven_cwd, ".cache", "prepared_runs"))
            os.makedirs(cache_root, exist_ok=True)
            # create a stable key from overrides dict

            cached = os.path.join(cache_root, key)
            if os.path.isdir(cached):
                return cached

            run_root = os.path.join(cache_root, key)
            os.makedirs(run_root, exist_ok=True)
            self.run_root = run_root
        
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

        
        
        # Update .rvt file to reflect pumping from agent - write a separate .rvt file to be read in
        if overrides:
            rvt_path = os.path.join(self.run_root, "Extraction.rvt")

            if os.path.exists(rvt_path) and "usage" in overrides:
                try:
                    p = Path(rvt_path)
                    # Ensure parent exists
                    p.parent.mkdir(parents=True, exist_ok=True)
                    usage = overrides["usage"]
                    usage = -1 * float(f"{usage:.6f}") # Raven expects negative for extraction

                    if not p.exists():
                        p.write_text(f"{usage}\n", encoding="utf-8")
                        return

                    content = p.read_text(encoding="utf-8")
                    with p.open("a", encoding="utf-8") as fh:
                        if not content.endswith("\n"):
                            fh.write("\n")
                        fh.write(f"     {usage}\n")

                        if self._t % self.horizon == 0:
                            logger.info("Add ending")
                            fh.write("\n")
                            fh.write(":EndObservationData\n")
                except Exception:
                    logger.exception("Failed to apply overrides to %s", rvt_path)

        return self.run_root