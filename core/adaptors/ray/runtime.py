import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import ray
import torch

DeviceType = Literal["cpu", "cuda", "mps"]


@dataclass
class RayRuntimeConfig:
    device: DeviceType = "cpu"

    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    omp_threads: int = 1
    disable_mps: bool = True
    disable_cuda: bool = True

    logging_level: str = "ERROR"
    runtime_env: Optional[Dict[str, Any]] = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def _apply_env_vars(self):
        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["RLLIB_NUM_GPUS"] = "0"
            os.environ["USE_CUDA"] = "0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

            if self.disable_mps:
                os.environ["RAY_USE_MPS"] = "0"

        if self.num_gpus is not None:
            os.environ["RLLIB_NUM_GPUS"] = str(self.num_gpus)

        if self.omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)

        torch.set_default_device(self.device)

    def initialize(self):
        self._apply_env_vars()
        ray.init(
            ignore_reinit_error=True,
            logging_level=self.logging_level,
            runtime_env=self.runtime_env,
            local_mode=True,  # for debugging only
            log_to_driver=False,
            include_dashboard=False,
            _system_config={
                "metrics_report_interval_ms": 0,
                "enable_metrics_collection": False,
            },
            **self.init_kwargs,
        )


class RayRuntime:
    _initialized = False

    @classmethod
    def ensure_initialized(cls, cfg: RayRuntimeConfig):
        if not ray.is_initialized():
            cfg.initialize()
            cls._initialized = True
