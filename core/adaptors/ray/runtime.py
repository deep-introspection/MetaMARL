import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import logging

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

    ray_debug: bool = True

    # local_mode=True runs everything in a single process (needed for the Ray
    # debugger / breakpoints). It was previously hard-coded True, but on Ray
    # 2.53 that crashes at init when the driver runs inside an editable-installed
    # package: Ray auto-captures the repo as `working_dir`, which local mode
    # cannot upload ("... is not a valid URI"). Default is now False (real
    # actors; Ray ships the package to workers). Set True only for step-through
    # debugging, and run from a directory outside the repo if you do.
    local_mode: bool = False

    def _apply_env_vars(self):
        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["RLLIB_NUM_GPUS"] = "0"
            os.environ["USE_CUDA"] = "0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            if self.disable_mps:
                os.environ["RAY_USE_MPS"] = "0"
        elif self.device == "mps":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["RLLIB_NUM_GPUS"] = "0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif self.device == "cuda":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            if self.disable_cuda:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if self.num_gpus is not None:
            os.environ["RLLIB_NUM_GPUS"] = str(self.num_gpus)

        if self.omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)

        if self.ray_debug:
            os.environ["RAY_DEBUG"] = "1"

        # logging behaviour for workers
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

        torch.set_default_device(self.device)

    def initialize(self):
        self._apply_env_vars()

        # Silence noisy loggers globally
        # # logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("ray.rllib").setLevel(logging.WARNING)
        logging.getLogger("ray.tune").setLevel(logging.WARNING)
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        logging.getLogger("asyncio").setLevel(logging.ERROR)

        ray.init(
            ignore_reinit_error=True,
            logging_level=self.logging_level,
            runtime_env=self.runtime_env,
            local_mode=self.local_mode,  # see RayRuntimeConfig.local_mode
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
