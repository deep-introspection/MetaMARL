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

    # local_mode runs everything in a single process — handy for step-through
    # debugging, but on Ray >= 2.5 it CANNOT upload an auto-captured working_dir,
    # so launching from inside the editable-installed repo raises
    # "... is not a valid URI". Keep False for normal runs; set True only for
    # debugging AND launch from outside the repo.
    local_mode: bool = False
    # Forward worker stdout/stderr to the driver. Kept False by default to avoid
    # noisy Ray/actor logs; the app's own [Bilevel]/[ES]/[PPO] logs are emitted
    # driver-side and surface via the root logger regardless.
    log_to_driver: bool = False

    # RAY_DEBUG=1 attaches the Ray distributed debugger to every worker — pure
    # overhead for normal runs. Off by default; turn on only when step-debugging.
    ray_debug: bool = False

    # Per-process math-thread caps. VECLIB_MAXIMUM_THREADS is the one that
    # limits Apple Accelerate (numpy's BLAS on macOS), which ignores
    # OMP_NUM_THREADS. All keyed off omp_threads so a single knob controls them.
    _THREAD_ENV_KEYS = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    def _thread_env_vars(self) -> Dict[str, str]:
        n = str(self.omp_threads if self.omp_threads is not None else 1)
        return {key: n for key in self._THREAD_ENV_KEYS}

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

        # Cap BLAS/OpenMP threads per process. On macOS numpy links against
        # Apple Accelerate, which IGNORES OMP_NUM_THREADS and fans out over all
        # cores via GCD unless VECLIB_MAXIMUM_THREADS is set. Without these caps
        # every Ray worker spins one math-thread per core, so N workers x C
        # cores threads oversubscribe the machine and starve the UI.
        for key, val in self._thread_env_vars().items():
            os.environ[key] = val

        if self.ray_debug:
            os.environ["RAY_DEBUG"] = "1"

        # logging behaviour for workers
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

        torch.set_default_device(self.device)

        # The driver has already imported torch, so the env vars above are too
        # late for its thread pool — set it explicitly. Workers instead inherit
        # the caps via runtime_env["env_vars"] before they import torch.
        n = max(1, self.omp_threads if self.omp_threads is not None else 1)
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # interop pool can only be resized before any parallel work starts
            pass

    def initialize(self):
        self._apply_env_vars()

        # Silence noisy loggers globally
        # # logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("ray.rllib").setLevel(logging.WARNING)
        logging.getLogger("ray.tune").setLevel(logging.WARNING)
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        logging.getLogger("asyncio").setLevel(logging.ERROR)

        # Propagate the thread caps to every Ray worker via runtime_env so they
        # are set BEFORE the worker imports torch/numpy (setting them in this
        # process's os.environ does not reach spawned workers).
        runtime_env = dict(self.runtime_env or {})
        env_vars = dict(runtime_env.get("env_vars", {}))
        for key, val in self._thread_env_vars().items():
            env_vars.setdefault(key, val)

        # Give workers the repo on their import path so they can import `core`
        # and (non-installed) `examples.*` WITHOUT Ray uploading a working_dir.
        # On a single machine the repo is on the shared filesystem, so a
        # working_dir upload is pure waste — and launching via `uv run` makes Ray
        # auto-capture + upload the whole cwd (incl. the ~1.2 GB .venv), which
        # thrashes /tmp/ray and macOS launchservicesd. Launch with the venv
        # python directly (e.g. `.venv/bin/python ...`), not `uv run`.
        repo_root = os.getcwd()
        existing_pp = env_vars.get("PYTHONPATH")
        env_vars["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
        )

        runtime_env["env_vars"] = env_vars

        # Disable Ray's per-task worker rename (setproctitle). On macOS it does a
        # synchronous XPC round-trip to launchservicesd on every task; with many
        # workers that saturates the Launch Services queue and freezes the whole
        # UI (CPU stays idle). The rename is cosmetic (Activity Monitor only).
        # Runs once per worker at start-up, before any task.
        runtime_env.setdefault(
            "worker_process_setup_hook",
            "core.adaptors.ray._worker_hooks.silence_setproctitle",
        )
        # Same for the driver process (workers get it via the hook above).
        try:
            ray._raylet.setproctitle = lambda *args, **kwargs: None
        except Exception:
            pass

        ray.init(
            ignore_reinit_error=True,
            logging_level=self.logging_level,
            runtime_env=runtime_env,
            local_mode=self.local_mode,
            log_to_driver=self.log_to_driver,
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
