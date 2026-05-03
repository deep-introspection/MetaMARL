import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import logging

import ray
import torch

DeviceType = Literal["cpu", "cuda", "mps"]


@dataclass
class RayRuntimeConfig:
    """Configuration for the Ray runtime environment.

    Centralises all Ray initialisation parameters and device-specific
    environment-variable settings.  Call ``initialize()`` (or
    ``RayRuntime.ensure_initialized()``) exactly once at the start of a
    training run.

    Environment variables are set *before* ``ray.init()`` is called so that
    they propagate to driver-side PyTorch and to workers that inherit the
    driver environment.

    .. note::
       ``uv run`` is intentionally avoided in this codebase to prevent Ray
       worker processes from picking up conflicting ``VIRTUAL_ENV`` /
       ``PATH`` overrides injected by ``uv``.  Launch scripts should invoke
       Python directly.

    Parameters
    ----------
    device : {"cpu", "cuda", "mps"}
        Target compute device.  Controls which CUDA/MPS environment variables
        are set and which PyTorch default device is configured.
    num_cpus : int, optional
        Number of CPUs to allocate to the Ray cluster.  ``None`` lets Ray
        auto-detect.
    num_gpus : int, optional
        Number of GPUs to allocate.  ``None`` lets Ray auto-detect.
    omp_threads : int
        Value for ``OMP_NUM_THREADS``; keep at 1 to avoid contention when
        many Ray workers share a machine.
    disable_mps : bool
        When ``True`` and ``device="cpu"``, sets ``RAY_USE_MPS=0`` to
        prevent accidental MPS usage in workers.
    disable_cuda : bool
        When ``True`` and ``device="cuda"``, hides CUDA devices via
        ``CUDA_VISIBLE_DEVICES=""`` (useful for CPU-only debugging).
    logging_level : str
        Logging level string passed to ``ray.init`` (e.g. ``"ERROR"``).
    runtime_env : dict, optional
        Ray ``runtime_env`` dictionary forwarded verbatim to ``ray.init``.
        Can specify pip packages, conda envs, or environment variables that
        all Ray workers should inherit.
    init_kwargs : dict
        Additional keyword arguments forwarded verbatim to ``ray.init``.
    ray_debug : bool
        When ``True``, sets ``RAY_DEBUG=1`` for verbose Ray-internal debug
        output.
    """

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

    def _apply_env_vars(self):
        """Set process-level environment variables for the selected device.

        Must be called before ``ray.init()`` so that driver-side PyTorch
        picks up the correct device settings.  Workers that inherit the
        driver's environment will also receive these variables.

        The mapping from ``device`` to env vars is:

        * ``"cpu"`` — hides CUDA devices, disables MPS fallback, optionally
          disables MPS entirely.
        * ``"mps"`` — hides CUDA, enables ``PYTORCH_ENABLE_MPS_FALLBACK``
          for ops not yet supported natively.
        * ``"cuda"`` — enables MPS fallback to CPU, optionally hides devices
          if ``disable_cuda`` is set (useful for profiling on a CPU node).

        Additionally sets ``OMP_NUM_THREADS``, ``RAY_DEBUG``, and silences
        Ray's noisy backend log channels.
        """
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
        """Apply environment variables and start the Ray runtime.

        Calls ``_apply_env_vars()`` first, then silences the most verbose
        Ray/RLlib/Tune loggers before calling ``ray.init()``.  Uses
        ``ignore_reinit_error=True`` so that repeated calls in interactive
        sessions do not raise.

        ``local_mode=True`` is set for deterministic single-process
        execution.  To run with true distributed workers, override this via
        ``init_kwargs``.
        """
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
            local_mode=True,  # Turn on for debugging only (DO NOT TURN OFF)
            log_to_driver=False,
            include_dashboard=False,
            _system_config={
                "metrics_report_interval_ms": 0,
                "enable_metrics_collection": False,
            },
            **self.init_kwargs,
        )


class RayRuntime:
    """Singleton-like guard that ensures Ray is initialised at most once.

    Wraps ``RayRuntimeConfig.initialize()`` with a ``ray.is_initialized()``
    guard so that multiple call sites can safely call
    ``RayRuntime.ensure_initialized(cfg)`` without raising a
    ``RaySystemError`` on re-initialisation.

    Attributes
    ----------
    _initialized : bool
        Class-level flag set to ``True`` after the first successful
        ``ray.init()`` call through this class.
    """

    _initialized = False

    @classmethod
    def ensure_initialized(cls, cfg: RayRuntimeConfig):
        """Initialise Ray using the provided config if not already running.

        Parameters
        ----------
        cfg : RayRuntimeConfig
            Runtime configuration specifying device, resource limits, and
            Ray init parameters.
        """
        if not ray.is_initialized():
            cfg.initialize()
            cls._initialized = True
