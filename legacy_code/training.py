"""Training utilities and PPO algorithm configuration.

This module provides functions for building and configuring PPO algorithms
for the multi-agent fishery environment.
"""

from typing import Tuple

from ray import init as ray_init
from ray import shutdown as ray_shutdown
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from legacy_code.mechanism import MechanismParameters
from legacy_code.environment import FisheryEnvFixed
from legacy_code.config import PPO_CONFIG


def initialize_ray(logging_level: str = "ERROR") -> None:
    """Initialize Ray with CPU-only settings.

    Args:
        logging_level: Ray logging level ("ERROR", "WARN", "INFO", "DEBUG")
    """
    import os

    import torch

    # Set environment variables for Ray workers to use CPU only
    cpu_env_vars = {
        "RAY_DISABLE_IMPORT_WARNING": "1",
        "CUDA_VISIBLE_DEVICES": "",  # Hide CUDA devices completely from all processes
        "RAY_USE_MPS": "0",  # Disable MPS usage
        "TORCH_USE_CUDA_DSA": "0",  # Disable CUDA DSA
        "PYTORCH_CUDA_ALLOC_CONF": "",  # Clear CUDA allocation config
        "RLLIB_NUM_GPUS": "0",  # Force RLlib to use 0 GPUs
        "OMP_NUM_THREADS": "1",  # Limit OpenMP threads
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",  # Disable MPS fallback
        "USE_CUDA": "0",  # Disable CUDA usage
        "CUDA_LAUNCH_BLOCKING": "0",  # Disable CUDA launch blocking
        "NVML_DISABLED": "1",  # Disable NVIDIA Management Library
    }

    # Set environment variables in current process
    for key, value in cpu_env_vars.items():
        os.environ[key] = value

    # Configure PyTorch for CPU only
    torch.set_default_device("cpu")

    # Explicitly disable CUDA in current process by monkey patching
    if hasattr(torch, "cuda") and torch.cuda is not None:
        # Save original functions
        _original_is_available = getattr(torch.cuda, "is_available", lambda: False)
        _original_device_count = getattr(torch.cuda, "device_count", lambda: 0)

        # Override CUDA functions to always return CPU-only values
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0

    print("✓ Configured PyTorch and Ray to use CPU only")

    # Initialize Ray with runtime environment exclusions to avoid large file upload
    runtime_env = {
        "excludes": [
            "runs/",
            "analysis_plots/",
            "*.png",
            "*.jpg",
            "*.tar.gz",
            "*.zip",
            ".venv/",
            "__pycache__/",
            ".git/",
            "wandb/",
            "ray_results/",
        ]
    }

    ray_init(
        ignore_reinit_error=True, logging_level=logging_level, runtime_env=runtime_env
    )


def shutdown_ray() -> None:
    """Shutdown Ray gracefully."""
    ray_shutdown()


def build_ppo_algorithm(
    num_fishermen: int,
    mechanism_params: MechanismParameters,
    num_workers: int = 2,
    env_config: dict = None,
) -> Tuple[object, FisheryEnvFixed]:
    """Build PPO algorithm configured for multi-agent fishery environment.

    Args:
        num_fishermen: Number of fishermen agents
        mechanism_params: Regulatory mechanism parameters
        num_workers: Number of parallel workers for training
        env_config: Environment configuration dictionary (optional)

    Returns:
        Tuple of (configured PPO algorithm, dummy environment instance)
    """

    def create_environment(rllib_env_config: dict) -> FisheryEnvFixed:
        """Environment factory function."""
        config = dict(env_config or {})  # Use passed env_config as base
        config.update(rllib_env_config)  # Update with RLLib's env_config
        config.update(
            {
                "num_fishermen": num_fishermen,
                "fixed_quota": mechanism_params.fixed_quota,
                "prop_quota": mechanism_params.prop_quota,
                "min_stock": mechanism_params.min_stock,
                "fine_amount": mechanism_params.fine_amount,
                "ban_period": mechanism_params.ban_period,
            }
        )
        return FisheryEnvFixed(config)

    # Register environment (fresh registration each time for new parameters)
    register_env("FisheryEnvFixed-v0", create_environment)

    # Create dummy environment to extract spaces
    dummy_env = create_environment({})
    observation_space = dummy_env.observation_space("fisherman_0")
    action_space = dummy_env.action_space("fisherman_0")

    # Configure PPO algorithm for CPU-only execution
    config = (
        PPOConfig()
        .environment(env="FisheryEnvFixed-v0")
        .framework(
            PPO_CONFIG["framework"],
            torch_compile_worker=False,
            torch_compile_worker_dynamo_backend=None,
        )
        .resources(num_gpus=0)  # Force CPU only
        .env_runners(
            num_env_runners=0
        )  # Force 0 workers to avoid CUDA issues in worker processes
        .training(
            gamma=PPO_CONFIG["gamma"],
            train_batch_size=PPO_CONFIG["train_batch_size"],
            minibatch_size=PPO_CONFIG["minibatch_size"],
            lr=PPO_CONFIG["lr"],
        )
        .multi_agent(
            policies={"fisher_policy": (None, observation_space, action_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "fisher_policy",
            policies_to_train=["fisher_policy"],
        )
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .fault_tolerance(restart_failed_env_runners=False)
    )

    algorithm = config.build_algo()
    return algorithm, dummy_env


def train_algorithm(
    algorithm, num_iterations: int, progress_callback=None, wandb_run=None
) -> list:
    """Train algorithm for specified number of iterations.

    Args:
        algorithm: PPO algorithm to train
        num_iterations: Number of training iterations
        progress_callback: Optional callback function for progress reporting
        wandb_run: Optional W&B run for logging

    Returns:
        List of training results
    """
    training_results = []

    for iteration in range(num_iterations):
        result = algorithm.train()
        training_results.append(result)

        # Log to W&B if available
        if wandb_run is not None:
            # Extract episode reward from env_runners metrics (newer RLlib format)
            episode_reward = result.get("env_runners", {}).get(
                "episode_reward_mean", float("nan")
            )
            wandb_run.log(
                {
                    "train/episode_reward_mean": episode_reward,
                    "train/iteration": iteration,
                }
            )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(iteration, result)

    return training_results


def save_algorithm_checkpoint(algorithm, checkpoint_dir: str = None) -> str:
    """Save algorithm checkpoint.

    Args:
        algorithm: Algorithm to checkpoint
        checkpoint_dir: Optional directory for checkpoint

    Returns:
        Path to saved checkpoint
    """
    if checkpoint_dir:
        return algorithm.save(checkpoint_dir)
    else:
        return algorithm.save()


def load_algorithm_checkpoint(algorithm, checkpoint_path: str) -> None:
    """Load algorithm from checkpoint.

    Args:
        algorithm: Algorithm instance to load into
        checkpoint_path: Path to checkpoint file
    """
    algorithm.restore(checkpoint_path)
