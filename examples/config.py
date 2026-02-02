"""Configuration and constants for bilevel fishery optimization.

This module contains all default configurations, hyperparameters, and constants
used throughout the bilevel optimization experiment.
"""

import numpy as np

# =====================================#
# CONSTANTS AND DEFAULT CONFIGURATIONS #
# =====================================#

# Default ecological parameters for Lotka-Volterra dynamics
DEFAULT_ECOLOGY_CONFIG = {
    "alpha": 1.0,  # Algae growth rate
    "beta": 0.5,  # Algae consumption by fish
    "delta": 0.5,  # Fish growth from algae
    "gamma": 1.0,  # Fish natural death rate
    "dt": 0.05,  # Time step size
    "horizon": 200,  # Episode length
    "algae_init": 1.0,  # Initial algae population
    "fish_init": 0.5,  # Initial fish population
}

# Default mechanism parameters
DEFAULT_MECHANISM_CONFIG = {
    "fixed_quota": 0.2,  # Fixed harvest quota
    "prop_quota": 0.1,  # Proportional quota factor
    "min_stock": 0.1,  # Minimum stock threshold
    "fine_amount": 1.0,  # Fine per unit over-harvest
    "ban_period": 2,  # Periods banned after violation
    "ban_success_prob": 1.0,  # Probability ban is enforced
    "fine_success_prob": 1.0,  # Probability fine is enforced
}

# Social/governance defaults (optional)
DEFAULT_SOCIAL_CONFIG = {
    "num_commons": 1,  # Number of distinct commons (resource pools)
    "enable_reputation": False,  # Include reputation state per agent
    "reputation_init": 0.5,
    "reputation_min": 0.0,
    "reputation_max": 1.0,
    "reputation_vote_weight": 0.05,  # Peer vote impact on reputation
    "reputation_vote_decay": 0.1,  # EMA decay for reputation votes
    "reputation_vote_weight_by_reputation": False,  # Weight votes by voter reputation
    "reputation_behavior_weight": 0.1,  # Over-harvest penalty on reputation
    "enable_governance": False,  # Enable voting over mechanism/access
    "governance_strength": 1.0,  # Blend between base params and votes
    "access_mode": "public",  # public|members|voted
    "access_scope": "all",  # all|members (only used for voted access)
    "access_threshold_init": 0.5,
    "commons": None,  # Optional list of per-commons overrides
}

# Gaussian noise defaults (optional)
DEFAULT_NOISE_CONFIG = {
    "action_noise_std": 0.0,  # Policy outputs mean; env samples action
    "action_noise_clip": 3.0,  # Clip noise in std-dev units
    "mechanism_noise_std": 0.0,  # Gaussian noise on mechanism params
    "mechanism_noise_mode": "episode",  # episode|step
    "ecology_noise_std": 0.0,  # Gaussian noise on ecology params
    "observation_noise_std": 0.0,  # Optional observation noise
}

# Private fishery defaults (optional)
DEFAULT_PRIVATE_FISHERY_CONFIG = {
    "enable_private_fishery": False,  # Additional reputation-allocated fishery
    "private_availability": 0.05,  # Fraction of private fish available each step
    "private_min_stock": DEFAULT_MECHANISM_CONFIG["min_stock"],
    "private_allocation_mode": "reputation",  # reputation|uniform
    "private_algae_init": DEFAULT_ECOLOGY_CONFIG["algae_init"],
    "private_fish_init": DEFAULT_ECOLOGY_CONFIG["fish_init"],
    "include_private_in_obs": False,
}

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    "num_fishermen": 3,  # Number of fishermen agents
    "horizon": 200,  # Episode length in timesteps
    "outer_iters": 10,  # Number of outer optimization iterations
    "inner_iters": 100,  # Number of inner training iterations
    "eval_episodes": 5,  # Number of evaluation episodes
    "pop_size": 16,  # Population size for evolution strategies (increased for antithetic sampling)
    "workers": 2,  # Number of parallel workers
    "sustain_weight": 5.0,  # Sustainability weight
    "sus_threshold": 0.1,  # Sustainability threshold
}

# Evolution Strategies algorithm parameters
ES_CONFIG = {
    "sigma": 0.15,  # Initial noise std
    "lr_mean": 0.1,  # Learning rate for mean
    "lr_sigma": 0.05,  # Learning rate for sigma adaptation
    "min_sigma": 1e-3,  # Minimum sigma
    "max_sigma": 0.5,  # Maximum sigma
}

# PPO training configuration
PPO_CONFIG = {
    "gamma": 0.99,  # Discount factor
    "train_batch_size": 4000,  # Training batch size
    "minibatch_size": 512,  # SGD minibatch size
    "lr": 3e-4,  # Learning rate
    "framework": "torch",  # Deep learning framework
}

# Observation/action space bounds
OBSERVATION_BOUNDS = {
    "low": 0.0,
    "high": np.finfo(np.float32).max,
    "shape": (2,),  # [algae_population, fish_population]
    "dtype": np.float32,
}

ACTION_BOUNDS = {
    "low": 0.0,  # Minimum harvest fraction
    "high": 1.0,  # Maximum harvest fraction
    "shape": (1,),
    "dtype": np.float32,
}

# Logging and output configuration
LOGGING_CONFIG = {
    "default_log_dir": "runs",
    "wandb_project": "bilevel-fishery",
    "progress_log_frequency": 10,  # Log every N% of training
}

# Numerical stability constant
EPS = 1e-8
