"""Test if PPO learns meaningful fishing behavior with current settings."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from examples.bilevel_fishery.mechanism import FisheryMechanism

EPS = 1e-8


class SimpleFisheryEnv(gym.Env):
    """Standalone fishery env matching FisheryRegulatedEnv for testing PPO convergence.

    This is a single-agent version that mirrors the multi-agent environment logic.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.mechanism: FisheryMechanism = config["mechanism"]
        self.horizon = config.get("horizon", 200)
        eco = config.get("ecology_cfg", {})

        self.max_fish = eco.get("max_fish", 5.0)
        self.max_algae = eco.get("max_algae", 5.0)
        self.alpha = eco.get("alpha", 0.5)
        self.beta = eco.get("beta", 0.1)
        self.delta = eco.get("delta", 0.2)
        self.gamma_eco = eco.get("gamma", 0.4)
        self.dt = eco.get("dt", 0.01)
        self.fish_init = eco.get("fish_init", 1.0)
        self.algae_init = eco.get("algae_init", 1.0)

        # Observation: 5 base features + mechanism params (matching real env)
        mechanism_dim = len(self.mechanism.to_vector())
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (5 + mechanism_dim,), np.float32
        )
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)

        self._t = 0
        self.fish = self.fish_init
        self.algae = self.algae_init
        self.ban_remaining = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        # Match real env: lognormal initialization with noise
        self.fish = max(EPS, self._rng.lognormal(np.log(self.fish_init), 0.05))
        self.algae = max(EPS, self._rng.lognormal(np.log(self.algae_init), 0.05))
        self.ban_remaining = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        """Match FisheryRegulatedEnv._observation + mechanism vector."""
        m = self.mechanism
        fish_norm = self.fish / self.max_fish
        algae_norm = self.algae / self.max_algae

        # Ban status: normalized remaining ban steps
        ban_norm = 0.0
        if m.ban_period > 0:
            ban_norm = self.ban_remaining / m.ban_period

        # Computed signals (matching real env)
        effective_quota = min(m.fixed_quota, m.prop_quota * fish_norm)
        no_fish_zone = float(fish_norm < m.min_stock)

        # Base observation + mechanism vector (matching marl_regulated.py observation())
        base_obs = np.array(
            [
                fish_norm,
                algae_norm,
                ban_norm,
                effective_quota,
                no_fish_zone,
            ],
            dtype=np.float32,
        )

        theta = m.to_vector()
        return np.concatenate([base_obs, theta], axis=0)

    def _intrinsic_utility(self, action: float) -> float:
        """Match FisheryRegulatedEnv.intrinsic_utility."""
        fish_norm = self.fish / self.max_fish
        return action * fish_norm

    def _violation_signal(self, u: float) -> float:
        """Match FisheryRegulatedEnv.violation_signal."""
        m = self.mechanism
        fish_norm = self.fish / self.max_fish

        # Quota violation
        quota = max(0.0, u - min(m.fixed_quota, m.prop_quota * fish_norm))
        # No-fish zone violation
        ban = float(fish_norm < m.min_stock) * u
        return float(quota + ban)

    def step(self, action: np.ndarray):
        self._t += 1
        action_val = float(np.clip(action[0], 0, 1))
        m = self.mechanism

        # Check ban status first (matching marl_regulated.py _step)
        if self.ban_remaining > 0:
            self.ban_remaining -= 1
            # Banned: zero action, mild penalty (matching marl_regulated.py)
            effective_action = 0.0
            reward = -0.01
        else:
            effective_action = action_val

            # Compute reward: u - penalty * violation (matching marl_regulated.py)
            u = self._intrinsic_utility(action_val)
            v = self._violation_signal(u)

            # Stochastic enforcement: only penalize if violation is detected
            if v > 0 and self._rng.random() < m.catch_prob:
                reward = u - m.fine_amount * v
                # Apply ban if violation detected
                if m.ban_period > 0:
                    self.ban_remaining = m.ban_period
            else:
                reward = u

        # Harvest calculation (matching FisheryRegulatedEnv.transition_kernel)
        # For single agent: H = max_fish * desired * scale
        # where desired = action * fish_norm, scale = min(1.0, fish_norm / max(EPS, desired))
        fish_norm = self.fish / self.max_fish
        desired = effective_action * fish_norm
        scale = min(1.0, fish_norm / max(EPS, desired)) if desired > 0 else 1.0
        H = self.max_fish * desired * scale

        # Lotka-Volterra dynamics (matching real env)
        fish_next = self.fish + self.dt * (
            self.delta * self.algae * self.fish - self.gamma_eco * self.fish - H
        )
        algae_next = self.algae + self.dt * (
            self.alpha * self.algae - self.beta * self.algae * self.fish
        )

        # Clamp for numerical stability
        self.fish = np.clip(fish_next, 0.0, self.max_fish)
        self.algae = np.clip(algae_next, 0.0, self.max_algae)

        terminated = False
        truncated = self._t >= self.horizon

        return self._obs(), reward, terminated, truncated, {}


def env_creator(config):
    return SimpleFisheryEnv(config)


def test_ppo_convergence(
    train_iters: int = 50,
    gamma: float = 0.95,
    lr: float = 0.001,
    num_envs: int = 16,
    horizon: int = 1000,
    mechanism_kwargs: dict | None = None,
    ecology_cfg: dict | None = None,
):
    """Train PPO and track learning progress."""

    ray.init(local_mode=True, ignore_reinit_error=True)
    register_env("fishery_test", env_creator)

    # Use provided mechanism kwargs or defaults
    mech_defaults = {
        "fixed_quota": 1.0,
        "prop_quota": 1.0,
        "min_stock": 0.10,
        "fine_amount": 3.0,  # High enough to make sustainable fishing optimal
        "ban_period": 0,
        "catch_prob": 1.0,
    }
    if mechanism_kwargs:
        mech_defaults.update(mechanism_kwargs)

    mechanism = FisheryMechanism(**mech_defaults)

    # Use provided ecology config or defaults
    eco_defaults = {
        "algae_init": 1.0,
        "fish_init": 1.0,
        "max_fish": 5.0,
        "max_algae": 5.0,
        "alpha": 0.5,
        "beta": 0.1,
        "delta": 0.2,
        "gamma": 0.4,
        "dt": 0.01,
    }
    if ecology_cfg:
        eco_defaults.update(ecology_cfg)

    env_config = {
        "mechanism": mechanism,
        "horizon": horizon,
        "ecology_cfg": eco_defaults,
    }

    config = (
        PPOConfig()
        .environment(env="fishery_test", env_config=env_config)
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=num_envs,
            rollout_fragment_length=horizon,
            batch_mode="complete_episodes",
        )
        .training(
            gamma=gamma,
            lr=lr,
            train_batch_size=num_envs * horizon,
            minibatch_size=1024,
            grad_clip=0.5,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    rewards = []
    policy_losses = []
    vf_losses = []
    entropies = []

    print(f"Training PPO for {train_iters} iterations...")
    print(f"  gamma={gamma}, lr={lr}, envs={num_envs}")
    print(
        f"  mechanism: fixed_q={mechanism.fixed_quota}, prop_q={mechanism.prop_quota}, "
        f"min_stock={mechanism.min_stock}, fine={mechanism.fine_amount}, ban={mechanism.ban_period}, "
        f"catch_prob={mechanism.catch_prob}"
    )
    print()

    for i in range(train_iters):
        result = algo.train()

        ep_reward = result.get("env_runners", result).get(
            "episode_reward_mean", result.get("episode_reward_mean", 0)
        )
        rewards.append(ep_reward if ep_reward else 0)

        # Extract learner stats from info.learner.default_policy.learner_stats
        learner_stats = (
            result.get("info", {})
            .get("learner", {})
            .get("default_policy", {})
            .get("learner_stats", {})
        )

        policy_losses.append(
            learner_stats.get("policy_loss", learner_stats.get("total_loss", 0))
        )
        vf_losses.append(
            learner_stats.get("vf_loss", learner_stats.get("vf_loss_unclipped", 0))
        )
        entropies.append(
            learner_stats.get("entropy", learner_stats.get("curr_entropy_coeff", 0))
        )

        if i == 0:
            # Debug: print learner_stats keys on first iteration
            print(f"  learner_stats keys: {list(learner_stats.keys())}")

        if (i + 1) % 10 == 0:
            print(
                f"  iter {i + 1:3d}: reward={rewards[-1]:.4f}, policy_loss={policy_losses[-1]:.4f}"
            )

    algo.stop()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"PPO Convergence (gamma={gamma}, iters={train_iters})")

    axes[0, 0].plot(rewards)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Episode Reward Mean")
    axes[0, 0].set_title("Reward")
    axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)

    axes[0, 1].plot(policy_losses)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Policy Loss")
    axes[0, 1].set_title("Policy Loss")

    axes[1, 0].plot(vf_losses)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("VF Loss")
    axes[1, 0].set_title("VF Loss")

    axes[1, 1].plot(entropies)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].set_title("Policy Entropy")

    plt.tight_layout()
    out_path = Path("results/ppo_convergence.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\n=== Summary ===")
    print(f"Initial reward: {rewards[0]:.4f}")
    print(f"Final reward:   {rewards[-1]:.4f}")
    print(f"Max reward:     {max(rewards):.4f}")

    if rewards[-1] > 0.5:
        print("\n✓ PPO learned meaningful fishing behavior")
    elif rewards[-1] > rewards[0] + 0.1:
        print("\n~ PPO is learning but may need more iterations")
    else:
        print("\n✗ PPO not learning - check reward scale or hyperparameters")

    return rewards


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML to load defaults"
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--catch-prob", type=float, default=None)
    args = parser.parse_args()

    # Load from config if provided
    mechanism_kwargs = {}
    ecology_cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        mechanism_kwargs = cfg["mechanism"]["default"]
        ecology_cfg = cfg["inner"]["environment"]["env_config"]["ecology_cfg"]
        print(f"Loaded config from {args.config}")

    # Override catch_prob if specified on command line
    if args.catch_prob is not None:
        mechanism_kwargs["catch_prob"] = args.catch_prob

    print(f"Testing PPO with horizon={args.horizon}, lr={args.lr}")

    test_ppo_convergence(
        train_iters=args.iters,
        gamma=args.gamma,
        lr=args.lr,
        horizon=args.horizon,
        mechanism_kwargs=mechanism_kwargs if mechanism_kwargs else None,
        ecology_cfg=ecology_cfg,
    )
