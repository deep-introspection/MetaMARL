"""Visualization module for bilevel fishery experiments.

Provides functions to visualize fish/algae populations, harvest actions,
and regulatory mechanism effects over time.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_ecosystem_dynamics(
    trajectories: list[dict[str, Any]],
    title: str = "Ecosystem Dynamics",
    save_path: Optional[str] = None,
    sustainability_threshold: float = 0.1,
) -> plt.Figure:
    """Plot fish and algae populations over time.

    Args:
        trajectories: List of trajectory records with keys:
            - episode: int
            - step: int
            - fish_population: float
            - algae_population: float
        title: Plot title
        save_path: Path to save figure (optional)
        sustainability_threshold: Fish population collapse threshold

    Returns:
        matplotlib Figure object
    """
    if not trajectories:
        raise ValueError("No trajectory data provided")

    episodes = _group_by_episode(trajectories)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    for i, (ep, data) in enumerate(episodes.items()):
        ax1.plot(
            data["steps"],
            data["fish"],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    ax1.axhline(
        y=sustainability_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Collapse threshold ({sustainability_threshold})",
    )
    ax1.set_ylabel("Fish Population", fontsize=12)
    ax1.set_title(f"{title} - Fish Population", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    for i, (ep, data) in enumerate(episodes.items()):
        ax2.plot(
            data["steps"],
            data["algae"],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("Algae Population", fontsize=12)
    ax2.set_title(f"{title} - Algae Population", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_combined_trial_analysis(
    trajectories: list[dict[str, Any]],
    mechanism_params: Optional[dict[str, float]] = None,
    sustainability_threshold: float = 0.1,
    title: str = "Trial Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive visualization combining ecosystem and action analysis.

    Args:
        trajectories: List of trajectory records with keys:
            - episode: int
            - step: int
            - fish_population: float
            - algae_population: float
            - total_harvest: float (optional)
            - quota_limit: float (optional)
        mechanism_params: Mechanism parameters for context
        sustainability_threshold: Fish population collapse threshold
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not trajectories:
        raise ValueError("No trajectory data provided")

    episodes = _group_by_episode(trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    # Top left: Fish population with sustainability threshold
    for i, (ep, data) in enumerate(episodes.items()):
        axes[0, 0].plot(
            data["steps"],
            data["fish"],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    axes[0, 0].axhline(
        y=sustainability_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Collapse threshold",
    )
    axes[0, 0].set_ylabel("Fish Population", fontsize=12)
    axes[0, 0].set_title("Fish Population Dynamics", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Top right: Algae population
    for i, (ep, data) in enumerate(episodes.items()):
        axes[0, 1].plot(
            data["steps"],
            data["algae"],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    axes[0, 1].set_ylabel("Algae Population", fontsize=12)
    axes[0, 1].set_title("Algae Population Dynamics", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Bottom left: Harvest vs quota (if available)
    has_harvest = any("harvest" in data and data["harvest"] for data in episodes.values())
    if has_harvest:
        for i, (ep, data) in enumerate(episodes.items()):
            if data.get("harvest"):
                axes[1, 0].plot(
                    data["steps"],
                    data["harvest"],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=f"Episode {ep} - Harvest",
                )
            if data.get("quota"):
                axes[1, 0].plot(
                    data["steps"],
                    data["quota"],
                    color=colors[i],
                    alpha=0.4,
                    linestyle=":",
                    linewidth=1,
                    label=f"Episode {ep} - Quota",
                )
        axes[1, 0].set_xlabel("Time Step", fontsize=12)
        axes[1, 0].set_ylabel("Amount", fontsize=12)
        axes[1, 0].set_title("Harvest vs Quota", fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        # Plot rewards instead if no harvest data
        has_rewards = any("rewards" in data and data["rewards"] for data in episodes.values())
        if has_rewards:
            for i, (ep, data) in enumerate(episodes.items()):
                if data.get("rewards"):
                    axes[1, 0].plot(
                        data["steps"],
                        data["rewards"],
                        color=colors[i],
                        alpha=0.7,
                        linewidth=1.5,
                        label=f"Episode {ep}",
                    )
            axes[1, 0].set_xlabel("Time Step", fontsize=12)
            axes[1, 0].set_ylabel("Reward", fontsize=12)
            axes[1, 0].set_title("Reward over Time", fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        else:
            axes[1, 0].text(
                0.5, 0.5, "No harvest/reward data",
                ha="center", va="center", transform=axes[1, 0].transAxes
            )

    # Bottom right: Phase plot (fish vs algae)
    scatter = None
    for i, (ep, data) in enumerate(episodes.items()):
        if data.get("rewards"):
            scatter = axes[1, 1].scatter(
                data["algae"],
                data["fish"],
                c=data["rewards"],
                cmap="viridis",
                alpha=0.6,
                s=30,
                label=f"Episode {ep}",
            )
        else:
            scatter = axes[1, 1].scatter(
                data["algae"],
                data["fish"],
                c=range(len(data["fish"])),
                cmap="viridis",
                alpha=0.6,
                s=30,
                label=f"Episode {ep}",
            )

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label("Time Step / Reward", fontsize=10)

    axes[1, 1].set_xlabel("Algae Population", fontsize=12)
    axes[1, 1].set_ylabel("Fish Population", fontsize=12)
    axes[1, 1].set_title("Phase Plot", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.98)

    if mechanism_params:
        info_parts = []
        for key in ["fixed_quota", "prop_quota", "fine_amount", "min_stock", "ban_period"]:
            if key in mechanism_params:
                info_parts.append(f"{key}={mechanism_params[key]:.3f}")
        if info_parts:
            info_text = "Mechanism: " + ", ".join(info_parts)
            fig.text(0.5, 0.01, info_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _group_by_episode(trajectories: list[dict[str, Any]]) -> dict[int, dict[str, list]]:
    """Group trajectory records by episode."""
    episodes: dict[int, dict[str, list]] = {}

    for record in trajectories:
        ep = record.get("episode", 0)
        if ep not in episodes:
            episodes[ep] = {
                "steps": [],
                "fish": [],
                "algae": [],
                "harvest": [],
                "quota": [],
                "rewards": [],
            }

        episodes[ep]["steps"].append(record.get("step", len(episodes[ep]["steps"])))
        episodes[ep]["fish"].append(record.get("fish_population", 0.0))
        episodes[ep]["algae"].append(record.get("algae_population", 0.0))

        if "total_harvest" in record:
            episodes[ep]["harvest"].append(record["total_harvest"])
        if "quota_limit" in record:
            episodes[ep]["quota"].append(record["quota_limit"])
        if "reward" in record:
            episodes[ep]["rewards"].append(record["reward"])

    return episodes


PARAM_NAMES = ["fixed_quota", "prop_quota", "min_stock", "fine_amount", "ban_period"]
DEFAULT_PARAM_SCALES = [1.0, 1.0, 1.0, 2.0, 10.0]  # Denormalization factors (legacy)


def plot_fitness_vs_parameters(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Fitness vs Mechanism Parameters",
    save_path: Optional[str] = None,
    param_scales: Optional[list[float]] = None,
) -> plt.Figure:
    """Plot fitness as a function of each mechanism parameter.

    Args:
        population_history: List of (iteration, (population, fitness)) tuples
            where population is (N, 5) and fitness is (N,)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not population_history:
        raise ValueError("No population history provided")

    scales = param_scales if param_scales is not None else DEFAULT_PARAM_SCALES

    # Collect all data points
    all_params = []
    all_fitness = []
    all_iters = []

    for iteration, (population, fitness) in population_history:
        for i in range(len(fitness)):
            all_params.append(population[i])
            all_fitness.append(fitness[i])
            all_iters.append(iteration)

    all_params = np.array(all_params)
    all_fitness = np.array(all_fitness)
    all_iters = np.array(all_iters)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each parameter vs fitness
    for i, (name, scale) in enumerate(zip(PARAM_NAMES, scales)):
        ax = axes[i]
        param_vals = all_params[:, i] * scale

        scatter = ax.scatter(
            param_vals,
            all_fitness,
            c=all_iters,
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        ax.set_xlabel(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Fitness", fontsize=11)
        ax.set_title(f"Fitness vs {name.replace('_', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add colorbar to last subplot
    cbar = plt.colorbar(scatter, ax=axes[4])
    cbar.set_label("Iteration", fontsize=10)

    # Use 6th subplot for best fitness over iterations
    ax = axes[5]
    best_per_iter = {}
    for iteration, (_, fitness) in population_history:
        if iteration not in best_per_iter:
            best_per_iter[iteration] = float(fitness.max())
        else:
            best_per_iter[iteration] = max(best_per_iter[iteration], float(fitness.max()))

    iters = sorted(best_per_iter.keys())
    best_vals = [best_per_iter[i] for i in iters]

    ax.plot(iters, best_vals, "o-", linewidth=2, markersize=6, color="tab:blue")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Best Fitness Over Iterations", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parameter_evolution(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Parameter Evolution",
    save_path: Optional[str] = None,
    param_scales: Optional[list[float]] = None,
) -> plt.Figure:
    """Plot how the best mechanism parameters evolve over iterations.

    Args:
        population_history: List of (iteration, (population, fitness)) tuples
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not population_history:
        raise ValueError("No population history provided")

    scales = param_scales if param_scales is not None else DEFAULT_PARAM_SCALES

    # Extract best candidate per iteration
    iterations = []
    best_params = []

    for iteration, (population, fitness) in population_history:
        best_idx = int(np.argmax(fitness))
        iterations.append(iteration)
        best_params.append(population[best_idx])

    best_params = np.array(best_params)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for i, (name, scale, color) in enumerate(zip(PARAM_NAMES, scales, colors)):
        param_vals = best_params[:, i] * scale
        ax.plot(
            iterations,
            param_vals,
            "o-",
            linewidth=2,
            markersize=5,
            color=color,
            label=name.replace("_", " ").title(),
            alpha=0.8,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Parameter Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
