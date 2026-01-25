"""Visualization module for fishery trials and ecosystem dynamics.

This module provides functionality to visualize fish/algae populations,
fishermen actions, and regulatory mechanism effects over time.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from legacy_code.utils import load_csv, load_json


def plot_ecosystem_dynamics(
    trajectories: List[Dict],
    title: str = "Ecosystem Dynamics",
    save_path: Optional[str] = None,
    show_sustainability_threshold: bool = True,
    sustainability_threshold: float = 0.1,
) -> plt.Figure:
    """Plot fish and algae populations over time.

    Args:
        trajectories: List of trajectory records from evaluation
        title: Plot title
        save_path: Path to save figure (optional)
        show_sustainability_threshold: Whether to show sustainability threshold line
        sustainability_threshold: Fish population collapse threshold

    Returns:
        matplotlib Figure object
    """
    if not trajectories:
        raise ValueError("No trajectory data provided")

    # Group by episode
    episodes = {}
    for record in trajectories:
        ep = record["episode"]
        if ep not in episodes:
            episodes[ep] = {"steps": [], "fish": [], "algae": []}
        episodes[ep]["steps"].append(record["step"])
        episodes[ep]["fish"].append(record["fish_population"])
        episodes[ep]["algae"].append(record["algae_population"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Color palette for episodes
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    # Plot fish populations
    for i, (ep, data) in enumerate(episodes.items()):
        ax1.plot(
            data["steps"],
            data["fish"],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    if show_sustainability_threshold:
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

    # Plot algae populations
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


def plot_fishermen_actions(
    trajectories: List[Dict],
    mechanism_params: Optional[Dict] = None,
    title: str = "Fishermen Actions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot fishermen harvest actions and regulatory responses.

    Args:
        trajectories: List of trajectory records from evaluation
        mechanism_params: Mechanism parameters for context
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not trajectories:
        raise ValueError("No trajectory data provided")

    # Group by episode
    episodes = {}
    for record in trajectories:
        ep = record["episode"]
        if ep not in episodes:
            episodes[ep] = {
                "steps": [],
                "harvest": [],
                "quota": [],
                "fish": [],
                "over_quota": [],
            }
        episodes[ep]["steps"].append(record["step"])
        episodes[ep]["harvest"].append(record["total_harvest"])
        episodes[ep]["quota"].append(record["quota_limit"])
        episodes[ep]["fish"].append(record["fish_population"])

        # Calculate if over quota
        over_quota = max(0, record["total_harvest"] - record["quota_limit"])
        episodes[ep]["over_quota"].append(over_quota)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    # Plot harvest vs quota
    for i, (ep, data) in enumerate(episodes.items()):
        axes[0].plot(
            data["steps"],
            data["harvest"],
            color=colors[i],
            alpha=0.7,
            linewidth=2,
            label=f"Episode {ep} - Harvest",
        )
        axes[0].plot(
            data["steps"],
            data["quota"],
            color=colors[i],
            alpha=0.5,
            linestyle="--",
            linewidth=1,
            label=f"Episode {ep} - Quota",
        )

    axes[0].set_ylabel("Harvest Amount", fontsize=12)
    axes[0].set_title(f"{title} - Harvest vs Quota", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot over-quota violations
    for i, (ep, data) in enumerate(episodes.items()):
        violation_steps = [
            s for s, v in zip(data["steps"], data["over_quota"]) if v > 0
        ]
        violation_amounts = [v for v in data["over_quota"] if v > 0]

        if violation_steps:
            axes[1].scatter(
                violation_steps,
                violation_amounts,
                color=colors[i],
                alpha=0.7,
                s=50,
                label=f"Episode {ep} violations",
            )

    axes[1].set_ylabel("Over-Quota Amount", fontsize=12)
    axes[1].set_title(f"{title} - Quota Violations", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    if any(ep for ep, data in episodes.items() if any(data["over_quota"])):
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot harvest rate (harvest / fish population)
    for i, (ep, data) in enumerate(episodes.items()):
        harvest_rate = [h / max(f, 1e-6) for h, f in zip(data["harvest"], data["fish"])]
        axes[2].plot(
            data["steps"],
            harvest_rate,
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )

    axes[2].set_xlabel("Time Step", fontsize=12)
    axes[2].set_ylabel("Harvest Rate (fraction)", fontsize=12)
    axes[2].set_title(f"{title} - Harvest Rate", fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add mechanism info as text if provided
    if mechanism_params:
        info_text = (
            f"Mechanism: Fixed Quota={mechanism_params.get('fixed_quota', 'N/A'):.3f}, "
            f"Prop Quota={mechanism_params.get('prop_quota', 'N/A'):.3f}, "
            f"Fine={mechanism_params.get('fine_amount', 'N/A'):.3f}"
        )
        fig.text(0.5, 0.02, info_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_combined_trial_analysis(
    trajectories: List[Dict],
    mechanism_params: Optional[Dict] = None,
    sustainability_threshold: float = 0.1,
    title: str = "Trial Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive visualization combining ecosystem and action analysis.

    Args:
        trajectories: List of trajectory records from evaluation
        mechanism_params: Mechanism parameters for context
        sustainability_threshold: Fish population collapse threshold
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not trajectories:
        raise ValueError("No trajectory data provided")

    # Group by episode
    episodes = {}
    for record in trajectories:
        ep = record["episode"]
        if ep not in episodes:
            episodes[ep] = {
                "steps": [],
                "fish": [],
                "algae": [],
                "harvest": [],
                "quota": [],
                "over_quota": [],
            }
        episodes[ep]["steps"].append(record["step"])
        episodes[ep]["fish"].append(record["fish_population"])
        episodes[ep]["algae"].append(record["algae_population"])
        episodes[ep]["harvest"].append(record["total_harvest"])
        episodes[ep]["quota"].append(record["quota_limit"])

        over_quota = max(0, record["total_harvest"] - record["quota_limit"])
        episodes[ep]["over_quota"].append(over_quota)

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

    # Bottom left: Harvest vs quota
    for i, (ep, data) in enumerate(episodes.items()):
        axes[1, 0].plot(
            data["steps"],
            data["harvest"],
            color=colors[i],
            alpha=0.8,
            linewidth=2,
            label=f"Episode {ep} - Harvest",
        )
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

    # Bottom right: Phase plot (fish vs algae) with harvest coloring
    for i, (ep, data) in enumerate(episodes.items()):
        scatter = axes[1, 1].scatter(
            data["algae"],
            data["fish"],
            c=data["harvest"],
            cmap="viridis",
            alpha=0.6,
            s=30,
            label=f"Episode {ep}",
        )

    # Add colorbar for harvest
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Total Harvest", fontsize=10)

    axes[1, 1].set_xlabel("Algae Population", fontsize=12)
    axes[1, 1].set_ylabel("Fish Population", fontsize=12)
    axes[1, 1].set_title("Phase Plot (colored by harvest)", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    # Add overall title and mechanism info
    fig.suptitle(title, fontsize=16, y=0.98)

    if mechanism_params:
        info_text = (
            f"Mechanism Parameters: "
            f"Fixed Quota={mechanism_params.get('fixed_quota', 'N/A'):.3f}, "
            f"Proportional Quota={mechanism_params.get('prop_quota', 'N/A'):.3f}, "
            f"Fine Amount={mechanism_params.get('fine_amount', 'N/A'):.3f}, "
            f"Min Stock={mechanism_params.get('min_stock', 'N/A'):.3f}"
        )
        fig.text(0.5, 0.01, info_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def load_and_visualize_experiment(
    experiment_dir: str,
    outer_iter: int,
    sustainability_threshold: float = 0.1,
    save_plots: bool = True,
) -> Dict[str, plt.Figure]:
    """Load experiment data and create visualizations.

    Args:
        experiment_dir: Path to experiment directory
        outer_iter: Outer iteration to visualize
        sustainability_threshold: Fish population collapse threshold
        save_plots: Whether to save plots to files

    Returns:
        Dictionary mapping plot names to Figure objects
    """
    import os

    # Load best candidate trajectory data
    trace_files = [
        f
        for f in os.listdir(f"{experiment_dir}/outer_{outer_iter}")
        if f.startswith("best_trace_") and f.endswith(".json")
    ]

    if not trace_files:
        raise FileNotFoundError(
            f"No trajectory files found in {experiment_dir}/outer_{outer_iter}"
        )

    # Load the first trajectory file
    trace_path = f"{experiment_dir}/outer_{outer_iter}/{trace_files[0]}"
    trajectories = load_json(trace_path)

    # Load mechanism parameters if available
    try:
        candidates_path = f"{experiment_dir}/outer_{outer_iter}/candidates.csv"
        headers, rows = load_csv(candidates_path)

        # Convert to list of dictionaries
        if headers and rows:
            candidates_data = [dict(zip(headers, row)) for row in rows]
            # Get best candidate (first row, which should be sorted by objective)
            best_candidate = candidates_data[0] if candidates_data else {}
            mechanism_params = {
                k: float(v)
                for k, v in best_candidate.items()
                if k
                in [
                    "fixed_quota",
                    "prop_quota",
                    "min_stock",
                    "fine_amount",
                    "ban_period",
                ]
            }
        else:
            mechanism_params = None
    except (FileNotFoundError, KeyError, IndexError, ValueError):
        mechanism_params = None

    figures = {}
    base_title = f"Outer Iteration {outer_iter}"

    # Create ecosystem dynamics plot
    figures["ecosystem"] = plot_ecosystem_dynamics(
        trajectories,
        title=f"{base_title} - Ecosystem Dynamics",
        sustainability_threshold=sustainability_threshold,
        save_path=f"{experiment_dir}/outer_{outer_iter}/ecosystem_dynamics.png"
        if save_plots
        else None,
    )

    # Create fishermen actions plot
    figures["actions"] = plot_fishermen_actions(
        trajectories,
        mechanism_params=mechanism_params,
        title=f"{base_title} - Fishermen Actions",
        save_path=f"{experiment_dir}/outer_{outer_iter}/fishermen_actions.png"
        if save_plots
        else None,
    )

    # Create combined analysis plot
    figures["combined"] = plot_combined_trial_analysis(
        trajectories,
        mechanism_params=mechanism_params,
        sustainability_threshold=sustainability_threshold,
        title=f"{base_title} - Combined Analysis",
        save_path=f"{experiment_dir}/outer_{outer_iter}/combined_analysis.png"
        if save_plots
        else None,
    )

    return figures


def create_experiment_summary_visualization(
    experiment_dir: str, save_path: Optional[str] = None
) -> plt.Figure:
    """Create summary visualization across all iterations of an experiment.

    Args:
        experiment_dir: Path to experiment directory
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    import os

    # Load experiment summary
    summary_path = f"{experiment_dir}/experiment_summary.csv"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Experiment summary not found: {summary_path}")

    headers, rows = load_csv(summary_path)

    if not headers or not rows:
        raise ValueError("Empty experiment summary")

    # Convert to list of dictionaries
    summary_data = [dict(zip(headers, row)) for row in rows]

    # Extract data for plotting with proper column names from CSV
    iterations = [row.get("outer_iter", i) for i, row in enumerate(summary_data)]
    objective_scores = [
        float(row.get("best_iteration_score", 0)) for row in summary_data
    ]
    mean_rewards = [float(row.get("mean_score", 0)) for row in summary_data]
    # Note: collapse_rate and sustainability_penalty may not exist in current CSV format
    collapse_rates = [float(row.get("collapse_rate", 0)) for row in summary_data]
    sustainability_penalties = [
        float(row.get("sustainability_penalty", 0)) for row in summary_data
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Objective score evolution
    axes[0, 0].plot(iterations, objective_scores, "o-", linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Objective Score")
    axes[0, 0].set_title("Objective Score Evolution")
    axes[0, 0].grid(True, alpha=0.3)

    # Mean reward evolution
    axes[0, 1].plot(
        iterations, mean_rewards, "o-", color="green", linewidth=2, markersize=6
    )
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Mean Reward")
    axes[0, 1].set_title("Economic Performance")
    axes[0, 1].grid(True, alpha=0.3)

    # Collapse rate evolution
    axes[1, 0].plot(
        iterations, collapse_rates, "o-", color="red", linewidth=2, markersize=6
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Collapse Rate")
    axes[1, 0].set_title("Ecosystem Collapse Rate")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Sustainability penalty evolution
    axes[1, 1].plot(
        iterations,
        sustainability_penalties,
        "o-",
        color="orange",
        linewidth=2,
        markersize=6,
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Sustainability Penalty")
    axes[1, 1].set_title("Sustainability Penalty")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Experiment Evolution Summary", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_mechanism_parameter_analysis(
    experiment_dir: str, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot analysis of all mechanism parameters tested across iterations.

    Args:
        experiment_dir: Path to experiment directory
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    import glob

    # Collect all candidate data from all iterations
    all_candidates = []
    candidate_files = glob.glob(f"{experiment_dir}/outer_*/candidates.csv")

    for candidate_file in sorted(candidate_files):
        iteration = int(candidate_file.split("outer_")[1].split("/")[0])
        headers, rows = load_csv(candidate_file)

        if headers and rows:
            for row in rows:
                candidate_data = dict(zip(headers, row))
                candidate_data["iteration"] = iteration
                all_candidates.append(candidate_data)

    if not all_candidates:
        raise ValueError("No candidate data found in experiment directory")

    # Extract mechanism parameters and metrics
    mechanism_params = [
        "fixed_quota",
        "prop_quota",
        "min_stock",
        "fine_amount",
        "ban_period",
    ]

    # Convert to float arrays
    param_data = {
        param: [float(c[param]) for c in all_candidates] for param in mechanism_params
    }
    objective_scores = [float(c["objective_score"]) for c in all_candidates]
    mean_rewards = [float(c["mean_reward"]) for c in all_candidates]
    sustainability_penalties = [
        float(c["sustainability_penalty"]) for c in all_candidates
    ]
    iterations = [int(c["iteration"]) for c in all_candidates]

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Color-code points by objective score
    scatter_kwargs = {"c": objective_scores, "cmap": "viridis", "alpha": 0.6, "s": 30}

    # Plot each mechanism parameter vs objective score
    for i, param in enumerate(mechanism_params):
        row, col = i // 2, i % 2
        if row < 2:  # First 4 parameters in top 2 rows
            sc = axes[row, col].scatter(
                param_data[param], objective_scores, **scatter_kwargs
            )
            axes[row, col].set_xlabel(param.replace("_", " ").title())
            axes[row, col].set_ylabel("Objective Score")
            axes[row, col].set_title(
                f"Objective Score vs {param.replace('_', ' ').title()}"
            )
            axes[row, col].grid(True, alpha=0.3)

    # Handle 5th parameter (ban_period) in bottom left
    sc = axes[2, 0].scatter(
        param_data["ban_period"], objective_scores, **scatter_kwargs
    )
    axes[2, 0].set_xlabel("Ban Period")
    axes[2, 0].set_ylabel("Objective Score")
    axes[2, 0].set_title("Objective Score vs Ban Period")
    axes[2, 0].grid(True, alpha=0.3)

    # Plot evolution over iterations in bottom right
    axes[2, 1].scatter(iterations, objective_scores, **scatter_kwargs)
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("Objective Score")
    axes[2, 1].set_title("Objective Score Evolution")
    axes[2, 1].grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label("Objective Score", rotation=270, labelpad=20)

    plt.suptitle("Mechanism Parameter Analysis - All Tested Candidates", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pareto_frontier_analysis(
    experiment_dir: str, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Pareto frontier analysis of reward vs sustainability.

    Args:
        experiment_dir: Path to experiment directory
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    import glob

    # Collect all candidate data
    all_candidates = []
    candidate_files = glob.glob(f"{experiment_dir}/outer_*/candidates.csv")

    for candidate_file in sorted(candidate_files):
        iteration = int(candidate_file.split("outer_")[1].split("/")[0])
        headers, rows = load_csv(candidate_file)

        if headers and rows:
            for row in rows:
                candidate_data = dict(zip(headers, row))
                candidate_data["iteration"] = iteration
                all_candidates.append(candidate_data)

    if not all_candidates:
        raise ValueError("No candidate data found")

    # Extract metrics
    mean_rewards = [float(c["mean_reward"]) for c in all_candidates]
    sustainability_penalties = [
        float(c["sustainability_penalty"]) for c in all_candidates
    ]
    objective_scores = [float(c["objective_score"]) for c in all_candidates]
    iterations = [int(c["iteration"]) for c in all_candidates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Reward vs Sustainability tradeoff
    scatter = ax1.scatter(
        sustainability_penalties,
        mean_rewards,
        c=objective_scores,
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    ax1.set_xlabel("Sustainability Penalty")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Reward vs Sustainability Tradeoff")
    ax1.grid(True, alpha=0.3)

    # Evolution over iterations
    ax2.scatter(
        iterations,
        objective_scores,
        c=objective_scores,
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Objective Score")
    ax2.set_title("Optimization Progress")
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Objective Score")

    plt.suptitle("Economic-Environmental Tradeoff Analysis", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_mechanism_parameter_pairs(
    experiment_dir: str, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot pairwise relationships between all mechanism parameters with objective score as color.

    Args:
        experiment_dir: Path to experiment directory
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    import glob

    # Collect all candidate data from all iterations
    all_candidates = []
    candidate_files = glob.glob(f"{experiment_dir}/outer_*/candidates.csv")

    for candidate_file in sorted(candidate_files):
        iteration = int(candidate_file.split("outer_")[1].split("/")[0])
        headers, rows = load_csv(candidate_file)

        if headers and rows:
            for row in rows:
                candidate_data = dict(zip(headers, row))
                candidate_data["iteration"] = iteration
                all_candidates.append(candidate_data)

    if not all_candidates:
        raise ValueError("No candidate data found in experiment directory")

    # Extract mechanism parameters and metrics
    mechanism_params = [
        "fixed_quota",
        "prop_quota",
        "min_stock",
        "fine_amount",
        "ban_period",
    ]
    param_labels = [
        "Fixed Quota",
        "Prop Quota",
        "Min Stock",
        "Fine Amount",
        "Ban Period",
    ]

    # Convert to float arrays
    param_data = {
        param: [float(c[param]) for c in all_candidates] for param in mechanism_params
    }
    objective_scores = [float(c["objective_score"]) for c in all_candidates]

    # Create pairwise plot matrix
    n_params = len(mechanism_params)
    fig, axes = plt.subplots(n_params, n_params, figsize=(20, 20))

    # Color mapping for objective scores
    scatter_kwargs = {"c": objective_scores, "cmap": "viridis", "alpha": 0.6, "s": 20}

    for i, param_x in enumerate(mechanism_params):
        for j, param_y in enumerate(mechanism_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram of parameter values colored by objective score
                # Use scatter plot positioned at histogram bins for color coding
                n_bins = 30
                counts, bins = np.histogram(param_data[param_x], bins=n_bins)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Calculate mean objective score for each bin
                bin_scores = []
                for k in range(len(bins) - 1):
                    mask = (np.array(param_data[param_x]) >= bins[k]) & (
                        np.array(param_data[param_x]) < bins[k + 1]
                    )
                    if np.any(mask):
                        bin_scores.append(np.mean(np.array(objective_scores)[mask]))
                    else:
                        bin_scores.append(0)

                # Plot histogram bars colored by mean objective score
                bars = ax.bar(
                    bin_centers,
                    counts,
                    width=bins[1] - bins[0],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Color bars based on bin scores
                if bin_scores:
                    norm = plt.Normalize(
                        vmin=min(objective_scores), vmax=max(objective_scores)
                    )
                    colors = plt.cm.viridis(norm(bin_scores))
                    for bar, color in zip(bars, colors):
                        bar.set_facecolor(color)

                ax.set_xlabel(param_labels[i])
                ax.set_ylabel("Count")

            else:
                # Off-diagonal: scatter plot of parameter pairs
                sc = ax.scatter(
                    param_data[param_x], param_data[param_y], **scatter_kwargs
                )
                ax.set_xlabel(param_labels[j])
                ax.set_ylabel(param_labels[i])
                ax.grid(True, alpha=0.3)

    # Add colorbar
    # Use the last scatter plot for colorbar reference
    cbar = plt.colorbar(sc, ax=axes, shrink=0.6, aspect=30)
    cbar.set_label("Objective Score", rotation=270, labelpad=20)

    plt.suptitle(
        "Mechanism Parameter Pairwise Analysis - All Tested Candidates", fontsize=16
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
