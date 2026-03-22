from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import wandb
from wandb.sdk.wandb_run import Run
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


"""Visualization module for bilevel fishery experiments.

Provides functions to visualize fish/algae populations, harvest actions,
and regulatory mechanism effects over time.
"""


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
    has_harvest = any(
        "harvest" in data and data["harvest"] for data in episodes.values()
    )
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
        has_rewards = any(
            "rewards" in data and data["rewards"] for data in episodes.values()
        )
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
                0.5,
                0.5,
                "No harvest/reward data",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
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
        info_parts = [f"{k}={v:.3f}" for k, v in mechanism_params.items()]
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


ALL_PARAM_NAMES = [
    "fixed_quota",
    "prop_quota",
    "min_stock",
    "fine_amount",
    "ban_period",
    "catch_prob",
]
ALL_PARAM_SCALES = [1.0, 1.0, 1.0, 5.0, 50.0, 1.0]  # Denormalization factors

# Default: only optimized params
DEFAULT_OPTIMIZE_PARAMS = ["min_stock", "fine_amount"]


def plot_fitness_vs_parameters(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Fitness vs Mechanism Parameters",
    save_path: Optional[str] = None,
    optimize_params: Optional[list[str]] = None,
    param_scales: Optional[dict[str, float]] = None,
) -> plt.Figure:
    """Plot fitness as a function of each mechanism parameter.

    Args:
        population_history: List of (iteration, (population, fitness)) tuples
            where population is (N, num_params) and fitness is (N,)
        title: Plot title
        save_path: Path to save figure (optional)
        optimize_params: List of parameter names being optimized
        param_scales: Dict mapping param names to their max values (for denormalization)

    Returns:
        matplotlib Figure object
    """
    if not population_history:
        raise ValueError("No population history provided")

    param_names = optimize_params or DEFAULT_OPTIMIZE_PARAMS
    if param_scales:
        scales = [
            param_scales.get(p, ALL_PARAM_SCALES[ALL_PARAM_NAMES.index(p)])
            for p in param_names
        ]
    else:
        scales = [ALL_PARAM_SCALES[ALL_PARAM_NAMES.index(p)] for p in param_names]

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

    # Dynamic grid layout based on number of params
    n_params = len(param_names)
    n_cols = min(3, n_params + 1)  # +1 for fitness plot
    n_rows = (n_params + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    axes = axes.flatten()

    # Plot each parameter vs fitness
    scatter = None
    for i, (name, scale) in enumerate(zip(param_names, scales)):
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

    # Best fitness over iterations in next subplot
    ax = axes[n_params]
    best_per_iter = {}
    for iteration, (_, fitness) in population_history:
        if iteration not in best_per_iter:
            best_per_iter[iteration] = float(fitness.max())
        else:
            best_per_iter[iteration] = max(
                best_per_iter[iteration], float(fitness.max())
            )

    iters = sorted(best_per_iter.keys())
    best_vals = [best_per_iter[i] for i in iters]

    ax.plot(iters, best_vals, "o-", linewidth=2, markersize=6, color="tab:blue")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Best Fitness Over Iterations", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=axes[n_params])
        cbar.set_label("Iteration", fontsize=10)

    # Hide unused axes
    for i in range(n_params + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_es_metrics(
    metrics_history: list[dict[str, Any]],
    title: str = "ES Metrics Over Generations",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ES metrics (fines, fish population, collapse rate) over generations.

    Args:
        metrics_history: List of metric dicts per generation with keys:
            - generation: int
            - total_fines: float
            - mean_fish: float
            - min_fish: float
            - mean_collapse_rate: float
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    if not metrics_history:
        raise ValueError("No metrics history provided")

    generations = [m.get("generation", i) for i, m in enumerate(metrics_history)]
    total_fines = [m.get("total_fines", 0.0) for m in metrics_history]
    mean_fish = [m.get("mean_fish", 0.0) for m in metrics_history]
    min_fish = [m.get("min_fish", 0.0) for m in metrics_history]
    collapse_rate = [m.get("mean_collapse_rate", 0.0) for m in metrics_history]
    best_fitness = [m.get("best_fitness", 0.0) for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Fines over generations
    ax = axes[0, 0]
    ax.plot(generations, total_fines, "o-", linewidth=2, markersize=5, color="tab:red")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Total Fines", fontsize=11)
    ax.set_title("Total Fines per Generation", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Top right: Fish population (mean and min)
    ax = axes[0, 1]
    ax.plot(
        generations,
        mean_fish,
        "o-",
        linewidth=2,
        markersize=5,
        color="tab:blue",
        label="Mean Fish",
    )
    ax.plot(
        generations,
        min_fish,
        "s--",
        linewidth=2,
        markersize=5,
        color="tab:orange",
        label="Min Fish",
    )
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fish Population (normalized)", fontsize=11)
    ax.set_title("Fish Population per Generation", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Collapse rate
    ax = axes[1, 0]
    ax.plot(
        generations, collapse_rate, "o-", linewidth=2, markersize=5, color="tab:purple"
    )
    ax.axhline(y=0.0, color="green", linestyle="--", alpha=0.5, label="No collapse")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Collapse Rate", fontsize=11)
    ax.set_title("Mean Collapse Rate per Generation", fontsize=12)
    ax.set_ylim(-0.05, max(0.5, max(collapse_rate) * 1.1) if collapse_rate else 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Best fitness
    ax = axes[1, 1]
    ax.plot(
        generations, best_fitness, "o-", linewidth=2, markersize=5, color="tab:green"
    )
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Best Fitness per Generation", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_parameter_evolution(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Parameter Evolution",
    save_path: Optional[str] = None,
    optimize_params: Optional[list[str]] = None,
    param_scales: Optional[dict[str, float]] = None,
) -> plt.Figure:
    """Plot how the best mechanism parameters evolve over iterations.

    Args:
        population_history: List of (iteration, (population, fitness)) tuples
        title: Plot title
        save_path: Path to save figure (optional)
        optimize_params: List of parameter names being optimized
        param_scales: Dict mapping param names to their max values (for denormalization)

    Returns:
        matplotlib Figure object
    """
    if not population_history:
        raise ValueError("No population history provided")

    param_names = optimize_params or DEFAULT_OPTIMIZE_PARAMS
    if param_scales:
        scales = [
            param_scales.get(p, ALL_PARAM_SCALES[ALL_PARAM_NAMES.index(p)])
            for p in param_names
        ]
    else:
        scales = [ALL_PARAM_SCALES[ALL_PARAM_NAMES.index(p)] for p in param_names]

    # Extract best candidate per iteration
    iterations = []
    best_params = []

    for iteration, (population, fitness) in population_history:
        best_idx = int(np.argmax(fitness))
        iterations.append(iteration)
        best_params.append(population[best_idx])

    best_params = np.array(best_params)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))

    for i, (name, scale, color) in enumerate(zip(param_names, scales, colors)):
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


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (np.generic,)):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _summarize_dict_of_scalars(d: Dict[str, Any]) -> Dict[str, float]:
    vals: list[float] = []
    for v in (d or {}).values():
        fv = _to_float(v)
        if fv is not None and np.isfinite(fv):
            vals.append(fv)
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "n": float(len(arr)),
    }


def _aggregate_learner_stats(
    info_learner: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    per_stat: Dict[str, list[float]] = {}
    for _, policy_block in (info_learner or {}).items():
        if not isinstance(policy_block, dict):
            continue
        ls = policy_block.get("learner_stats", {})
        if not isinstance(ls, dict):
            continue
        for k, v in ls.items():
            fv = _to_float(v)
            if fv is None or not np.isfinite(fv):
                continue
            per_stat.setdefault(k, []).append(fv)

    out: Dict[str, Dict[str, float]] = {}
    for k, vals in per_stat.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
            "n": float(len(arr)),
        }
    return out


_MECH_REWARD_TABLES: dict[int, wandb.Table] = {}  # run-scoped cache


def plot_training_results(
    wandb_run: Run,
    outer_iter: int,
    training_episode: int,
    results: dict,
    *,
    prefix_base: str = "ppo",
    num_mechanisms: int = 16,
) -> None:
    """
    Fixed:
      - No attrs on wandb_run (W&B forbids it)
      - Persistent table per run (cached in module dict)
      - Stable metric names (no outer_iter in metric key path)
      - ONE wandb_run.log call per step
    """
    if wandb_run is None:
        return

    # RLlib blocks
    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}
    info = results.get("info", {}) or {}
    info_learner = info.get("learner", {}) or {}

    prefix = prefix_base  # keep stable keys so charts show up easily

    # --- persistent table per run ---
    run_key = id(wandb_run)  # stable for life of this process
    table = _MECH_REWARD_TABLES.get(run_key)
    if table is None:
        table = wandb.Table(columns=["outer_iter", "ppo_step", "mech", "reward_mean"])
        _MECH_REWARD_TABLES[run_key] = table

    metrics: Dict[str, Any] = {
        f"{prefix}/outer_iter": outer_iter,
        f"{prefix}/ppo_step": training_episode,
        f"{prefix}/episode_reward_mean": _to_float(env.get("episode_reward_mean")),
        f"{prefix}/episode_reward_min": _to_float(env.get("episode_reward_min")),
        f"{prefix}/episode_reward_max": _to_float(env.get("episode_reward_max")),
        f"{prefix}/episode_len_mean": _to_float(env.get("episode_len_mean")),
        f"{prefix}/num_episodes": _to_float(
            env.get("num_episodes", env.get("episodes_this_iter"))
        ),
    }

    # Per-policy rewards (may be empty depending on RLlib config)
    policy_reward_mean = env.get("policy_reward_mean", {}) or {}
    policy_reward_min = env.get("policy_reward_min", {}) or {}
    policy_reward_max = env.get("policy_reward_max", {}) or {}

    # Add points (this is what makes the line chart actually form lines)
    for i in range(num_mechanisms):
        pid = f"fisher_policy_{i}"
        if pid in policy_reward_mean:
            fv = _to_float(policy_reward_mean[pid])
            if fv is not None and np.isfinite(fv):
                table.add_data(outer_iter, training_episode, f"m{i:02d}", fv)
                metrics[f"{prefix}/mech_reward_mean/m{i:02d}"] = fv  # optional scalars

    # Summary stats across policies
    for k, v in _summarize_dict_of_scalars(policy_reward_mean).items():
        metrics[f"{prefix}/policy_reward_mean_{k}"] = v
    for k, v in _summarize_dict_of_scalars(policy_reward_min).items():
        metrics[f"{prefix}/policy_reward_min_{k}"] = v
    for k, v in _summarize_dict_of_scalars(policy_reward_max).items():
        metrics[f"{prefix}/policy_reward_max_{k}"] = v

    # Perf/timers
    metrics[f"{prefix}/perf/env_steps_this_iter"] = _to_float(
        results.get("num_env_steps_sampled_this_iter")
    )
    metrics[f"{prefix}/perf/env_steps_per_sec"] = _to_float(
        results.get("num_env_steps_sampled_throughput_per_sec")
    )

    t = _to_float(timers.get("training_iteration_time_ms"))
    if t is not None:
        metrics[f"{prefix}/perf/training_iter_time_s"] = t / 1000.0
    t = _to_float(timers.get("sample_time_ms"))
    if t is not None:
        metrics[f"{prefix}/perf/sample_time_s"] = t / 1000.0
    t = _to_float(timers.get("learn_time_ms"))
    if t is not None:
        metrics[f"{prefix}/perf/learn_time_s"] = t / 1000.0
    t = _to_float(timers.get("synch_weights_time_ms"))
    if t is not None:
        metrics[f"{prefix}/perf/sync_weights_time_s"] = t / 1000.0

    metrics[f"{prefix}/perf/learn_throughput"] = _to_float(
        timers.get("learn_throughput")
    )

    # Learner stats aggregated across policies
    learner_stats_summary = _aggregate_learner_stats(info_learner)
    for stat_name, stats in learner_stats_summary.items():
        for agg, val in stats.items():
            metrics[f"{prefix}/learner/{stat_name}_{agg}"] = val

    for key in [
        "kl",
        "entropy",
        "vf_loss",
        "policy_loss",
        "total_loss",
        "vf_explained_var",
        "grad_gnorm",
        "cur_lr",
        "cur_kl_coeff",
    ]:
        if key in learner_stats_summary:
            metrics[f"{prefix}/ppo/{key}"] = learner_stats_summary[key]["mean"]

    # Add the “one chart with 16 lines”
    metrics[f"{prefix}/mech_reward_mean_table"] = table
    # Most compatible across wandb versions: use positional args.

    # clean out None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    wandb_run.log(metrics, step=training_episode, commit=True)
