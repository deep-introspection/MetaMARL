# =========================================
# file: core/reporting/bilevel_viz_reporter.py
# =========================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.sdk.wandb_run import Run


# =========================
#   Generic helpers
# =========================


def _ensure_dir(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (np.generic,)):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _wandb_log_fig(wandb_run: Run, key: str, fig: plt.Figure, step: int) -> None:
    if wandb_run is None:
        return
    wandb_run.log({key: wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)


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


# =========================
#   Mechanism param extract
# =========================


def extract_mechanism_params_full(
    mechanism_space: Any, candidate: Any
) -> Optional[dict[str, float]]:
    """
    Returns ALL mechanism params from decoded mechanism.
    Does NOT require any changes to Mechanism or MechanismSpace.
    """
    if mechanism_space is None or candidate is None:
        return None

    try:
        mech = mechanism_space.decode(candidate)
    except Exception:
        return None

    # Prefer explicit param_names() if present (you have this in FisheryMechanism)
    if hasattr(mech, "param_names") and callable(getattr(mech, "param_names")):
        out: dict[str, float] = {}
        for k in mech.param_names():
            v = getattr(mech, k, None)
            fv = _to_float(v)
            if fv is not None and np.isfinite(fv):
                out[k] = fv
        return out or None

    # Fallback: numeric fields in __dict__
    out: dict[str, float] = {}
    for k, v in vars(mech).items():
        if k.startswith("_"):
            continue
        fv = _to_float(v)
        if fv is not None and np.isfinite(fv):
            out[k] = fv
    return out or None


# =========================
#   Trajectory plots
# =========================


def _group_by_episode(trajectories: list[dict[str, Any]]) -> dict[int, dict[str, list]]:
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


def plot_combined_trial_analysis(
    trajectories: list[dict[str, Any]],
    mechanism_params: Optional[dict[str, float]] = None,
    sustainability_threshold: float = 0.1,
    title: str = "Trial Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    if not trajectories:
        raise ValueError("No trajectory data provided")

    episodes = _group_by_episode(trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fish
    for ep, data in episodes.items():
        axes[0, 0].plot(
            data["steps"], data["fish"], alpha=0.7, linewidth=1.5, label=f"Episode {ep}"
        )
    axes[0, 0].axhline(
        y=sustainability_threshold,
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Collapse threshold",
    )
    axes[0, 0].set_ylabel("Fish Population", fontsize=12)
    axes[0, 0].set_title("Fish Population Dynamics", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Algae
    for ep, data in episodes.items():
        axes[0, 1].plot(
            data["steps"],
            data["algae"],
            alpha=0.7,
            linewidth=1.5,
            label=f"Episode {ep}",
        )
    axes[0, 1].set_ylabel("Algae Population", fontsize=12)
    axes[0, 1].set_title("Algae Population Dynamics", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Harvest vs quota OR rewards
    has_harvest = any(bool(d.get("harvest")) for d in episodes.values())
    if has_harvest:
        for ep, data in episodes.items():
            if data.get("harvest"):
                axes[1, 0].plot(
                    data["steps"],
                    data["harvest"],
                    alpha=0.8,
                    linewidth=2,
                    label=f"Episode {ep} - Harvest",
                )
            if data.get("quota"):
                axes[1, 0].plot(
                    data["steps"],
                    data["quota"],
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
        has_rewards = any(bool(d.get("rewards")) for d in episodes.values())
        if has_rewards:
            for ep, data in episodes.items():
                if data.get("rewards"):
                    axes[1, 0].plot(
                        data["steps"],
                        data["rewards"],
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

    # Phase plot
    scatter = None
    for _, data in episodes.items():
        if data.get("rewards"):
            scatter = axes[1, 1].scatter(
                data["algae"],
                data["fish"],
                c=data["rewards"],
                cmap="viridis",
                alpha=0.6,
                s=30,
            )
        else:
            scatter = axes[1, 1].scatter(
                data["algae"],
                data["fish"],
                c=range(len(data["fish"])),
                cmap="viridis",
                alpha=0.6,
                s=30,
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
        info_text = "Mechanism: " + ", ".join(
            [f"{k}={v:.3f}" for k, v in mechanism_params.items()]
        )
        fig.text(0.5, 0.01, info_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =========================
#   ES plots (optimized dims)
# =========================

ALL_PARAM_NAMES = [
    "fixed_quota",
    "prop_quota",
    "min_stock",
    "fine_amount",
    "ban_period",
    "catch_prob",
]
ALL_PARAM_SCALES = [1.0, 1.0, 1.0, 5.0, 50.0, 1.0]
DEFAULT_OPTIMIZE_PARAMS = ["min_stock", "fine_amount"]


def plot_fitness_vs_parameters(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Fitness vs Mechanism Parameters",
    save_path: Optional[str] = None,
    optimize_params: Optional[list[str]] = None,
    param_scales: Optional[dict[str, float]] = None,
) -> plt.Figure:
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

    all_params, all_fitness, all_iters = [], [], []
    for iteration, (population, fitness) in population_history:
        for j in range(len(fitness)):
            all_params.append(population[j])
            all_fitness.append(fitness[j])
            all_iters.append(iteration)

    all_params = np.asarray(all_params)
    all_fitness = np.asarray(all_fitness)
    all_iters = np.asarray(all_iters)

    n_params = len(param_names)
    n_cols = min(3, n_params + 1)
    n_rows = (n_params + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    axes = axes.flatten()

    scatter = None
    for i, (name, scale) in enumerate(zip(param_names, scales)):
        ax = axes[i]
        param_vals = all_params[:, i] * scale
        scatter = ax.scatter(
            param_vals, all_fitness, c=all_iters, cmap="viridis", alpha=0.6, s=20
        )
        ax.set_xlabel(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Fitness", fontsize=11)
        ax.set_title(f"Fitness vs {name.replace('_', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3)

    ax = axes[n_params]
    best_per_iter: dict[int, float] = {}
    for iteration, (_, fitness) in population_history:
        best_per_iter[iteration] = max(
            best_per_iter.get(iteration, -np.inf), float(np.max(fitness))
        )

    iters = sorted(best_per_iter.keys())
    best_vals = [best_per_iter[k] for k in iters]
    ax.plot(iters, best_vals, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Best Fitness Over Iterations", fontsize=12)
    ax.grid(True, alpha=0.3)

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Iteration", fontsize=10)

    for k in range(n_params + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parameter_evolution(
    population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    title: str = "Parameter Evolution",
    save_path: Optional[str] = None,
    optimize_params: Optional[list[str]] = None,
    param_scales: Optional[dict[str, float]] = None,
) -> plt.Figure:
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

    iterations, best_params = [], []
    for iteration, (population, fitness) in population_history:
        best_idx = int(np.argmax(fitness))
        iterations.append(iteration)
        best_params.append(population[best_idx])

    best_params = np.asarray(best_params)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, scale) in enumerate(zip(param_names, scales)):
        param_vals = best_params[:, i] * scale
        ax.plot(
            iterations,
            param_vals,
            "o-",
            linewidth=2,
            markersize=5,
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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_es_metrics(
    metrics_history: list[dict[str, Any]],
    title: str = "ES Metrics Over Generations",
    save_path: Optional[str] = None,
) -> plt.Figure:
    if not metrics_history:
        raise ValueError("No metrics history provided")

    generations = [m.get("generation", i) for i, m in enumerate(metrics_history)]
    total_fines = [m.get("total_fines", 0.0) for m in metrics_history]
    mean_fish = [m.get("mean_fish", 0.0) for m in metrics_history]
    min_fish = [m.get("min_fish", 0.0) for m in metrics_history]
    collapse_rate = [m.get("mean_collapse_rate", 0.0) for m in metrics_history]
    best_fitness = [m.get("best_fitness", 0.0) for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(generations, total_fines, "o-", linewidth=2, markersize=5)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Total Fines", fontsize=11)
    ax.set_title("Total Fines per Generation", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(generations, mean_fish, "o-", linewidth=2, markersize=5, label="Mean Fish")
    ax.plot(generations, min_fish, "s--", linewidth=2, markersize=5, label="Min Fish")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fish Population (normalized)", fontsize=11)
    ax.set_title("Fish Population per Generation", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(generations, collapse_rate, "o-", linewidth=2, markersize=5)
    ax.axhline(y=0.0, linestyle="--", alpha=0.5, label="No collapse")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Collapse Rate", fontsize=11)
    ax.set_title("Mean Collapse Rate per Generation", fontsize=12)
    ax.set_ylim(-0.05, max(0.5, max(collapse_rate) * 1.1) if collapse_rate else 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(generations, best_fitness, "o-", linewidth=2, markersize=5)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_title("Best Fitness per Generation", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =========================
#   Inner training logging
# =========================

_MECH_REWARD_TABLES: dict[int, wandb.Table] = {}


def plot_training_results(
    wandb_run: Run,
    outer_iter: int,
    training_episode: int,
    results: dict,
    *,
    prefix_base: str = "ppo",
    num_mechanisms: int = 16,
) -> None:
    if wandb_run is None:
        return

    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}
    info = results.get("info", {}) or {}
    info_learner = info.get("learner", {}) or {}

    prefix = prefix_base

    run_key = id(wandb_run)
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

    policy_reward_mean = env.get("policy_reward_mean", {}) or {}
    policy_reward_min = env.get("policy_reward_min", {}) or {}
    policy_reward_max = env.get("policy_reward_max", {}) or {}

    for i in range(num_mechanisms):
        pid = f"fisher_policy_{i}"
        if pid in policy_reward_mean:
            fv = _to_float(policy_reward_mean[pid])
            if fv is not None and np.isfinite(fv):
                table.add_data(outer_iter, training_episode, f"m{i:02d}", fv)
                metrics[f"{prefix}/mech_reward_mean/m{i:02d}"] = fv

    for k, v in _summarize_dict_of_scalars(policy_reward_mean).items():
        metrics[f"{prefix}/policy_reward_mean_{k}"] = v
    for k, v in _summarize_dict_of_scalars(policy_reward_min).items():
        metrics[f"{prefix}/policy_reward_min_{k}"] = v
    for k, v in _summarize_dict_of_scalars(policy_reward_max).items():
        metrics[f"{prefix}/policy_reward_max_{k}"] = v

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

    metrics[f"{prefix}/mech_reward_mean_table"] = table
    metrics = {k: v for k, v in metrics.items() if v is not None}

    wandb_run.log(metrics, step=training_episode, commit=True)


# =========================
#   High-level reporter
# =========================

_ES_TABLES: dict[int, wandb.Table] = {}


@dataclass
class BilevelVizReporter:
    wandb_run: Optional[Run]
    output_dir: Optional[str] = None
    mechanism_space: Any = None
    optimize_params: Optional[list[str]] = None  # for ES-vector plots only
    num_mechanisms: int = 16

    def __post_init__(self):
        self._out_path = _ensure_dir(self.output_dir)
        self.population_history: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
        self.es_metrics_history: list[dict[str, Any]] = []

        if self.wandb_run is not None:
            wandb.define_metric("bilevel/outer_iter")
            wandb.define_metric("bilevel/*", step_metric="bilevel/outer_iter")
            wandb.define_metric("viz/*", step_metric="bilevel/outer_iter")
            wandb.define_metric("tables/*", step_metric="bilevel/outer_iter")
            wandb.define_metric("plots/*", step_metric="bilevel/outer_iter")
            wandb.define_metric("mech/*", step_metric="bilevel/outer_iter")

    def on_outer_iteration_end(
        self, *, outer_iter: int, outer: Any, outer_metrics: dict
    ) -> None:
        if self.wandb_run is None:
            return

        payload = {}
        trajectory = outer_metrics.get("best_trajectory")
        fitness = float(outer_metrics.get("best_fitness", -float("inf")))

        pop_history = outer_metrics.get("population_history", []) or []
        if pop_history:
            self.population_history.append((outer_iter, pop_history[-1]))

        self._collect_es_metrics(outer_iter=outer_iter, fitness=fitness, outer=outer)

        # --- trajectory fig
        if trajectory:
            mech_params = self._extract_mechanism_params_full(outer)
            fig = plot_combined_trial_analysis(
                trajectory,
                mechanism_params=mech_params,
                title=f"Iteration {outer_iter} (fitness={fitness:.4f})",
            )
            payload["viz/trajectory"] = wandb.Image(fig)
            plt.close(fig)

            if mech_params:
                for k, v in mech_params.items():
                    payload[f"mech/{k}"] = v

        # --- ES plots
        if self.population_history:
            param_scales = self._param_scales()

            fig1 = plot_fitness_vs_parameters(
                self.population_history,
                optimize_params=self.optimize_params,
                param_scales=param_scales,
            )
            payload["viz/fitness_vs_params"] = wandb.Image(fig1)
            plt.close(fig1)

            fig2 = plot_parameter_evolution(
                self.population_history,
                optimize_params=self.optimize_params,
                param_scales=param_scales,
            )
            payload["viz/param_evolution"] = wandb.Image(fig2)
            plt.close(fig2)

        # --- ES metrics fig + table
        if self.es_metrics_history:
            fig3 = plot_es_metrics(self.es_metrics_history)
            payload["viz/es_metrics"] = wandb.Image(fig3)
            plt.close(fig3)

            # build/append table
            run_key = id(self.wandb_run)
            table = _ES_TABLES.get(run_key)
            if table is None:
                cols = [
                    "generation",
                    "best_fitness",
                    "total_fines",
                    "mean_fish",
                    "min_fish",
                    "mean_collapse_rate",
                ]
                table = wandb.Table(columns=cols)
                _ES_TABLES[run_key] = table

            m = self.es_metrics_history[-1]
            table.add_data(
                int(m["generation"]),
                float(m["best_fitness"]),
                float(m["total_fines"]),
                float(m["mean_fish"]),
                float(m["min_fish"]),
                float(m["mean_collapse_rate"]),
            )

            payload["tables/es_metrics"] = table

            # (Optional) skip wandb.plot.line(...) until plots show up;
            # W&B UI can plot table columns directly anyway.

        # --- bilevel scalars
        payload["bilevel/outer_iter"] = outer_iter
        payload["bilevel/best_fitness"] = fitness

        self.wandb_run.log(payload, step=outer_iter, commit=True)

    def finish(self) -> None:
        return

    # ----- internal helpers -----

    def _param_scales(self) -> Optional[dict[str, float]]:
        if self.mechanism_space is None:
            return None
        return {
            "fine_amount": float(getattr(self.mechanism_space, "max_fine", 5.0)),
            "ban_period": float(getattr(self.mechanism_space, "max_ban", 50)),
        }

    def _extract_mechanism_params_full(self, outer: Any) -> Optional[dict[str, float]]:
        best_candidate = getattr(outer, "best_candidate", None)
        return extract_mechanism_params_full(self.mechanism_space, best_candidate)

    def _collect_es_metrics(
        self, *, outer_iter: int, fitness: float, outer: Any
    ) -> None:
        metrics = {
            "generation": int(outer_iter),
            "best_fitness": float(fitness),
            "total_fines": 0.0,
            "mean_fish": 0.0,
            "min_fish": 1.0,
            "mean_collapse_rate": 0.0,
        }

        env = getattr(outer, "env", None)
        env_metrics = getattr(env, "last_metrics", None) if env is not None else None

        if env_metrics:
            metrics["total_fines"] = float(
                sum(m.get("total_fines", 0.0) for m in env_metrics)
            )
            metrics["mean_fish"] = float(
                np.mean([m.get("mean_fish", 0.0) for m in env_metrics])
            )
            metrics["min_fish"] = float(
                min(m.get("min_fish", 1.0) for m in env_metrics)
            )
            metrics["mean_collapse_rate"] = float(
                np.mean([m.get("collapse_rate", 0.0) for m in env_metrics])
            )

        self.es_metrics_history.append(metrics)

    def _log_es_metrics_table(self, *, step: int) -> None:
        if self.wandb_run is None:
            return

        run_key = id(self.wandb_run)
        table = _ES_TABLES.get(run_key)
        if table is None:
            cols = [
                "generation",
                "best_fitness",
                "total_fines",
                "mean_fish",
                "min_fish",
                "mean_collapse_rate",
            ]
            table = wandb.Table(columns=cols)
            _ES_TABLES[run_key] = table

        # append ONLY latest row (don’t rebuild table each time)
        m = self.es_metrics_history[-1]
        table.add_data(
            int(m["generation"]),
            float(m["best_fitness"]),
            float(m["total_fines"]),
            float(m["mean_fish"]),
            float(m["min_fish"]),
            float(m["mean_collapse_rate"]),
        )

        self.wandb_run.log(
            {
                "tables/es_metrics": table,
                "plots/best_fitness": wandb.plot.line(
                    table, "generation", "best_fitness", title="Best fitness"
                ),
                "plots/mean_fish": wandb.plot.line(
                    table, "generation", "mean_fish", title="Mean fish"
                ),
                "plots/min_fish": wandb.plot.line(
                    table, "generation", "min_fish", title="Min fish"
                ),
                "plots/total_fines": wandb.plot.line(
                    table, "generation", "total_fines", title="Total fines"
                ),
                "plots/collapse_rate": wandb.plot.line(
                    table,
                    "generation",
                    "mean_collapse_rate",
                    title="Mean collapse rate",
                ),
            },
            step=step,
            commit=False,
        )
