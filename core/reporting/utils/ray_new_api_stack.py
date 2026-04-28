from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from core.utils import safe_ratio, to_float, finite, sanitize_key

logger = logging.getLogger(__name__)


def _cap_table(table: wandb.Table, max_rows: int) -> wandb.Table:
    if table is None:
        return table
    data = table.data
    if data is None or len(data) <= max_rows:
        return table
    tail = data[-max_rows:]
    new_t = wandb.Table(columns=table.columns)
    for r in tail:
        new_t.add_data(*r)
    return new_t


def _summarize_dict_of_scalars(d: Dict[str, Any]) -> Dict[str, float]:
    vals: list[float] = []
    for v in (d or {}).values():
        fv = finite(v)
        if fv is not None:
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


def _global_step(outer_iter: int, train_step: int) -> int:
    # monotonic across outer iters
    return int(outer_iter) * 1_000_000 + int(train_step)


def extract_episode_metrics_newstack(
    results: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    env = results.get("env_runners", {}) or {}

    # new-stack first, fallback to old-stack names if present
    return_mean = finite(env.get("episode_return_mean"))
    if return_mean is None:
        return_mean = finite(env.get("episode_reward_mean"))

    return {
        "episode_return_mean": return_mean,
        "episode_return_min": finite(env.get("episode_return_min")),
        "episode_return_max": finite(env.get("episode_return_max")),
        "episode_len_mean": finite(env.get("episode_len_mean")),
        "episode_len_min": finite(env.get("episode_len_min")),
        "episode_len_max": finite(env.get("episode_len_max")),
        "num_episodes": finite(env.get("num_episodes")),
        "num_episodes_lifetime": finite(env.get("num_episodes_lifetime")),
    }


def extract_series_returns_newstack(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer module-level returns (RLModule ids), else fall back to agent ids.
    - env_runners["module_episode_returns_mean"] e.g. {"fisher_policy_0": ...}
    - env_runners["agent_episode_returns_mean"]  e.g. {"fisher:0": ...}
    """
    env = results.get("env_runners", {}) or {}

    module_means = env.get("module_episode_returns_mean")
    if isinstance(module_means, dict) and module_means:
        return module_means

    agent_means = env.get("agent_episode_returns_mean")
    if isinstance(agent_means, dict) and agent_means:
        return agent_means

    return {}


def extract_perf_newstack(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}

    thr = env.get("num_env_steps_sampled_lifetime_throughput")
    throughput = None
    if isinstance(thr, dict):
        throughput = finite(thr.get("throughput_since_last_reduce")) or finite(
            thr.get("throughput_since_last_restore")
        )

    agent_steps = env.get("num_agent_steps_sampled")
    agent_steps_lt = env.get("num_agent_steps_sampled_lifetime")

    agent_steps_sum = None
    agent_steps_lt_sum = None
    if isinstance(agent_steps, dict):
        agent_steps_sum = finite(
            sum((to_float(v) or 0.0) for v in agent_steps.values())
        )
    if isinstance(agent_steps_lt, dict):
        agent_steps_lt_sum = finite(
            sum((to_float(v) or 0.0) for v in agent_steps_lt.values())
        )

    return {
        "env_steps_this_iter": finite(env.get("num_env_steps_sampled")),
        "env_steps_lifetime": finite(env.get("num_env_steps_sampled_lifetime")),
        "agent_steps_this_iter_sum": agent_steps_sum,
        "agent_steps_lifetime_sum": agent_steps_lt_sum,
        "env_steps_throughput": throughput,
        "training_iteration_s": finite(timers.get("training_iteration")),
        "training_step_s": finite(timers.get("training_step")),
        "sample_s": finite(timers.get("sample")),
        "learner_update_s": finite(timers.get("learner_update_timer")),
        "weights_seq_no": finite(env.get("weights_seq_no")),
    }


def extract_learner_metrics_newstack(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns:
      { learner_id: {metric_name: float_or_none, ...}, ... }
    where learner_id is something like 'fisher_policy_0', '__all_modules__', etc.
    """
    learners = results.get("learners", {}) or {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not isinstance(learners, dict):
        return out

    learner_group = results.get("learner_group", {}) or {}
    mean_training_calls_since_sync = finite(
        results.get("mean_num_training_step_calls_since_last_synch_worker_weights")
    )
    outstanding_async_reqs = finite(
        learner_group.get("actor_manager_num_outstanding_async_reqs")
    )

    # Useful global proxy source from __all_modules__
    all_modules_stats = learners.get("__all_modules__", {}) or {}
    learner_queue_wait = finite(
        all_modules_stats.get("learner_thread_in_queue_wait_timer")
    )

    for learner_id, stats in learners.items():
        if not isinstance(stats, dict):
            continue

        m: Dict[str, Optional[float]] = {}

        # Log all scalar-ish keys automatically (scales as RLlib changes).
        # Skip nested dicts/lists, and skip NaN/inf.
        for k, v in stats.items():
            if isinstance(v, (dict, list, tuple)):
                continue
            fv = finite(v)
            if fv is None:
                continue
            m[str(k)] = fv

        # Optionally unpack throughput dict if present (nice to have)
        thr = stats.get("num_module_steps_trained_lifetime_throughput")
        if isinstance(thr, dict):
            m["module_steps_throughput_since_last_reduce"] = finite(
                thr.get("throughput_since_last_reduce")
            )
            m["module_steps_throughput_since_last_restore"] = finite(
                thr.get("throughput_since_last_restore")
            )

        # Derived metrics
        # policy_relative_entropy = entropy / entropy coeff
        entropy = m.get("entropy")
        entropy_coeff = m.get("curr_entropy_coeff")

        m["policy_relative_entropy"] = safe_ratio(entropy, entropy_coeff)

        # optional but often useful
        if finite(entropy) is not None and finite(entropy_coeff) is not None:
            m["entropy_pressure"] = float(entropy) * float(entropy_coeff)

        # sample staleness proxy
        # best available ingredients from your dump:
        # - diff_num_grad_updates_vs_sampler_policy
        # - mean_num_training_step_calls_since_last_synch_worker_weights
        # - learner_group.actor_manager_num_outstanding_async_reqs
        # - learner_thread_in_queue_wait_timer (__all_modules__)
        lag1 = m.get("diff_num_grad_updates_vs_sampler_policy")
        lag2 = mean_training_calls_since_sync
        lag3 = outstanding_async_reqs
        lag4 = learner_queue_wait

        parts = [x for x in (lag1, lag2, lag3, lag4) if x is not None]
        m["sample_staleness"] = float(sum(parts)) if parts else None

        out[str(learner_id)] = m

    return out


# --------------------------
# W&B: plot caches
# --------------------------

# Returns plot table: one per run
_RETURNS_TABLES: dict[int, wandb.Table] = {}

# Learner metric plot tables: per run -> per metric -> table
# table columns: ["step", "outer_iter", "train_step", "policy", "value"]
_LEARNER_METRIC_TABLES: dict[int, dict[str, wandb.Table]] = {}

# Keys we never want to plot (even if whitelisted by substring)
_DEFAULT_SKIP_PLOT_KEYS = {
    "num_module_steps_trained",
    "num_module_steps_trained_lifetime",
    "num_trainable_parameters",
    "num_non_trainable_parameters",
    "weights_seq_no",
    "diff_num_grad_updates_vs_sampler_policy",
    "module_train_batch_size_mean",
    "module_steps_throughput_since_last_reduce",
    "module_steps_throughput_since_last_restore",
}

# Only make plots for these (substring match on lower-cased key).
# Keep small to avoid W&B UI spam.
_DEFAULT_LEARNER_PLOT_WHITELIST = {
    "total_loss",
    "policy_loss",
    "vf_loss",
    "entropy",
    # "curr_entropy_coeff",
    "policy_relative_entropy",
    "entropy_pressure",
    "kl",
    # "curr_kl_coeff",
    "vf_explained_var",
    "grad_gnorm",
    "sample_staleness",
    # "lr",
    # "learning_rate",
    # "default_optimizer_learning_rate",
}


def _should_plot_metric(metric_key: str, whitelist: set[str]) -> bool:
    return str(metric_key).lower() in whitelist


def _table_to_line_series_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """
    Convert a wandb.Table into (xs, ys, keys) for wandb.plot.line_series(xs=..., ys=..., keys=...).

    The table can contain extra columns; we only use x_col/y_col/series_col.
    """
    if table is None or table.data is None:
        return [], [], []

    cols = list(table.columns)
    try:
        ix = cols.index(x_col)
        iy = cols.index(y_col)
        iser = cols.index(series_col)
    except ValueError as e:
        raise ValueError(
            f"Table missing required columns. Have={cols}, need={[x_col, series_col, y_col]}"
        ) from e

    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])
        s = row[iser]
        if x is None or y is None:
            continue
        key = str(s)
        grouped.setdefault(key, []).append((float(x), float(y)))

    keys = sorted(grouped.keys())
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for k in keys:
        pts = sorted(grouped[k], key=lambda t: t[0])
        xs.append([p[0] for p in pts])
        ys.append([p[1] for p in pts])

    return xs, ys, keys


def _log_multiline_table_as_plot(
    *,
    wandb_run: Run,
    prefix: str,
    plot_name: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    step: int,
    max_rows: int,
    title: str,
) -> None:
    table = _cap_table(table, max_rows)

    xs, ys, keys = _table_to_line_series_arrays(
        table,
        x_col=x_col,
        y_col=y_col,
        series_col=series_col,
    )

    payload: Dict[str, Any] = {}

    # Only create plot if we have data.
    if keys and xs and ys:
        payload[f"{prefix}/plots/{plot_name}"] = wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=title,
            xname=x_col,
        )

    wandb_run.log(payload, step=step, commit=False)


# --------------------------
# main logger
# --------------------------


def plot_training_results_new_stack(
    wandb_run: Run,
    *,
    outer_iter: int,
    training_episode: int,
    results: Dict[str, Any],
    prefix: str = "rllib",
    # plotting controls
    max_lines_returns: int = 64,
    max_rows_returns: int = 50_000,
    max_rows_per_learner_metric: int = 50_000,
    include_all_modules_in_learner_plots: bool = False,  # usually False
    skip_learner_plot_keys: Optional[set[str]] = None,
    learner_plot_whitelist: Optional[set[str]] = None,
    # UI spam controls
    log_per_policy_learner_scalars: bool = False,
    learner_scalar_whitelist: Optional[set[str]] = None,
) -> None:
    """
    Logs:
      - scalar summaries (episode/perf)
      - per-series return scalars (optional)
      - OPTIONAL per-policy learner scalars (off by default to reduce UI noise)
      - Multi-line plots:
          * returns: one plot with lines=policies/agents
          * learner metrics: ONE plot per metric with lines=policies (WHITELISTED)
    """
    if wandb_run is None or results is None:
        return

    gs = _global_step(outer_iter, training_episode)

    eps = extract_episode_metrics_newstack(results)
    perf = extract_perf_newstack(results)
    series_means = extract_series_returns_newstack(results)
    learner_by_policy = extract_learner_metrics_newstack(results)

    # --------------------------
    # 1) scalar logging (small + stable)
    # --------------------------
    metrics: Dict[str, Any] = {
        f"{prefix}/outer_iter": outer_iter,
        f"{prefix}/train_step": training_episode,
        f"{prefix}/episode/return_mean": eps["episode_return_mean"],
        f"{prefix}/episode/return_min": eps["episode_return_min"],
        f"{prefix}/episode/return_max": eps["episode_return_max"],
        f"{prefix}/episode/len_mean": eps["episode_len_mean"],
        f"{prefix}/episode/num_episodes": eps["num_episodes"],
        f"{prefix}/perf/env_steps_this_iter": perf["env_steps_this_iter"],
        f"{prefix}/perf/env_steps_lifetime": perf["env_steps_lifetime"],
        f"{prefix}/perf/agent_steps_this_iter_sum": perf["agent_steps_this_iter_sum"],
        f"{prefix}/perf/agent_steps_lifetime_sum": perf["agent_steps_lifetime_sum"],
        f"{prefix}/perf/env_steps_throughput": perf["env_steps_throughput"],
        f"{prefix}/perf/training_iteration_s": perf["training_iteration_s"],
        f"{prefix}/perf/training_step_s": perf["training_step_s"],
        f"{prefix}/perf/sample_s": perf["sample_s"],
        f"{prefix}/perf/learner_update_s": perf["learner_update_s"],
        f"{prefix}/perf/weights_seq_no": perf["weights_seq_no"],
    }

    # per-series scalar metrics + summary (useful; not too spammy if capped)
    if isinstance(series_means, dict) and series_means:
        series_ids = list(series_means.keys())[:max_lines_returns]
        for sid in series_ids:
            fv = finite(series_means.get(sid))
            if fv is None:
                continue
            sid_clean = sanitize_key(sid)
            metrics[f"{prefix}/series/return_mean/{sid_clean}"] = fv

        summary = _summarize_dict_of_scalars(
            {sid: series_means.get(sid) for sid in series_ids}
        )
        for k, v in summary.items():
            metrics[f"{prefix}/series/return_mean_{k}"] = v

    # OPTIONAL: per-policy learner scalars (OFF by default)
    if log_per_policy_learner_scalars and isinstance(learner_by_policy, dict):
        scalar_wl = learner_scalar_whitelist or _DEFAULT_LEARNER_PLOT_WHITELIST
        for learner_id, lm in (learner_by_policy or {}).items():
            learner_clean = sanitize_key(learner_id)
            for k, v in (lm or {}).items():
                if v is None:
                    continue
                if not _should_plot_metric(k, scalar_wl):
                    continue
                k_clean = sanitize_key(k)
                metrics[f"{prefix}/learner/{learner_clean}/{k_clean}"] = v

    # drop Nones
    metrics = {k: v for k, v in metrics.items() if v is not None}
    wandb_run.log(metrics, step=gs, commit=False)

    # --------------------------
    # 2) MULTI-LINE PLOT: returns (ONE plot, lines=series)
    # --------------------------
    if isinstance(series_means, dict) and series_means:
        run_key = id(wandb_run)
        t = _RETURNS_TABLES.get(run_key)
        if t is None:
            t = wandb.Table(
                columns=["step", "outer_iter", "train_step", "series", "value"]
            )
            _RETURNS_TABLES[run_key] = t

        series_ids = list(series_means.keys())[:max_lines_returns]
        for sid in series_ids:
            fv = finite(series_means.get(sid))
            if fv is None:
                continue
            t.add_data(
                int(gs), int(outer_iter), int(training_episode), str(sid), float(fv)
            )

        _log_multiline_table_as_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            plot_name="returns/all_series_return_mean",
            table=t,
            x_col="step",
            y_col="value",
            series_col="series",
            step=gs,
            max_rows=max_rows_returns,
            title="Return mean (all policies/series)",
        )

    # --------------------------
    # 3) MULTI-LINE PLOTS: learner metrics (ONE plot per metric; lines=policies)
    #     IMPORTANT: whitelist metrics to avoid UI spam
    # --------------------------
    skip_keys = set(skip_learner_plot_keys or set()) | _DEFAULT_SKIP_PLOT_KEYS
    plot_wl = set(learner_plot_whitelist or _DEFAULT_LEARNER_PLOT_WHITELIST)

    if isinstance(learner_by_policy, dict) and learner_by_policy:
        run_key = id(wandb_run)
        per_metric = _LEARNER_METRIC_TABLES.setdefault(run_key, {})
        touched: set[str] = set()

        for learner_id, lm in learner_by_policy.items():
            if (
                not include_all_modules_in_learner_plots
            ) and learner_id == "__all_modules__":
                continue

            series_name = str(learner_id)
            for k, v in (lm or {}).items():
                if v is None:
                    continue
                if k in skip_keys:
                    continue
                if not _should_plot_metric(k, plot_wl):
                    continue  # <- key: do NOT plot every learner scalar

                metric_name = sanitize_key(k)
                t = per_metric.get(metric_name)
                if t is None:
                    t = wandb.Table(
                        columns=["step", "outer_iter", "train_step", "policy", "value"]
                    )
                    per_metric[metric_name] = t

                t.add_data(
                    int(gs),
                    int(outer_iter),
                    int(training_episode),
                    series_name,
                    float(v),
                )
                touched.add(metric_name)

        for metric_name in touched:
            t = per_metric[metric_name]
            _log_multiline_table_as_plot(
                wandb_run=wandb_run,
                prefix=prefix,
                plot_name=f"learner/{metric_name}",
                table=t,
                x_col="step",
                y_col="value",
                series_col="policy",
                step=gs,
                max_rows=max_rows_per_learner_metric,
                title=f"{metric_name} (all policies)",
            )

    # finalize
    wandb_run.log({}, step=gs, commit=False)
