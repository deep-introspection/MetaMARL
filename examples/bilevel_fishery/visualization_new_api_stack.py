from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run
import re

# --------------------------
# helpers
# --------------------------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _finite(x: Any) -> Optional[float]:
    fx = _to_float(x)
    if fx is None or not np.isfinite(fx):
        return None
    return fx


def _summarize_dict_of_scalars(d: Dict[str, Any]) -> Dict[str, float]:
    vals: list[float] = []
    for v in (d or {}).values():
        fv = _finite(v)
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

def _cap_table(table: wandb.Table, max_rows: int) -> wandb.Table:
    # wandb.Table doesn't support popping rows; easiest is to rebuild from tail.
    if table is None:
        return table
    data = table.data  # list of rows
    if data is None or len(data) <= max_rows:
        return table
    tail = data[-max_rows:]
    new_t = wandb.Table(columns=table.columns)
    for r in tail:
        new_t.add_data(*r)
    return new_t

def _sanitize_key(s: str) -> str:
    # keep alnum, underscore, dash; replace everything else with underscore
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(s))


# Run-scoped (process-scoped) cache: one persistent W&B Table per run.
_LINE_TABLES: dict[int, wandb.Table] = {}


# --------------------------
# core extractors (new stack)
# --------------------------

def extract_episode_metrics_newstack(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    env = results.get("env_runners", {}) or {}

    # new-stack first, fallback to old-stack names if present
    return_mean = _finite(env.get("episode_return_mean"))
    if return_mean is None:
        return_mean = _finite(env.get("episode_reward_mean"))

    return {
        "episode_return_mean": return_mean,
        "episode_return_min": _finite(env.get("episode_return_min")),
        "episode_return_max": _finite(env.get("episode_return_max")),
        "episode_len_mean": _finite(env.get("episode_len_mean")),
        "episode_len_min": _finite(env.get("episode_len_min")),
        "episode_len_max": _finite(env.get("episode_len_max")),
        "num_episodes": _finite(env.get("num_episodes")),
        "num_episodes_lifetime": _finite(env.get("num_episodes_lifetime")),
    }

def extract_series_returns_newstack(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer module-level returns (RLModule ids), else fall back to agent ids.
    Your run shows:
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
        throughput = _finite(thr.get("throughput_since_last_reduce")) or _finite(
            thr.get("throughput_since_last_restore")
        )

    # These are dicts in your output; summarize them so you get something useful.
    agent_steps = env.get("num_agent_steps_sampled")
    agent_steps_lt = env.get("num_agent_steps_sampled_lifetime")

    agent_steps_sum = None
    agent_steps_lt_sum = None
    if isinstance(agent_steps, dict):
        agent_steps_sum = _finite(sum(_to_float(v) or 0.0 for v in agent_steps.values()))
    if isinstance(agent_steps_lt, dict):
        agent_steps_lt_sum = _finite(sum(_to_float(v) or 0.0 for v in agent_steps_lt.values()))

    return {
        "env_steps_this_iter": _finite(env.get("num_env_steps_sampled")),
        "env_steps_lifetime": _finite(env.get("num_env_steps_sampled_lifetime")),
        "agent_steps_this_iter_sum": agent_steps_sum,
        "agent_steps_lifetime_sum": agent_steps_lt_sum,
        "env_steps_throughput": throughput,

        "training_iteration_s": _finite(timers.get("training_iteration")),
        "training_step_s": _finite(timers.get("training_step")),
        "sample_s": _finite(timers.get("sample")),
        "learner_update_s": _finite(timers.get("learner_update_timer")),
        "weights_seq_no": _finite(env.get("weights_seq_no")),
    }

# TODO add learner extraction block
# --------------------------
# W&B logging (new stack)
# --------------------------

def plot_training_results_new_stack(
    wandb_run: Run,
    *,
    outer_iter: int,
    training_episode: int,
    results: Dict[str, Any],
    prefix: str = "rllib",
    series_ids: Optional[list[str]] = None,
    max_lines: int = 64,
    max_table_rows: int = 5000,
) -> None:
    if wandb_run is None or results is None:
        return

    # persistent W&B table per run
    run_key = id(wandb_run)
    table = _LINE_TABLES.get(run_key)
    if table is None:
        table = wandb.Table(columns=["outer_iter", "train_step", "series", "return_mean"])
        _LINE_TABLES[run_key] = table

    eps = extract_episode_metrics_newstack(results)
    perf = extract_perf_newstack(results)
    series_means = extract_series_returns_newstack(results)

    # choose which lines to log
    if series_ids is None:
        series_ids = list(series_means.keys())[:max_lines]

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

        # useful “is broadcasting happening?”
        f"{prefix}/perf/weights_seq_no": perf["weights_seq_no"],
    }

    # per-series line logging
    for sid in series_ids:
        fv = _finite(series_means.get(sid))
        if fv is None:
            continue
        sid_clean = _sanitize_key(sid)
        table.add_data(outer_iter, training_episode, sid_clean, fv)
        metrics[f"{prefix}/series/return_mean/{sid_clean}"] = fv

    # summarize across series
    summary = _summarize_dict_of_scalars({sid: series_means.get(sid) for sid in series_ids})
    for k, v in summary.items():
        metrics[f"{prefix}/series/return_mean_{k}"] = v

    # cap the table size
    table = _cap_table(table, max_table_rows)
    _LINE_TABLES[run_key] = table
    metrics[f"{prefix}/series/return_mean_table"] = table

    # drop Nones
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # make W&B step monotonic across outer iters
    global_step = outer_iter * 1_000_000 + training_episode
    wandb_run.log(metrics, step=global_step, commit=True)


def _get_env(result: dict) -> dict:
    return result.get("env_runners", {}) or {}

def _get_episode_return_mean(result: dict) -> float:
    env = _get_env(result)
    # new-stack
    v = _to_float(env.get("episode_return_mean"))
    if v is not None:
        return v
    # old-stack fallback
    v = _to_float(result.get("episode_reward_mean")) or _to_float(env.get("episode_reward_mean"))
    return v if v is not None else 0.0

def _get_env_steps(result: dict) -> tuple[int, int]:
    env = _get_env(result)
    steps_iter = _to_float(env.get("num_env_steps_sampled")) or _to_float(result.get("timesteps_this_iter"))
    steps_life = _to_float(env.get("num_env_steps_sampled_lifetime")) or _to_float(result.get("timesteps_total"))
    return int(steps_iter or 0), int(steps_life or 0)

def _get_policy_loss_if_present(result: dict) -> float:
    # New stack often doesn’t include learner stats in `result`.
    # Keep this as "best effort" with old-stack fallback.
    learner_info = (result.get("info") or {}).get("learner") or {}
    losses = []
    if isinstance(learner_info, dict):
        for _, policy_stats in learner_info.items():
            ls = (policy_stats or {}).get("learner_stats") or {}
            v = _to_float(ls.get("policy_loss"))
            if v is not None:
                losses.append(v)
    return float(np.mean(losses)) if losses else float("nan")
