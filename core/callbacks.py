import logging
import time
from typing import Any

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.evaluation.metrics import summarize_episodes
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    NUM_EPISODES
)

logger = logging.getLogger(__name__)

from core.envs.base import BaseEnv

# def tag_episode_with_env_idx(*, episode: MultiAgentEpisode, env_index: int, **kwargs):
#     episode_id = episode.id_
#     episode.id_ = f"{env_index}|{episode_id}"

# def tag_episode_with_env_identity(
def tag_episode_with_env_idx(
        *, 
        episode: MultiAgentEpisode, 
        env_runner: MultiAgentEnvRunner, 
        env: VectorMultiAgentEnv, 
        env_index: int, 
        **kwargs):
    
    # Get env identity
    env: BaseEnv = env_runner.env.envs[env_index].unwrapped

    # Access env seed and mechanism id
    if getattr(env, "mechanism_id", None) is None :
        raise RuntimeError("Env has no mechanism_id. It must be assigned at construction.")
    if getattr(env, "seed", None) is None :
        raise RuntimeError("Env has no seed. It must be assigned at construction.")
    if getattr(env, "env_id", None) is None : 
        env.env_id = env_index
    elif env.env_id != env_index:
        raise RuntimeError(
            f"Immutable env_id changed: {env.env_id}, new={env_index}"
        )
    
    mechanism_id = env.mechanism_id
    seed = env.seed
    policy_seed = env.policy_seed

    # set env id
    raw_episode_id = episode.id_
    

    # Store structured metadata for policy mapping / logging.
    if not raw_episode_id.startswith("env="):
        episode.id_ = f"env={env_index}|m={mechanism_id}|ps={policy_seed}|ss={seed}|raw={raw_episode_id}"

    # TODO inject policy_id to env for traceability and debugging

# TODO to be moved to a separate actor in the future for extensibility
def log_and_report_episode_metrics(
    *,
    episode: MultiAgentEpisode,
    env_runner: MultiAgentEnvRunner,
    env: VectorMultiAgentEnv,
    env_index: int,
    metrics_logger: MetricsLogger,
    **kwargs,
) -> None:
    env: BaseEnv = env_runner.env.envs[env_index].unwrapped

    metrics = env.logger.peek()
    env.reporter.report(metrics)
    reduced = env.logger.reduce()

    episode_id = episode.id_ .partition("|raw=")[0]

    # TODO could we just have the EnvRolloutSchema here ?
    metrics_logger.log_value(key=("by_episode", episode_id), value=reduced, reduce="item")

def _evaluate_with_fixed_duration_once(algo, eval_env_runner_group):
    # How many episodes/timesteps do we need to run?
    unit = algo.config.evaluation_duration_unit
    eval_cfg = algo.evaluation_config
    num_workers = algo.config.evaluation_num_env_runners
    force_reset = algo.config.evaluation_force_reset_envs_before_iteration
    time_out = algo.config.evaluation_sample_timeout_s

    # Remote function used on healthy EnvRunners to sample, get metrics, and
    # step counts.
    def _env_runner_remote(worker, num, round, iter, _force_reset):
        # Sample AND get_metrics, but only return metrics (and steps actually taken)
        # to save time. Also return the iteration to check, whether we should
        # discard and outdated result (from a slow worker).
        episodes = worker.sample(
            num_timesteps=(
                num[worker.worker_index] if unit == "timesteps" else None
            ),
            num_episodes=(num[worker.worker_index] if unit == "episodes" else None),
            force_reset=_force_reset and round == 0,
        )
        metrics = worker.get_metrics()
        env_steps = sum(e.env_steps() for e in episodes)
        agent_steps = sum(e.agent_steps() for e in episodes)
        return env_steps, agent_steps, metrics, iter

    all_metrics = []
    all_batches = []

    # How many episodes have we run (across all eval workers)?
    num_units_done = 0
    num_healthy_workers = eval_env_runner_group.num_healthy_remote_workers()

    env_steps = agent_steps = 0

    t_last_result = time.time()
    _round = -1
    algo_iteration = algo.iteration

    # In case all the remote evaluation workers die during a round of
    # evaluation, we need to stop.
    while num_healthy_workers > 0:
        units_left_to_do = algo.config.evaluation_duration - num_units_done
        if units_left_to_do <= 0:
            break

        _round += 1

        # New API stack -> EnvRunners return Episodes.
        if algo.config.enable_env_runner_and_connector_v2:
            _num = [None] + [  # [None]: skip idx=0 (local worker)
                (units_left_to_do // num_healthy_workers)
                + bool(i <= (units_left_to_do % num_healthy_workers))
                for i in range(1, num_workers + 1)
            ]

            results = (
                eval_env_runner_group.foreach_env_runner(
                    func=_env_runner_remote,
                    kwargs={
                        "num": _num,
                        "round": _round,
                        "iter": algo_iteration,
                        "_force_reset": force_reset,
                    },
                    local_env_runner=False,
                    timeout_seconds=time_out,
                )
            )

            # CHANGED
            # Exact-once behavior: Do not retry workers whose synchronous
            # request did not return before the timeout.
            if len(results) != num_healthy_workers:
                raise RuntimeError(
                    "Evaluation did not return exactly one result per "
                    "healthy EnvRunner: "
                    f"expected={num_healthy_workers}, "
                    f"received={len(results)}, "
                    f"timeout={time_out}s"
                )
            ## END OF CHANGE

            for env_s, ag_s, met, iter in results:
                if iter != algo.iteration:
                    continue
                env_steps += env_s
                agent_steps += ag_s
                all_metrics.append(met)
                num_units_done += (
                    (met[NUM_EPISODES].peek() if NUM_EPISODES in met else 0)
                    if unit == "episodes"
                    else (
                        env_s if algo.config.count_steps_by == "env_steps" else ag_s
                    )
                )
            if num_units_done != algo.config.evaluation_duration:
                raise RuntimeError(
                    "The single evaluation round returned fewer units than requested: "
                    f"requested={algo.config.evaluation_duration}, "
                    f"completed={num_units_done}, "
                    f"unit={unit}"
                )
            break
        # Old API stack -> RolloutWorkers return batches.
        else:
            units_per_healthy_remote_worker = (
                1
                if unit == "episodes"
                else eval_cfg.rollout_fragment_length
                * eval_cfg.num_envs_per_env_runner
            )
            # Select proper number of evaluation workers for this round.
            selected_eval_worker_ids = [
                worker_id
                for i, worker_id in enumerate(
                    eval_env_runner_group.healthy_worker_ids()
                )
                if i * units_per_healthy_remote_worker < units_left_to_do
            ]

            results = (
                algo.eval_env_runner_group.foreach_env_runner_async_fetch_ready(
                    func=lambda w: (w.sample(), w.get_metrics(), algo_iteration),
                    remote_worker_ids=selected_eval_worker_ids,
                    tag="env_runner_sample_and_get_metrics",
                )
            )
            # Make sure we properly time out if we have not received any results
            # for more than `time_out` seconds.
            time_now = time.time()
            if not results and time_now - t_last_result > time_out:
                break
            elif results:
                t_last_result = time_now
            for batch, metrics, iter in results:
                if iter != algo.iteration:
                    continue
                env_steps += batch.env_steps()
                agent_steps += batch.agent_steps()
                all_metrics.extend(metrics)
                if algo.reward_estimators:
                    # TODO: (kourosh) This approach will cause an OOM issue when
                    #  the dataset gets huge (should be ok for now).
                    all_batches.append(batch)

            # 1 episode per returned batch.
            if unit == "episodes":
                num_units_done += len(results)
            # n timesteps per returned batch.
            else:
                num_units_done = (
                    env_steps
                    if algo.config.count_steps_by == "env_steps"
                    else agent_steps
                )

        # Update correct number of healthy remote workers.
        num_healthy_workers = (
            algo.eval_env_runner_group.num_healthy_remote_workers()
        )

    if num_healthy_workers == 0:
        logger.warning(
            "Calling `sample()` on your remote evaluation worker(s) "
            "resulted in all workers crashing! Make sure a) your environment is not"
            " too unstable, b) you have enough evaluation workers "
            "(`config.evaluation(evaluation_num_env_runners=...)`) to cover for "
            "occasional losses, and c) you use the `config.fault_tolerance("
            "restart_failed_env_runners=True)` setting."
        )

    if not algo.config.enable_env_runner_and_connector_v2:
        env_runner_results = summarize_episodes(
            all_metrics,
            all_metrics,
            keep_custom_metrics=(
                algo.evaluation_config.keep_per_episode_custom_metrics
            ),
        )
        num_episodes = env_runner_results[NUM_EPISODES]
        eval_results = {
            ENV_RUNNER_RESULTS: env_runner_results,
        }
    else:
        algo.metrics.aggregate(
            all_metrics,
            key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS),
        )
        num_episodes = algo.metrics.peek(
            (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES),
            default=0,
            latest_merged_only=True,
        )
        # CHANGED:
        eval_results = algo.metrics.peek(
            EVALUATION_RESULTS,
            default={},
            latest_merged_only=True,
        )

    # Warn if results are empty, it could be that this is because the eval timesteps
    # are not enough to run through one full episode.
    if num_episodes == 0:
        logger.warning(
            "This evaluation iteration resulted in an empty set of episode summary "
            "results! It's possible that your configured duration timesteps are not"
            " enough to finish even a single episode. You have configured "
            f"{algo.config.evaluation_duration} "
            f"{algo.config.evaluation_duration_unit}. For 'timesteps', try "
            "increasing this value via the `config.evaluation(evaluation_duration="
            "...)` OR change the unit to 'episodes' via `config.evaluation("
            "evaluation_duration_unit='episodes')` OR try increasing the timeout "
            "threshold via `config.evaluation(evaluation_sample_timeout_s=...)` OR "
            "you can also set `config.evaluation_force_reset_envs_before_iteration`"
            " to False. However, keep in mind that in the latter case, the "
            "evaluation results may contain some episode stats generated with "
            "earlier weights versions."
        )

    # CHANGED:
    # A public custom evaluation function returns three values, not the
    # internal fixed-duration function's five values.
    return eval_results, env_steps, agent_steps



