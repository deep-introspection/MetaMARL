"""RLlib hooks used by the inner optimizer.

Three pieces live here. ``tag_episode_with_env_idx`` is an
``on_episode_created`` callback that stamps each episode ID with the identity
of the sub-environment that produced it, which is how the ``policy_mapping_fn``
in ``RayOptimizerConfig`` picks the right RLModule.
``log_and_report_episode_metrics`` is an episode-end callback that flushes the
environment's metric logger to its reporter and into RLlib's
``MetricsLogger``. ``_evaluate_with_fixed_duration_once`` is a
``custom_evaluation_function`` that replaces RLlib's fixed-duration evaluation
loop with a strict single round.
"""

import logging
from typing import Any

from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS, NUM_EPISODES
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from core.envs.base import BaseEnv

logger = logging.getLogger(__name__)

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
    **kwargs: Any,
) -> None:
    """Rewrite the episode ID to carry the sub-environment identity.

    Registered as ``callbacks(on_episode_created=...)`` on the new API stack.
    The ID becomes ``"env=<i>|m=<mech>|ps=<policy_seed>|ss=<seed>|raw=<id>"``
    where ``i`` is the vector index of the sub-environment in its runner,
    ``mech`` the mechanism candidate index, ``policy_seed`` the seed of the
    RLModule that should act, ``seed`` the environment dynamics seed, and
    ``raw`` RLlib's original random ID (kept so the ID stays unique).

    Why go through the episode ID: on the new API stack the
    ``policy_mapping_fn`` only receives ``(agent_id, episode)``, with no
    direct handle on the environment instance, so the environment's identity
    has to be smuggled into the episode itself. Encoding it in the ID also
    makes the mapping visible in logs and metrics. IDs already starting with
    ``env=`` are left untouched.

    Parameters
    ----------
    episode : MultiAgentEpisode
        Freshly created episode whose ``id_`` is rewritten in place.
    env_runner : MultiAgentEnvRunner
        Runner owning the vectorised environment; the sub-env is fetched as
        ``env_runner.env.envs[env_index].unwrapped``.
    env : VectorMultiAgentEnv
        Vector env passed by RLlib; immediately shadowed by the unwrapped
        sub-environment and otherwise unused.
    env_index : int
        Position of the sub-environment in the vector env.
    **kwargs
        Other callback arguments, ignored.

    Raises
    ------
    RuntimeError
        If the sub-environment has no ``mechanism_id`` or ``seed`` (they must
        be injected by the env creator), or if its ``env_id`` was already set
        to a different index. ``policy_seed`` is read without such a check.

    Notes
    -----
    Side effect: on the first episode the sub-environment's ``env_id`` is set
    to ``env_index``; ``EnvStepContext.env_id`` is ``None`` before that.
    """

    # Get env identity
    env: BaseEnv = env_runner.env.envs[env_index].unwrapped

    # Access env seed and mechanism id
    if getattr(env, "mechanism_id", None) is None:
        raise RuntimeError(
            "Env has no mechanism_id. It must be assigned at construction."
        )
    if getattr(env, "seed", None) is None:
        raise RuntimeError("Env has no seed. It must be assigned at construction.")
    if getattr(env, "env_id", None) is None:
        env.env_id = env_index
    elif env.env_id != env_index:
        raise RuntimeError(f"Immutable env_id changed: {env.env_id}, new={env_index}")

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
    **kwargs: Any,
) -> None:
    """Report the sub-environment's episode metrics and hand them to RLlib.

    Intended for the ``on_episode_end`` hook. The sub-environment's
    ``MetricLogger`` is peeked and sent to the env-level reporter, then
    reduced and stored in RLlib's ``MetricsLogger`` under
    ``("by_episode", <episode_id>)`` with ``reduce="item"``, where
    ``<episode_id>`` is the structured prefix written by
    ``tag_episode_with_env_idx`` (the ``|raw=...`` suffix is stripped).
    ``build_rollout`` later regroups these entries by mechanism and seed.

    Parameters
    ----------
    episode : MultiAgentEpisode
        Episode that just ended; only its ``id_`` is used.
    env_runner : MultiAgentEnvRunner
        Runner owning the vectorised environment; the sub-env is fetched as
        ``env_runner.env.envs[env_index].unwrapped``.
    env : VectorMultiAgentEnv
        Vector env passed by RLlib; shadowed by the unwrapped sub-environment.
    env_index : int
        Position of the sub-environment in the vector env.
    metrics_logger : MetricsLogger
        RLlib metrics logger of the env runner.
    **kwargs
        Other callback arguments, ignored.
    """
    env: BaseEnv = env_runner.env.envs[env_index].unwrapped

    metrics = env.logger.peek()
    env.reporter.report(metrics)
    reduced = env.logger.reduce()

    episode_id = episode.id_.partition("|raw=")[0]

    # TODO could we just have the EnvRolloutSchema here ?
    metrics_logger.log_value(
        key=("by_episode", episode_id), value=reduced, reduce="item"
    )


def _evaluate_with_fixed_duration_once(algo, eval_env_runner_group):
    """Evaluate for exactly ``evaluation_duration`` units in a single round.

    Adapted from RLlib's ``Algorithm._evaluate_with_fixed_duration``. Used as
    ``custom_evaluation_function`` so that, with one evaluation env runner per
    ``(train seed, eval seed)`` and one sub-environment per mechanism, every
    ``(mechanism, policy seed, eval seed)`` triple is rolled out exactly once.

    Differences from the RLlib original, all on the new API stack path:

    - The remaining units are split across healthy runners once and the loop
      exits after that round instead of iterating until the duration is met.
    - If fewer results than healthy runners come back within
      ``evaluation_sample_timeout_s``, a ``RuntimeError`` is raised rather
      than re-issuing the work to other runners. Redistribution would
      evaluate some triples twice and others never, biasing the fitness the
      regulator computes from the published step contexts.
    - If the units completed differ from ``evaluation_duration``, a
      ``RuntimeError`` is raised instead of another round being started.
    - Evaluation metrics are read back with ``latest_merged_only=True`` so
      only this call's results are returned.
    - Three values are returned (the public ``custom_evaluation_function``
      contract) instead of the five the private RLlib method returns.

    The old API stack branch of the RLlib original is dropped: this project
    runs on the new API stack only.

    Parameters
    ----------
    algo : Algorithm
        The algorithm being evaluated; supplies the evaluation config.
    eval_env_runner_group : EnvRunnerGroup
        Group of evaluation env runners to sample from.

    Returns
    -------
    tuple[dict, int, int]
        ``(eval_results, env_steps, agent_steps)`` where ``eval_results`` is
        the ``EVALUATION_RESULTS`` subtree of the metrics logger.

    Raises
    ------
    RuntimeError
        On a missing runner result or an incomplete round (see above).
    """
    # How many episodes/timesteps do we need to run?
    unit = algo.config.evaluation_duration_unit
    num_workers = algo.config.evaluation_num_env_runners
    force_reset = algo.config.evaluation_force_reset_envs_before_iteration
    time_out = algo.config.evaluation_sample_timeout_s

    # Remote function used on healthy EnvRunners to sample, get metrics, and
    # step counts.
    def _env_runner_remote(worker, num, round, iter, _force_reset):
        """Sample ``num[worker_index]`` units on one runner and return metrics.

        Returns ``(env_steps, agent_steps, metrics, iter)``; ``iter`` echoes
        the algorithm iteration so stale results from slow workers can be
        discarded. Environments are force-reset only on round 0.
        """
        # Sample AND get_metrics, but only return metrics (and steps actually taken)
        # to save time. Also return the iteration to check, whether we should
        # discard and outdated result (from a slow worker).
        episodes = worker.sample(
            num_timesteps=(num[worker.worker_index] if unit == "timesteps" else None),
            num_episodes=(num[worker.worker_index] if unit == "episodes" else None),
            force_reset=_force_reset and round == 0,
        )
        metrics = worker.get_metrics()
        env_steps = sum(e.env_steps() for e in episodes)
        agent_steps = sum(e.agent_steps() for e in episodes)
        return env_steps, agent_steps, metrics, iter

    all_metrics = []

    # How many episodes have we run (across all eval workers)?
    num_units_done = 0
    num_healthy_workers = eval_env_runner_group.num_healthy_remote_workers()

    env_steps = agent_steps = 0

    _round = -1
    algo_iteration = algo.iteration

    # In case all the remote evaluation workers die during a round of
    # evaluation, we need to stop.
    while num_healthy_workers > 0:
        units_left_to_do = algo.config.evaluation_duration - num_units_done
        if units_left_to_do <= 0:
            break

        _round += 1

        # New API stack: EnvRunners return Episodes.
        _num = [None] + [  # [None]: skip idx=0 (local worker)
            (units_left_to_do // num_healthy_workers)
            + bool(i <= (units_left_to_do % num_healthy_workers))
            for i in range(1, num_workers + 1)
        ]

        results = eval_env_runner_group.foreach_env_runner(
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
                else (env_s if algo.config.count_steps_by == "env_steps" else ag_s)
            )
        if num_units_done != algo.config.evaluation_duration:
            raise RuntimeError(
                "The single evaluation round returned fewer units than requested: "
                f"requested={algo.config.evaluation_duration}, "
                f"completed={num_units_done}, "
                f"unit={unit}"
            )
        break

    if num_healthy_workers == 0:
        logger.warning(
            "Calling `sample()` on your remote evaluation worker(s) "
            "resulted in all workers crashing! Make sure a) your environment is not"
            " too unstable, b) you have enough evaluation workers "
            "(`config.evaluation(evaluation_num_env_runners=...)`) to cover for "
            "occasional losses, and c) you use the `config.fault_tolerance("
            "restart_failed_env_runners=True)` setting."
        )

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
