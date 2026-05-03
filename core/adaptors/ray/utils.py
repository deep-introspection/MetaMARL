import numpy as np
from core.utils import to_float


def _get_env(result: dict) -> dict:
    """Extract the ``env_runners`` sub-dict from an RLlib ``ResultDict``.

    Parameters
    ----------
    result : dict
        RLlib ``ResultDict`` returned by ``Algorithm.train()`` or
        ``Algorithm.evaluate()``.

    Returns
    -------
    dict
        The ``"env_runners"`` nested dictionary, or an empty dict if the
        key is absent or the value is falsy (handles both old and new
        API stack result layouts).
    """
    return result.get("env_runners", {}) or {}


def get_episode_return_mean(result: dict) -> float:
    """Extract the mean episode return from an RLlib result dictionary.

    Checks multiple key locations to remain compatible with both the new
    API stack (``env_runners.episode_return_mean``) and the classic API
    stack (``episode_reward_mean`` at top level or inside ``env_runners``).

    Parameters
    ----------
    result : dict
        RLlib ``ResultDict`` returned by ``Algorithm.train()``.

    Returns
    -------
    float
        Mean episode return for the most recent training iteration, or
        ``0.0`` if no metric is found.
    """
    env = _get_env(result)
    v = to_float(env.get("episode_return_mean"))
    if v is not None:
        return v
    v = to_float(result.get("episode_reward_mean")) or to_float(
        env.get("episode_reward_mean")
    )
    return v if v is not None else 0.0


def get_env_steps(result: dict) -> tuple[int, int]:
    """Extract per-iteration and lifetime environment step counts from a result dict.

    Handles key-name differences between the new API stack
    (``num_env_steps_sampled`` / ``num_env_steps_sampled_lifetime`` inside
    ``env_runners``) and the classic API stack (``timesteps_this_iter`` /
    ``timesteps_total`` at the top level).

    Parameters
    ----------
    result : dict
        RLlib ``ResultDict`` returned by ``Algorithm.train()``.

    Returns
    -------
    tuple[int, int]
        ``(steps_this_iter, steps_lifetime)`` — number of environment steps
        collected in the current iteration and total steps across the entire
        training run, respectively.  Both default to 0 if not found.
    """
    env = _get_env(result)
    steps_iter = to_float(env.get("num_env_steps_sampled")) or to_float(
        result.get("timesteps_this_iter")
    )
    steps_life = to_float(env.get("num_env_steps_sampled_lifetime")) or to_float(
        result.get("timesteps_total")
    )
    return int(steps_iter or 0), int(steps_life or 0)


def get_policy_loss_if_present(result: dict) -> float:
    """Extract the mean policy loss across all policies from a result dict.

    Navigates the ``info.learner.<policy_id>.learner_stats.policy_loss``
    path that RLlib populates in its classic API stack result.  Returns
    ``nan`` when the learner info block is absent (e.g. new API stack or
    evaluation result).

    Parameters
    ----------
    result : dict
        RLlib ``ResultDict`` returned by ``Algorithm.train()``.

    Returns
    -------
    float
        Mean ``policy_loss`` averaged over all policies for which the
        metric is present, or ``float("nan")`` if none are found.
    """
    learner_info = (result.get("info") or {}).get("learner") or {}
    losses = []
    if isinstance(learner_info, dict):
        for _, policy_stats in learner_info.items():
            ls = (policy_stats or {}).get("learner_stats") or {}
            v = to_float(ls.get("policy_loss"))
            if v is not None:
                losses.append(v)
    return float(np.mean(losses)) if losses else float("nan")
