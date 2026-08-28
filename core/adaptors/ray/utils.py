"""Helpers for reading RLlib result dictionaries and fingerprinting weights.

The metric getters accept both the new API stack layout (``env_runners/...``)
and the classic one (top-level ``episode_reward_mean``, ``timesteps_total``,
``info/learner/...``), returning a neutral default when a key is absent.
"""

import hashlib

import numpy as np
import torch
from core.utils import to_float


def _get_env(result: dict) -> dict:
    """Return the ``env_runners`` sub-dict of a result, or ``{}``."""
    return result.get("env_runners", {}) or {}


def get_episode_return_mean(result: dict) -> float:
    """Extract the mean episode return from an RLlib result.

    Looks at ``env_runners/episode_return_mean`` first (new API stack), then
    the legacy ``episode_reward_mean`` keys.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    float
        The mean return, or ``0.0`` if no key is present.
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
    """Extract environment step counters from an RLlib result.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    tuple[int, int]
        ``(steps_this_iteration, steps_lifetime)`` read from
        ``env_runners/num_env_steps_sampled[_lifetime]`` with the legacy
        ``timesteps_this_iter`` / ``timesteps_total`` as fallback; ``0`` when
        missing.
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
    """Average ``policy_loss`` across policies, if the result exposes it.

    Reads the classic layout ``info/learner/<policy>/learner_stats/
    policy_loss``. The new API stack reports losses under
    ``learners/<module>/policy_loss`` instead, so on that stack this returns
    NaN and the training log prints ``policy_loss=NA``.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    float
        Mean of the per-policy losses, or ``nan`` when none is found.
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


def hash_weights(weights) -> str:
    """Compute a SHA-256 fingerprint of a (nested) weights structure.

    Used to check that ``PolicyActor.reset`` restores identical parameters
    across outer iterations and across runs.

    Parameters
    ----------
    weights : dict or torch.Tensor or numpy.ndarray or Any
        Typically the nested dict returned by ``Algorithm.get_weights()``.
        Dict keys are visited in sorted order so the hash is independent of
        insertion order; each leaf is hashed together with its ``/``-joined
        key path.

    Returns
    -------
    str
        Hex digest. Tensors are moved to CPU and hashed by raw bytes, arrays
        likewise, and any other leaf by its ``repr``.
    """
    h = hashlib.sha256()

    def update(obj, prefix=""):
        """Recursively feed ``obj`` into the running hash under ``prefix``."""
        if isinstance(obj, dict):
            for key in sorted(obj):
                update(obj[key], f"{prefix}/{key}")

        elif isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous().numpy()
            h.update(prefix.encode())
            h.update(array.tobytes())

        elif isinstance(obj, np.ndarray):
            h.update(prefix.encode())
            h.update(np.ascontiguousarray(obj).tobytes())

        else:
            h.update(prefix.encode())
            h.update(repr(obj).encode())

    update(weights)
    return h.hexdigest()
