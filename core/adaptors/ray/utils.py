import hashlib

import numpy as np
import torch
from core.utils import to_float


def _get_env(result: dict) -> dict:
    return result.get("env_runners", {}) or {}


def get_episode_return_mean(result: dict) -> float:
    env = _get_env(result)
    v = to_float(env.get("episode_return_mean"))
    if v is not None:
        return v
    v = to_float(result.get("episode_reward_mean")) or to_float(
        env.get("episode_reward_mean")
    )
    return v if v is not None else 0.0


def get_env_steps(result: dict) -> tuple[int, int]:
    env = _get_env(result)
    steps_iter = to_float(env.get("num_env_steps_sampled")) or to_float(
        result.get("timesteps_this_iter")
    )
    steps_life = to_float(env.get("num_env_steps_sampled_lifetime")) or to_float(
        result.get("timesteps_total")
    )
    return int(steps_iter or 0), int(steps_life or 0)


def get_policy_loss_if_present(result: dict) -> float:
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
    h = hashlib.sha256()

    def update(obj, prefix=""):
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
