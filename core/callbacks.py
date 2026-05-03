from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
# from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv


def tag_episode_with_env_idx(*, episode: MultiAgentEpisode, env_index: int, **kwargs):
    """Prefix the episode ID with the vectorised-environment index.

    Ray RLlib's new-stack ``on_episode_created`` callback receives one
    ``MultiAgentEpisode`` per worker.  When multiple environments are
    vectorised inside a single worker (``num_envs_per_env_runner > 1``),
    different environments share the same episode-ID space.  This callback
    disambiguates episodes by prepending the integer vector index so that
    ``"<env_index>|<original_id>"`` is unique across the worker.

    Parameters
    ----------
    episode : MultiAgentEpisode
        The episode object created by RLlib whose ``id_`` attribute will be
        mutated in-place.
    env_index : int
        Zero-based index of the vectorised environment that owns this episode.
    **kwargs
        Additional keyword arguments forwarded by RLlib's callback system;
        ignored here.
    """
    episode_id = episode.id_
    episode.id_ = f"{env_index}|{episode_id}"
