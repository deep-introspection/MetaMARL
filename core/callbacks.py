from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
# from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv


def tag_episode_with_env_idx(*, episode: MultiAgentEpisode, env_index: int, **kwargs):
    episode_id = episode.id_
    episode.id_ = f"{env_index}|{episode_id}"
