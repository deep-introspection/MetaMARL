from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

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

    # set env id
    raw_episode_id = episode.id_
    

    # Store structured metadata for policy mapping / logging.
    if not raw_episode_id.startswith("env="):
        episode.id_ = f"env={env_index}|m={mechanism_id}|ps={seed}|raw={raw_episode_id}"

    # TODO inject policy_id to env for traceability and debugging
