from typing import Optional
import numpy as np
import ray

from core.world.base import World


def _run_evaluation_runner(
    *,
    policy_actor,          # Ray ActorHandle
    config,
    world: World,          # Ray ActorHandle
    opt_id: str,
    eval_episodes: int,
    eval_base_seed: Optional[int],
):
    # Build agent → policy mapping
    agent_to_policy = {}
    for agent_type, spec in config.agent_specs.items():
        policy_id = spec["policy"]
        for i in range(spec["count"]):
            agent_to_policy[f"{agent_type}:{i}"] = policy_id

    agents = list(agent_to_policy.keys())

    # Run evaluation episodes
    for ep in range(eval_episodes):
        seed = None if eval_base_seed is None else eval_base_seed + ep

        env = config._env_creator(
            world=world,
            opt_id=opt_id,
            agents=agents,
            **config.env_config,
        )

        observations, _ = env.reset(seed=seed)
        terminated = {aid: False for aid in agents}
        truncated = {aid: False for aid in agents}
        step_count = 0

        while (
            not any(terminated.values())
            and not any(truncated.values())
            and step_count < env.horizon
        ):
            actions = {}

            for agent_id in agents:
                obs = observations[agent_id]
                policy_id = agent_to_policy[agent_id]

                # 🔑 Policy access via Ray actor (SAFE)
                action = ray.get(
                    policy_actor.compute_action.remote(policy_id, obs)
                )

                act_space = config.env_config["action_spaces"][agent_id]
                action = np.asarray(action, dtype=act_space.dtype)
                action = np.clip(action, act_space.low, act_space.high)

                actions[agent_id] = action

            observations, rewards, terminated, truncated, infos = env.step(actions)
            step_count += 1

        env.close()


@ray.remote(num_cpus=1)
def _evaluation_runner_remote(**kwargs):
    _run_evaluation_runner(**kwargs)
