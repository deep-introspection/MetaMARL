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
    policy_to_agents = {}

    for agent_type, spec in config.agent_specs.items():
        pid = spec["policy"]
        for i in range(spec["count"]):
            aid = f"{agent_type}:{i}"
            agent_to_policy[aid] = pid
            policy_to_agents.setdefault(pid, []).append(aid)

    agents = list(agent_to_policy.keys())

    env = config._env_creator(
        world=world,
        opt_id=opt_id,
        agents=agents,
        **config.env_config,
    )

    # Run evaluation episodes
    for ep in range(eval_episodes):
        seed = None if eval_base_seed is None else eval_base_seed + ep
        observations, _ = env.reset(seed=seed)

        terminated = {a: False for a in agents}
        truncated = {a: False for a in agents}

        while not any(terminated.values()) and not any(truncated.values()):
            actions = {}

            # -------------------------
            # Batched per-policy inference
            # -------------------------
            for policy_id, agent_ids in policy_to_agents.items():
                obs_batch = np.stack([observations[a] for a in agent_ids])
                act_batch = ray.get(
                    policy_actor.compute_actions.remote(policy_id, obs_batch)
                )

                for aid, act in zip(agent_ids, act_batch):
                    space = config.env_config["action_spaces"][aid]
                    act = np.asarray(act, dtype=space.dtype)
                    act = np.clip(act, space.low, space.high)
                    actions[aid] = act

            observations, _, terminated, truncated, _ = env.step(actions)

        env.close()


@ray.remote(num_cpus=1)
def _evaluation_runner_remote(**kwargs):
    _run_evaluation_runner(**kwargs)
