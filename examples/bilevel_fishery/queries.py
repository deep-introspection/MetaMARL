"""Reporting queries for the fishery experiment.

Three metric schemas are logged during a run and each has its own query
bundle:

- the regulated environment (``FisheryMetricSchema``): one series per
  environment step (``FISHERY_ENV_QUERIES``, ``fishery_agent_queries``);
- the inner RLlib optimizer (``RaySchema``): one value per training iteration
  (``RAY_ROLLOUT_QUERIES``, ``RAY_PERFORMANCE_QUERIES``, ``ray_policy_queries``);
- the outer ES optimizer (``ESSchema``): one value per generation
  (``ES_QUERIES``, ``es_parameter_fitness_queries``).

Runtime ids (agent, policy, candidate) can be named explicitly or matched with
the wildcard ``"*"`` (one series per id, sorted; with ``reduce="mean"`` the
first wildcard groups and the others are averaged).
"""

from core.reporting.query import Query

# --- environment: one series per env step ------------------------------------

FISHERY_ENV_QUERIES = tuple(
    Query(title=title, x=("iter",), y=(field,))
    for title, field in (
        ("Fish biomass", "fish_norm"),
        ("Next fish biomass", "fish_norm_next"),
        ("Fish stock", "fish_stock"),
        ("Next fish stock", "fish_stock_next"),
        ("Biological growth", "growth"),
        ("Growth noise", "growth_noise"),
        ("Attempted harvest", "H_attempted"),
        ("Realized harvest", "H_realized"),
        ("Allowed harvest", "allowed_harvest"),
        ("Total usage normalized", "total_usage_norm"),
        ("Quota stress", "quota_stress"),
        ("Mean reward", "reward_mean"),
    )
)


def fishery_agent_queries(agent_id: str) -> tuple[Query, ...]:
    """Per-agent environment series for ``agent_id``."""
    base = ("by_agent", agent_id)
    return tuple(
        Query(title=f"{title} — {agent_id}", x=("iter",), y=base + (field,))
        for title, field in (
            ("Reward", "reward"),
            ("Requested harvest", "requested_harvest"),
            ("Delivered harvest", "delivered_harvest"),
            ("Requested harvest fraction", "requested_frac"),
            ("Quota violation", "quota_violation"),
            ("Quota penalty", "quota_penalty"),
            ("Risk penalty", "risk_penalty"),
        )
    )


FISHERY_ALL_AGENTS_QUERIES = (
    Query(title="Reward — all agents", x=("iter",), y=("by_agent", "*", "reward")),
    Query(
        title="Delivered harvest — all agents",
        x=("iter",),
        y=("by_agent", "*", "delivered_harvest"),
    ),
    Query(
        title="Mean reward across agents ±1 std",
        x=("iter",),
        y=("by_agent", "*", "reward"),
        reduce="mean",
        error="std",
    ),
)


# --- inner optimizer: one value per RLlib training iteration --------------------

RAY_ROLLOUT_QUERIES = (
    Query(
        title="Train reward",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "reward_mean"),
            ("train", "rollout", "aggregate", "reward_min"),
            ("train", "rollout", "aggregate", "reward_max"),
        ),
    ),
    Query(
        title="Train episode length",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "episode_len_mean"),
            ("train", "rollout", "aggregate", "episode_len_min"),
            ("train", "rollout", "aggregate", "episode_len_max"),
        ),
    ),
    Query(
        title="Train episodes",
        x=("iter",),
        y=("train", "rollout", "aggregate", "num_episodes"),
    ),
)

RAY_PERFORMANCE_QUERIES = (
    Query(
        title="Train environment steps",
        x=("iter",),
        y=(
            ("train", "performance", "env_steps_this_iter"),
            ("train", "performance", "env_steps_lifetime"),
        ),
    ),
    Query(
        title="Training timing",
        x=("iter",),
        y=(
            ("train", "performance", "training_iteration_s"),
            ("train", "performance", "sample_s"),
            ("train", "performance", "learner_update_s"),
        ),
    ),
)


RAY_ALL_POLICIES_QUERIES = (
    Query(
        title="Total loss — all policies",
        x=("iter",),
        y=("train", "learner", "by_policy", "*", "total_loss"),
    ),
    Query(
        title="Policy entropy — all policies",
        x=("iter",),
        y=("train", "learner", "by_policy", "*", "policy_entropy"),
    ),
)


def ray_policy_queries(policy_id: str) -> tuple[Query, ...]:
    """Learner series for one RLModule ``policy_id``."""
    base = ("train", "learner", "by_policy", policy_id)
    return tuple(
        Query(title=f"{title} — {policy_id}", x=("iter",), y=base + (field,))
        for title, field in (
            ("Total loss", "total_loss"),
            ("Policy loss", "policy_loss"),
            ("Policy entropy", "policy_entropy"),
            ("Policy KL", "policy_kl"),
            ("Value loss", "value_loss"),
            ("Gradient norm", "gradient_norm"),
        )
    )


RAY_QUERIES = RAY_ROLLOUT_QUERIES + RAY_PERFORMANCE_QUERIES + RAY_ALL_POLICIES_QUERIES

# --- outer optimizer: one value per ES generation --------------------------------

ES_QUERIES = (
    Query(
        title="Fitness over generations",
        x=("generation",),
        y=(("fitness_mean",), ("fitness_best",), ("best_fitness_global",)),
    ),
    Query(
        title="Candidate fitness (all candidates)",
        x=("generation",),
        y=("by_mechanism", "*", "fitness"),
    ),
    Query(
        title="Mean candidate fitness ±1 std",
        x=("generation",),
        y=("by_mechanism", "*", "fitness"),
        reduce="mean",
        error="std",
    ),
    Query(title="ES sigma", x=("generation",), y=("sigma",)),
    Query(title="ES population size", x=("generation",), y=("population_size",)),
)


def es_parameter_queries(parameter_names: tuple[str, ...]) -> tuple[Query, ...]:
    """Search mean, global best and generation best for each optimized parameter."""
    return (
        Query(
            title="ES search mean",
            x=("generation",),
            y=tuple(("search_mean", name, "value") for name in parameter_names),
        ),
        Query(
            title="Global-best mechanism parameters",
            x=("generation",),
            y=tuple(("global_best", name, "value") for name in parameter_names),
        ),
        Query(
            title="Generation-best mechanism parameters",
            x=("generation",),
            y=tuple(("generation_best", name, "value") for name in parameter_names),
        ),
    )


def es_candidate_fitness_queries(num_candidates: int) -> tuple[Query, ...]:
    """Per-candidate fitness series (candidate ids are ``"0"``..``"n-1"``)."""
    return (
        Query(
            title="Candidate fitness over generations",
            x=("generation",),
            y=tuple(("by_mechanism", str(i), "fitness") for i in range(num_candidates)),
        ),
    )


def es_parameter_fitness_queries(parameter_names: tuple[str, ...]) -> tuple[Query, ...]:
    """Fitness against parameter value, all candidates in one query per parameter.

    The wildcard binds the same candidate on the x and y sides, so each series
    is one candidate's (parameter value, fitness) trajectory over generations.
    """
    return tuple(
        Query(
            title=f"Fitness vs {name}",
            x=("by_mechanism", "*", "by_parameter", name, "value"),
            y=("by_mechanism", "*", "fitness"),
        )
        for name in parameter_names
    )
