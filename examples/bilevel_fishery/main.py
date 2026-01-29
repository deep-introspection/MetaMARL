# TODO Rename PPOptimizerConfig to PPOConfig
from core.optimizers.es.config import ESConfig

# TODO Move both th config and the optimizer into same file
from core.optimizers.ppo.config import PPOptimizerConfig
from core.world.base import World
from examples.bilevel_fishery.mechanism import FisheryMechnaismSpace
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

WORLD_NAME = "fishery_world"
world = World.options(name=WORLD_NAME).remote()

# Initialize inner optimizer
# TODO move world cfg into own method
# TODO documentation for each of these where ? -> link to RAY docs
# TODO easy way to add the observation_space and action_space so that it connects with envs

# TODO these params must be configurable :
# Default ecological parameters for Lotka-Volterra dynamics
# DEFAULT_ECOLOGY_CONFIG = {
#     "alpha": 1.0,  # Algae growth rate
#     "beta": 0.5,  # Algae consumption by fish
#     "delta": 0.5,  # Fish growth from algae
#     "gamma": 1.0,  # Fish natural death rate
#     "dt": 0.05,  # Time step size
#     "horizon": 200,  # Episode length
#     "algae_init": 1.0,  # Initial algae population
#     "fish_init": 0.5,  # Initial fish population
# }

# Default mechanism parameters
# DEFAULT_MECHANISM_CONFIG = {
#     "fixed_quota": 0.2,  # Fixed harvest quota
#     "prop_quota": 0.1,  # Proportional quota factor
#     "min_stock": 0.1,  # Minimum stock threshold
#     "fine_amount": 1.0,  # Fine per unit over-harvest
#     "ban_period": 2,  # Periods banned after violation
# }

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    "num_fishermen": 3,  # Number of fishermen agents - num of agents in multiagent env
    "horizon": 200,  # Episode length in timesteps - termination condition in env
    "outer_iters": 10,  # Number of outer optimization iterations
    "workers": 2,  # Number of parallel workers --> env_runners
    "sustain_weight": 5.0,  # Sustainability weight
    "sus_threshold": 0.1,  # Sustainability threshold
}
# Observation/action space bounds
# OBSERVATION_BOUNDS = {
#     "low": 0.0,
#     "high": np.finfo(np.float32).max,
#     "shape": (2,),  # [algae_population, fish_population]
#     "dtype": np.float32,
# }

# ACTION_BOUNDS = {
#     "low": 0.0,  # Minimum harvest fraction
#     "high": 1.0,  # Maximum harvest fraction
#     "shape": (1,),
#     "dtype": np.float32,
# }

ppo_cfg: PPOptimizerConfig = (
    PPOptimizerConfig()
    .environment(env=FisheryRegulatedEnv, env_config={"world_name": WORLD_NAME})
    .framework(
        framework="torch",
    )
    .env_runners(
        num_env_runners=0,  # num_workers
    )
    .training(gramma=0.99, train_batch_size=4000, minibatch_size=512, lr=3e-4)
    .multi_agent(
        policies={"fisher_policy": (None, observation_space, action_space, {})},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "fisher_policy",
        policies_to_train=["fisher_policy"],
    )
    .api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
    .fault_tolerance(restart_failed_env_runners=False)
)

es_cfg: ESConfig = (
    ESConfig()
    .training(
        dimension=5,  # TODO dimension inferred from mechanism ?
        pop_size=16,
        sigma=0.15,
        mean_lr=0.1,
        sigma_lr=0.05,
        min_sigma=1e-3,
        max_sigma=0.5,
    )
    .environment(
        env=FisheryRegulatorEnv,
        env_config={
            "mechanism_space": FisheryMechnaismSpace(),
        },
        train_iters=100,
        eval_iters=5,
    )
)


ppo = ppo_cfg.build_optimizer(world=world, world_name=WORLD_NAME)
es = es_cfg.build_optimizer(world=world, inner_opt=ppo)
