import numpy as np
import ray
from gymnasium import spaces

from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig

from core.reporting.query import Query
from core.reporting.wandb import WandbConfig
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanismSpace
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.regulated_env_shaefer import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

ray.shutdown()

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .reporter(
            config=WandbConfig(
            project = "bilevel",
            x_disable_stats = True,
            x_disable_meta = True,
            quiet = True,
            max_end_of_run_summary_metrics = 0,
            max_end_of_run_history_metrics = 0,
        )
    )
    .mechanism(
        # TODO adding defaults
        space=FisheryMechanismSpace(
            optimize_params=["fixed_quota", "restoration_subsidy"], #", "risk_penalty_scale", "risk_penalty_power", "fine_amount", , "max_demand_frac", 
            # default_fixed_quota=0.7812058329582214,
            # default_min_demand_frac=0.1059612140059471,
            # default_max_demand_frac=0.5705976366996766,
            # default_fine_amount=0.05723086595535279,
            # default_risk_penalty_scale=0.5466187596321106,
            # default_risk_penalty_power=3.8254001140594482,
            # default_under_irrigation_penalty_scale=0.5174465179443359,

            # permissive quota  
            # default_fixed_quota=0.60,
            # default_min_demand_frac=1.0,
            # default_max_demand_frac=1.0,
            # default_fine_amount=0.0,
            # default_risk_penalty_scale=0.0,
            # default_risk_penalty_power=1.0,
            # default_under_irrigation_penalty_scale=0.0,

            # extremely restrictive quota
            default_fixed_quota=0.56224, #0.90 #0.52
            default_max_demand_frac=1.0,
            default_restoration_subsidy=0.10,

            default_fine_amount=0.20, 
            default_risk_penalty_scale=0.0, #1.0
            default_risk_penalty_power=1.0,

            # moderatley restrictive quota 
            # default_fixed_quota=0.80,
            # default_min_demand_frac=0.25,
            # default_max_demand_frac=0.60,

            # default_fine_amount=0.05,
            # default_risk_penalty_scale=0.40,
            # default_risk_penalty_power=2.0,
            # default_under_irrigation_penalty_scale=0.0,

            # # maybe set default near middle
            # # 
            # default_max_farm_area_m2=500_000.0,
        )
    )
    .training(outer_iters=1000)
    .ray(
        device="cpu",
        num_cpus=4,
        omp_threads=1,
        logging_level="ERROR",
        runtime_env={
            "excludes": [
                "wandb/",
                ".git/",
                ".venv/",
                "__pycache__/",
                "ray_results/",
                "runs/",
            ]
        },
    )
    .outer(
        ESConfig()
        .training(
            sigma=0.15,
            mean_lr=0.10,
            sigma_decay=1.0,
            sigma_lr=0.00,
            min_sigma=0.15,
            max_sigma=0.15,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "ecology_cfg": {
                    "sustainability_weight": 2, # assert between 0 and 5
                    "sustainability_threshold": 0.20,
                    "K": 5_000, # HAS to match environmnet K
                },
            },
            horizon=100,
            train_iters=50,
        )
        .debugging(
            seed=42,
            num_seeds=1,
        )
    #     .reporting(
    #         schema=ESchema,
    #         queries=[
    #             Query(
    #                 title="",
    #                 x=("generation"),
    #                 y=("fish_norm_mean")
    #             ),

    #         ],
    #     )
    )
    .inner(
        APPOptimizerConfig()
        .resources(
            num_cpus_for_main_process=1,
        )
        .framework(
            framework="torch",
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=FisheryRegulatedEnv,
            env_config={
                "ecology_cfg": {
                    # Pella-Tomlinson / Schaefer single-stock dynamics
                    "r": 0.3,
                    "K": 5_000,
                    "p": 1.0,
                    "B0": 4_000,
                     "fish_init": 4_000,

                    # Env stochasticity
                    "sigma": 0.02,
                    "initial_stock_log_sigma": 0.05,

                    "unregulated_f_multiplier": 2.0,
                    "collapse_stock_frac": 0.20,
                    "collapse_transition_width": 0.03,
                    "quota_transition_width": 0.05,
                    "harvest_transition_width": 0.005,
                    "violation_transition_width": 0.03,

                    # restorative
                    "restoration_effectiveness": 0.02,
                    "restoration_effort_cost": 0.25,
                    
                },
                "seed": 0,
            },
            horizon=100,
            disable_env_checking=False,
            schema=FisheryMetricSchema,
            queries=[
                Query(
                    title="Fish biomass",
                    x=("iter",),
                    y=("fish_norm",),
                ),
            ],
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=4,
            rollout_fragment_length=100,
            batch_mode="truncate_episodes",
            max_requests_in_flight_per_env_runner=1,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
        )
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,
            on_episode_end=log_and_report_episode_metrics,
        )
        .training(
            vtrace=True,
            circular_buffer_num_batches=4,
            circular_buffer_iterations_per_batch=1,
            broadcast_interval=1,
            timeout_s_sampler_manager=10,
            timeout_s_aggregator_manager=10,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=100,
            minibatch_size=100,
            num_epochs=1,
            entropy_coeff=0.001,
            grad_clip=40.0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=0,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                "rollout_fragment_length": 100,
                "batch_mode": "complete_episodes",
                "max_requests_in_flight_per_env_runner": 1,
            },
            base_seed = 42,
            num_seeds = 3
        )
        .agents(
            {
                "utilizer": {
                    "count":10,
                    "policy": "fisher_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3 + FisheryMechanismSpace().full_dimension,
                        ),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Box(
                        # Action is the fraction of the agent's maximum pull capacity.
                        #
                        # action = 0.0 -> requests 0% of max_pull_fraction
                        # action = 0.5 -> requests 50% of max_pull_fraction
                        # action = 1.0 -> requests 100% of max_pull_fraction
                        #
                        # Actual requested flow:
                        # requested_m3s = action * max_pull_fraction * current_streamflow
                        #
                        # With max_pull_fraction = 0.005:
                        # action high=1.0 -> up to 0.5% of current streamflow
                        # action high=0.5 -> up to 0.25% of current streamflow
                        # action high=0.2 -> up to 0.10% of current streamflow
                        # action high=0.1 -> up to 0.05% of current streamflow
                        #
                        # Rough agent interpretation:
                        # high=0.05 -> individual/small farm-scale user
                        # high=0.1  -> large farm / small irrigation user
                        # high=0.2  -> irrigation district / small utility
                        # high=0.5  -> municipality / industrial user
                        # high=1.0  -> large municipality / regional user
                        #
                        # NOTE:
                        # This does not change max_pull_fraction itself. It changes how much
                        # of that maximum capacity the policy is allowed to request.
                        low=-np.inf,
                        high=np.inf,
                        shape=(2,),
                        dtype=np.float32,
                    ),
                }
            }
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
        )
        .debugging(
            seed=42,
            num_seeds=1,
        )
        # .reporting(
        #     min_time_s_per_iteration=0,
        #     min_sample_timesteps_per_iteration=0,
        #     min_train_timesteps_per_iteration=0,
        #     schema=RaySchema,
        #     queries=[
        #         Query(
        #             title="",
        #             x=("train_iter"),
        #             y=("fish_norm_mean")
        #         ),

        #     ],
        # )
    )
)

bilevel_opt = bilevel_opt_cfg.build_optimizer()
bilevel_opt.run()

ray.shutdown()