import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Self

import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import register_env

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.utils import generate_uuid
from core.world.base import World

# TODO override environment to attach docstrings


@dataclass
class AgentSpec:
    count: int
    policy: str
    observation_space: Space
    action_space: Space


class RayOptimizerConfig(OptimizerConfig):
    # TODO review this
    # must be overriden in subclasses
    algo_class: type[Algorithm] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")
        super().__init__(opt_class=RayOptimizer)

        # TODO termporary setting until find out how to share world context accross runners
        self._cfg_ops: list[Callable[[AlgorithmConfig], AlgorithmConfig]] = []
        self.rllib_cfg: AlgorithmConfig | None = None
        self.agent_specs: Optional[dict] = None  # TODO default
        self.world_name: Optional[str] = None

    def rllib_config_mutator(fn):
        def wrapper(self, *args, **kwargs):
            self._cfg_ops.append(lambda cfg: fn(cfg, *args, **kwargs))
            return self

        return wrapper

    @rllib_config_mutator
    def validate(cfg, **kwargs) -> None:
        return cfg.validate(**kwargs)

    @rllib_config_mutator
    def get_config_for_module(cfg, **kwargs) -> None:
        return cfg.get_config_for_module(**kwargs)

    @rllib_config_mutator
    def python_environment(cfg, **kwargs) -> None:
        """Sets the config's python environment settings.

        Args:
            extra_python_environs_for_driver: Any extra python env vars to set in the
                algorithm's process, e.g., {"OMP_NUM_THREADS": "16"}.
            extra_python_environs_for_worker: The extra python environments need to set
                for worker processes.
        """
        return cfg.python_environment(**kwargs)

    @rllib_config_mutator
    def resources(cfg, **kwargs) -> None:
        """Specifies resources allocated for an Algorithm and its ray actors/workers.

        Args:
            num_cpus_for_main_process: Number of CPUs to allocate for the main algorithm
                process that runs `Algorithm.training_step()`.
                Note: This is only relevant when running RLlib through Tune. Otherwise,
                `Algorithm.training_step()` runs in the main program (driver).
            placement_strategy: The strategy for the placement group factory returned by
                `Algorithm.default_resource_request()`. A PlacementGroup defines, which
                devices (resources) should always be co-located on the same node.
                For example, an Algorithm with 2 EnvRunners and 1 Learner (with
                1 GPU) requests a placement group with the bundles:
                [{"cpu": 1}, {"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], where the
                first bundle is for the local (main Algorithm) process, the second one
                for the 1 Learner worker and the last 2 bundles are for the two
                EnvRunners. These bundles can now be "placed" on the same or different
                nodes depending on the value of `placement_strategy`:
                "PACK": Packs bundles into as few nodes as possible.
                "SPREAD": Places bundles across distinct nodes as even as possible.
                "STRICT_PACK": Packs bundles into one node. The group is not allowed
                to span multiple nodes.
                "STRICT_SPREAD": Packs bundles across distinct nodes.
        """
        return cfg.resources(**kwargs)

    @rllib_config_mutator
    def framework(cfg, **kwargs) -> None:
        """Sets the config's DL framework settings.

        Args:
            framework: torch: PyTorch; tf2: TensorFlow 2.x (eager execution or traced
                if eager_tracing=True); tf: TensorFlow (static-graph);
            eager_tracing: Enable tracing in eager mode. This greatly improves
                performance (speedup ~2x), but makes it slightly harder to debug
                since Python code won't be evaluated after the initial eager pass.
                Only possible if framework=tf2.
            eager_max_retraces: Maximum number of tf.function re-traces before a
                runtime error is raised. This is to prevent unnoticed retraces of
                methods inside the `..._eager_traced` Policy, which could slow down
                execution by a factor of 4, without the user noticing what the root
                cause for this slowdown could be.
                Only necessary for framework=tf2.
                Set to None to ignore the re-trace count and never throw an error.
            tf_session_args: Configures TF for single-process operation by default.
            local_tf_session_args: Override the following tf session args on the local
                worker
            torch_compile_learner: If True, forward_train methods on TorchRLModules
                on the learner are compiled. If not specified, the default is to compile
                forward train on the learner.
            torch_compile_learner_what_to_compile: A TorchCompileWhatToCompile
                mode specifying what to compile on the learner side if
                torch_compile_learner is True. See TorchCompileWhatToCompile for
                details and advice on its usage.
            torch_compile_learner_dynamo_backend: The torch dynamo backend to use on
                the learner.
            torch_compile_learner_dynamo_mode: The torch dynamo mode to use on the
                learner.
            torch_compile_worker: If True, forward exploration and inference methods on
                TorchRLModules on the workers are compiled. If not specified,
                the default is to not compile forward methods on the workers because
                retracing can be expensive.
            torch_compile_worker_dynamo_backend: The torch dynamo backend to use on
                the workers.
            torch_compile_worker_dynamo_mode: The torch dynamo mode to use on the
                workers.
            torch_ddp_kwargs: The kwargs to pass into
                `torch.nn.parallel.DistributedDataParallel` when using `num_learners
                > 1`. This is specifically helpful when searching for unused parameters
                that are not used in the backward pass. This can give hints for errors
                in custom models where some parameters do not get touched in the
                backward pass although they should.
            torch_skip_nan_gradients: If updates with `nan` gradients should be entirely
                skipped. This skips updates in the optimizer entirely if they contain
                any `nan` gradient. This can help to avoid biasing moving-average based
                optimizers - like Adam. This can help in training phases where policy
                updates can be highly unstable such as during the early stages of
                training or with highly exploratory policies. In such phases many
                gradients might turn `nan` and setting them to zero could corrupt the
                optimizer's internal state. The default is `False` and turns `nan`
                gradients to zero. If many `nan` gradients are encountered consider (a)
                monitoring gradients by setting `log_gradients` in `AlgorithmConfig` to
                `True`, (b) use proper weight initialization (e.g. Xavier, Kaiming) via
                the `model_config_dict` in `AlgorithmConfig.rl_module` and/or (c)
                gradient clipping via `grad_clip` in `AlgorithmConfig.training`.
        """
        return cfg.framework(**kwargs)

    @rllib_config_mutator
    def api_stack(cfg, **kwargs) -> None:
        """Sets the config's API stack settings.

        Args:
            enable_rl_module_and_learner: Enables the usage of `RLModule` (instead of
                `ModelV2`) and Learner (instead of the training-related parts of
                `Policy`). Must be used with `enable_env_runner_and_connector_v2=True`.
                Together, these two settings activate the "new API stack" of RLlib.
            enable_env_runner_and_connector_v2: Enables the usage of EnvRunners
                (SingleAgentEnvRunner and MultiAgentEnvRunner) and ConnectorV2.
                When setting this to True, `enable_rl_module_and_learner` must be True
                as well. Together, these two settings activate the "new API stack" of
                RLlib.

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.api_stack(**kwargs)

    @rllib_config_mutator
    def env_runners(cfg, **kwargs) -> None:
        """Sets the rollout worker configuration.

        Args:
            env_runner_cls: The EnvRunner class to use for environment rollouts (data
                collection).
            num_env_runners: Number of EnvRunner actors to create for parallel sampling.
                Setting this to 0 forces sampling to be done in the local
                EnvRunner (main process or the Algorithm's actor when using Tune).
            create_local_env_runner: If True, create a local EnvRunner instance, besides
                the `num_env_runners` remote EnvRunner actors. If `num_env_runners` is
                0, this setting is ignored and one local EnvRunner is created
                regardless.
            create_env_on_local_worker: When `num_env_runners` > 0, the driver
                (local_worker; worker-idx=0) does not need an environment. This is
                because it doesn't have to sample (done by remote_workers;
                worker_indices > 0) nor evaluate (done by evaluation workers;
                see below).
            num_envs_per_env_runner: Number of environments to step through
                (vector-wise) per EnvRunner. This enables batching when computing
                actions through RLModule inference, which can improve performance
                for inference-bottlenecked workloads.
            gym_env_vectorize_mode: The gymnasium vectorization mode for vector envs.
                Must be a `gymnasium.VectorizeMode` (enum) value.
                Default is SYNC. Set this to ASYNC to parallelize the individual sub
                environments within the vector. This can speed up your EnvRunners
                significantly when using heavier environments. Set this to
                VECTOR_ENTRY_POINT in case your env creator, also known as
                "gym entry point", already returns a gym.vector.VectorEnv and you
                don't need RLlib to vectorize the environments for the runners.
            num_cpus_per_env_runner: Number of CPUs to allocate per EnvRunner.
            num_gpus_per_env_runner: Number of GPUs to allocate per EnvRunner. This can
                be fractional. This is usually needed only if your env itself requires a
                GPU (i.e., it is a GPU-intensive video game), or model inference is
                unusually expensive.
            custom_resources_per_env_runner: Any custom Ray resources to allocate per
                EnvRunner.
            validate_env_runners_after_construction: Whether to validate that each
                created remote EnvRunner is healthy after its construction process.
            sample_timeout_s: The timeout in seconds for calling `sample()` on remote
                EnvRunner workers. Results (episode list) from workers that take longer
                than this time are discarded. Only used by algorithms that sample
                synchronously in turn with their update step (e.g., PPO or DQN). Not
                relevant for any algos that sample asynchronously, such as APPO or
                IMPALA.
            max_requests_in_flight_per_env_runner: Max number of in-flight requests
                to each EnvRunner (actor)). See the
                `ray.rllib.utils.actor_manager.FaultTolerantActorManager` class for more
                details.
                Tuning these values is important when running experiments with
                large sample batches, where there is the risk that the object store may
                fill up, causing spilling of objects to disk. This can cause any
                asynchronous requests to become very slow, making your experiment run
                slowly as well. You can inspect the object store during your experiment
                through a call to `ray memory` on your head node, and by using the Ray
                dashboard. If you're seeing that the object store is filling up,
                turn down the number of remote requests in flight or enable compression
                or increase the object store memory through, for example:
                `ray.init(object_store_memory=10 * 1024 * 1024 * 1024)  # =10 GB`
            env_to_module_connector: A callable taking an Env as input arg and returning
                an env-to-module ConnectorV2 (might be a pipeline) object.
            module_to_env_connector: A callable taking an Env and an RLModule as input
                args and returning a module-to-env ConnectorV2 (might be a pipeline)
                object.
            add_default_connectors_to_env_to_module_pipeline: If True (default), RLlib's
                EnvRunners automatically add the default env-to-module ConnectorV2
                pieces to the EnvToModulePipeline. These automatically perform adding
                observations and states (in case of stateful Module(s)), agent-to-module
                mapping, batching, and conversion to tensor data. Only if you know
                exactly what you are doing, you should set this setting to False.
                Note that this setting is only relevant if the new API stack is used
                (including the new EnvRunner classes).
            add_default_connectors_to_module_to_env_pipeline: If True (default), RLlib's
                EnvRunners automatically add the default module-to-env ConnectorV2
                pieces to the ModuleToEnvPipeline. These automatically perform removing
                the additional time-rank (if applicable, in case of stateful
                Module(s)), module-to-agent unmapping, un-batching (to lists), and
                conversion from tensor data to numpy. Only if you know exactly what you
                are doing, you should set this setting to False.
                Note that this setting is only relevant if the new API stack is used
                (including the new EnvRunner classes).
            episode_lookback_horizon: The amount of data (in timesteps) to keep from the
                preceeding episode chunk when a new chunk (for the same episode) is
                generated to continue sampling at a later time. The larger this value,
                the more an env-to-module connector can look back in time
                and compile RLModule input data from this information. For example, if
                your custom env-to-module connector (and your custom RLModule) requires
                the previous 10 rewards as inputs, you must set this to at least 10.
            merge_env_runner_states: True, if remote EnvRunner actor states should be
                merged into central connector pipelines. Use "training_only" (default)
                for only doing this for the training EnvRunners, NOT for the evaluation
                EnvRunners.
            broadcast_env_runner_states: True, if merged EnvRunner states (from the
                central connector pipelines) should be broadcast back to all remote
                EnvRunner actors.
            use_worker_filter_stats: Whether to use the workers in the EnvRunnerGroup to
                update the central filters (held by the local worker). If False, stats
                from the workers aren't used and are discarded.
            update_worker_filter_stats: Whether to push filter updates from the central
                filters (held by the local worker) to the remote workers' filters.
                Setting this to True might be useful within the evaluation config in
                order to disable the usage of evaluation trajectories for synching
                the central filter (used for training).
            rollout_fragment_length: Divide episodes into fragments of this many steps
                each during sampling. Trajectories of this size are collected from
                EnvRunners and combined into a larger batch of `train_batch_size`
                for learning.
                For example, given rollout_fragment_length=100 and
                train_batch_size=1000:
                1. RLlib collects 10 fragments of 100 steps each from rollout workers.
                2. These fragments are concatenated and we perform an epoch of SGD.
                When using multiple envs per worker, the fragment size is multiplied by
                `num_envs_per_env_runner`. This is since we are collecting steps from
                multiple envs in parallel. For example, if num_envs_per_env_runner=5,
                then EnvRunners return experiences in chunks of 5*100 = 500 steps.
                The dataflow here can vary per algorithm. For example, PPO further
                divides the train batch into minibatches for multi-epoch SGD.
                Set `rollout_fragment_length` to "auto" to have RLlib compute an exact
                value to match the given batch size.
            batch_mode: How to build individual batches with the EnvRunner(s). Batches
                coming from distributed EnvRunners are usually concat'd to form the
                train batch. Note that "steps" below can mean different things (either
                env- or agent-steps) and depends on the `count_steps_by` setting,
                adjustable via `AlgorithmConfig.multi_agent(count_steps_by=..)`:
                1) "truncate_episodes": Each call to `EnvRunner.sample()` returns a
                batch of at most `rollout_fragment_length * num_envs_per_env_runner` in
                size. The batch is exactly `rollout_fragment_length * num_envs`
                in size if postprocessing does not change batch sizes. Episodes
                may be truncated in order to meet this size requirement.
                This mode guarantees evenly sized batches, but increases
                variance as the future return must now be estimated at truncation
                boundaries.
                2) "complete_episodes": Each call to `EnvRunner.sample()` returns a
                batch of at least `rollout_fragment_length * num_envs_per_env_runner` in
                size. Episodes aren't truncated, but multiple episodes
                may be packed within one batch to meet the (minimum) batch size.
                Note that when `num_envs_per_env_runner > 1`, episode steps are
                buffered until the episode completes, and hence batches may contain
                significant amounts of off-policy data.
            explore: Default exploration behavior, iff `explore=None` is passed into
                compute_action(s). Set to False for no exploration behavior (e.g.,
                for evaluation).
            episodes_to_numpy: Whether to numpy'ize episodes before
                returning them from an EnvRunner. False by default. If True, EnvRunners
                call `to_numpy()` on those episode (chunks) to be returned by
                `EnvRunners.sample()`.
            compress_observations: Whether to LZ4 compress individual observations
                in the SampleBatches collected during rollouts.
        """
        return cfg.env_runners(**kwargs)

    @rllib_config_mutator
    def learners(cfg, **kwargs) -> None:
        """Sets LearnerGroup and Learner worker related configurations.

        Args:
            num_learners: Number of Learner workers used for updating the RLModule.
                A value of 0 means training takes place on a local Learner on main
                process CPUs or 1 GPU (determined by `num_gpus_per_learner`).
                For multi-gpu training, you have to set `num_learners` to > 1 and set
                `num_gpus_per_learner` accordingly (e.g., 4 GPUs total and model fits on
                1 GPU: `num_learners=4; num_gpus_per_learner=1` OR 4 GPUs total and
                model requires 2 GPUs: `num_learners=2; num_gpus_per_learner=2`).
            num_cpus_per_learner: Number of CPUs allocated per Learner worker.
                If "auto" (default), use 1 if `num_gpus_per_learner=0`, otherwise 0.
                Only necessary for custom processing pipeline inside each Learner
                requiring multiple CPU cores.
                If `num_learners=0`, RLlib creates only one local Learner instance and
                the number of CPUs on the main process is
                `max(num_cpus_per_learner, num_cpus_for_main_process)`.
            num_gpus_per_learner: Number of GPUs allocated per Learner worker. If
                `num_learners=0`, any value greater than 0 runs the
                training on a single GPU on the main process, while a value of 0 runs
                the training on main process CPUs.
            num_aggregator_actors_per_learner: The number of aggregator actors per
                Learner (if num_learners=0, one local learner is created). Must be at
                least 1. Aggregator actors perform the task of a) converting episodes
                into a train batch and b) move that train batch to the same GPU that
                the corresponding learner is located on. Good values are 1 or 2, but
                this strongly depends on your setup and `EnvRunner` throughput.
            max_requests_in_flight_per_aggregator_actor: How many in-flight requests
                are allowed per aggregator actor before new requests are dropped?
            local_gpu_idx: If `num_gpus_per_learner` > 0, and
                `num_learners` < 2, then RLlib uses this GPU index for training. This is
                an index into the available
                CUDA devices. For example if `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`
                and `local_gpu_idx=0`, RLlib uses the GPU with ID=1 on the node.
            max_requests_in_flight_per_learner: Max number of in-flight requests
                to each Learner (actor). You normally do not have to tune this setting
                (default is 3), however, for asynchronous algorithms, this determines
                the "queue" size for incoming batches (or lists of episodes) into each
                Learner worker, thus also determining, how much off-policy'ness would be
                acceptable. The off-policy'ness is the difference between the numbers of
                updates a policy has undergone on the Learner vs the EnvRunners.
                See the `ray.rllib.utils.actor_manager.FaultTolerantActorManager` class
                for more details.
            learner_class: The `Learner` class to use for (distributed) updating of the
                RLModule.
            learner_connector: A callable taking an env observation space and an env
                action space as inputs and returning a learner ConnectorV2 or
                list of ConnectorV2's as part of pipeline object.
            add_default_connectors_to_learner_pipeline: If True (default), RLlib's
                Learners automatically add the default Learner ConnectorV2
                pieces to the LearnerPipeline. These automatically perform:
                a) adding observations from episodes to the train batch, if this has not
                already been done by a user-provided connector piece
                b) if RLModule is stateful, add a time rank to the train batch, zero-pad
                the data, and add the correct state inputs, if this has not already been
                done by a user-provided connector piece.
                c) add all other information (actions, rewards, terminateds, etc..) to
                the train batch, if this has not already been done by a user-provided
                connector piece.
                Only if you know exactly what you are doing, you
                should set this setting to False.
            learner_config_dict: A dict to insert any settings accessible from within
                the Learner instance. This should only be used in connection with custom
                Learner subclasses and in case the user doesn't want to write an extra
                `AlgorithmConfig` subclass just to add a few settings to the base Algo's
                own config class.
        """
        return cfg.python_environment(**kwargs)

    @rllib_config_mutator
    def callbacks(cfg, **kwargs) -> None:
        """Sets the callbacks configuration.

        Args:
            callbacks_class: RLlibCallback class, whose methods are called during
                various phases of training and RL environment sample collection.
                TODO (sven): Change the link to new rst callbacks page.
                See the `RLlibCallback` class and
                `examples/metrics/custom_metrics_and_callbacks.py` for more information.
            on_algorithm_init: A callable or a list of callables. If a list, RLlib calls
                the items in the same sequence. `on_algorithm_init` methods overridden
                in `callbacks_class` take precedence and are called first.
                See
                :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_algorithm_init`  # noqa
                for more information.
            on_evaluate_start: A callable or a list of callables. If a list, RLlib calls
                the items in the same sequence. `on_evaluate_start` methods overridden
                in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_evaluate_start`  # noqa
                for more information.
            on_evaluate_end: A callable or a list of callables. If a list, RLlib calls
                the items in the same sequence. `on_evaluate_end` methods overridden
                in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_evaluate_end`  # noqa
                for more information.
            on_env_runners_recreated: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_env_runners_recreated`
                methods overridden in `callbacks_class` take precedence and are called
                first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_env_runners_recreated`  # noqa
                for more information.
            on_checkpoint_loaded: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_checkpoint_loaded`
                methods overridden in `callbacks_class` take precedence and are called
                first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_checkpoint_loaded`  # noqa
                for more information.
            on_environment_created: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_environment_created`
                methods overridden in `callbacks_class` take precedence and are called
                first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_environment_created`  # noqa
                for more information.
            on_episode_created: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_episode_created` methods
                overridden in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_created`  # noqa
                for more information.
            on_episode_start: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_episode_start` methods
                overridden in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_start`  # noqa
                for more information.
            on_episode_step: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_episode_step` methods
                overridden in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step`  # noqa
                for more information.
            on_episode_end: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_episode_end` methods
                overridden in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_end`  # noqa
                for more information.
            on_sample_end: A callable or a list of callables. If a list,
                RLlib calls the items in the same sequence. `on_sample_end` methods
                overridden in `callbacks_class` take precedence and are called first.
                See :py:meth:`~ray.rllib.callbacks.callbacks.RLlibCallback.on_sample_end`  # noqa
                for more information.

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.callbacks(**kwargs)

    @rllib_config_mutator
    def evaluation(cfg, **kwargs) -> None:
        """Sets the config's evaluation settings.

        Args:
            evaluation_interval: Evaluate with every `evaluation_interval` training
                iterations. The evaluation stats are reported under the "evaluation"
                metric key. Set to None (or 0) for no evaluation.
            evaluation_duration: Duration for which to run evaluation each
                `evaluation_interval`. The unit for the duration can be set via
                `evaluation_duration_unit` to either "episodes" (default) or
                "timesteps". If using multiple evaluation workers (EnvRunners) in the
                `evaluation_num_env_runners > 1` setting, the amount of
                episodes/timesteps to run are split amongst these.
                A special value of "auto" can be used in case
                `evaluation_parallel_to_training=True`. This is the recommended way when
                trying to save as much time on evaluation as possible. The Algorithm
                then runs as many timesteps via the evaluation workers as possible,
                while not taking longer than the parallely running training step and
                thus, never wasting any idle time on either training- or evaluation
                workers. When using this setting (`evaluation_duration="auto"`), it is
                strongly advised to set `evaluation_interval=1` and
                `evaluation_force_reset_envs_before_iteration=True` at the same time.
            evaluation_duration_unit: The unit, with which to count the evaluation
                duration. Either "episodes" (default) or "timesteps". Note that this
                setting is ignored if `evaluation_duration="auto"`.
            evaluation_auto_duration_min_env_steps_per_sample: If `evaluation_duration`
                is "auto" (in which case `evaluation_duration_unit` is always
                "timesteps"), at least how many timesteps should be done per remote
                `sample()` call.
            evaluation_auto_duration_max_env_steps_per_sample: If `evaluation_duration`
                is "auto" (in which case `evaluation_duration_unit` is always
                "timesteps"), at most how many timesteps should be done per remote
                `sample()` call.
            evaluation_sample_timeout_s: The timeout (in seconds) for evaluation workers
                to sample a complete episode in the case your config settings are:
                `evaluation_duration != auto` and `evaluation_duration_unit=episode`.
                After this time, the user receives a warning and instructions on how
                to fix the issue.
            evaluation_parallel_to_training: Whether to run evaluation in parallel to
                the `Algorithm.training_step()` call, using threading. Default=False.
                E.g. for evaluation_interval=1 -> In every call to `Algorithm.train()`,
                the `Algorithm.training_step()` and `Algorithm.evaluate()` calls
                run in parallel. Note that this setting - albeit extremely efficient b/c
                it wastes no extra time for evaluation - causes the evaluation results
                to lag one iteration behind the rest of the training results. This is
                important when picking a good checkpoint. For example, if iteration 42
                reports a good evaluation `episode_return_mean`, be aware that these
                results were achieved on the weights trained in iteration 41, so you
                should probably pick the iteration 41 checkpoint instead.
            evaluation_force_reset_envs_before_iteration: Whether all environments
                should be force-reset (even if they are not done yet) right before
                the evaluation step of the iteration begins. Setting this to True
                (default) makes sure that the evaluation results aren't polluted with
                episode statistics that were actually (at least partially) achieved with
                an earlier set of weights. Note that this setting is only
                supported on the new API stack w/ EnvRunners and ConnectorV2
                (`config.enable_rl_module_and_learner=True` AND
                `config.enable_env_runner_and_connector_v2=True`).
            evaluation_config: Typical usage is to pass extra args to evaluation env
                creator and to disable exploration by computing deterministic actions.
                IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
                policy, even if this is a stochastic one. Setting "explore=False" here
                results in the evaluation workers not using this optimal policy!
            off_policy_estimation_methods: Specify how to evaluate the current policy,
                along with any optional config parameters. This only has an effect when
                reading offline experiences ("input" is not "sampler").
                Available keys:
                {ope_method_name: {"type": ope_type, ...}} where `ope_method_name`
                is a user-defined string to save the OPE results under, and
                `ope_type` can be any subclass of OffPolicyEstimator, e.g.
                ray.rllib.offline.estimators.is::ImportanceSampling
                or your own custom subclass, or the full class path to the subclass.
                You can also add additional config arguments to be passed to the
                OffPolicyEstimator in the dict, e.g.
                {"qreg_dr": {"type": DoublyRobust, "q_model_type": "qreg", "k": 5}}
            ope_split_batch_by_episode: Whether to use SampleBatch.split_by_episode() to
                split the input batch to episodes before estimating the ope metrics. In
                case of bandits you should make this False to see improvements in ope
                evaluation speed. In case of bandits, it is ok to not split by episode,
                since each record is one timestep already. The default is True.
            evaluation_num_env_runners: Number of parallel EnvRunners to use for
                evaluation. Note that this is set to zero by default, which means
                evaluation is run in the algorithm process (only if
                `evaluation_interval` is not 0 or None). If you increase this, also
                increases the Ray resource usage of the algorithm since evaluation
                workers are created separately from those EnvRunners used to sample data
                for training.
            custom_evaluation_function: Customize the evaluation method. This must be a
                function of signature (algo: Algorithm, eval_workers: EnvRunnerGroup) ->
                (metrics: dict, env_steps: int, agent_steps: int) (metrics: dict if
                `enable_env_runner_and_connector_v2=True`), where `env_steps` and
                `agent_steps` define the number of sampled steps during the evaluation
                iteration. See the Algorithm.evaluate() method to see the default
                implementation. The Algorithm guarantees all eval workers have the
                latest policy state before this function is called.
            offline_evaluation_interval: Evaluate offline with every
                `offline_evaluation_interval` training iterations. The offline evaluation
                stats are reported under the "evaluation/offline_evaluation" metric key. Set
                to None (or 0) for no offline evaluation.
            num_offline_eval_runners: Number of OfflineEvaluationRunner actors to create
                for parallel evaluation. Setting this to 0 forces sampling to be done in the
                local OfflineEvaluationRunner (main process or the Algorithm's actor when
                using Tune).
            offline_evaluation_type: Type of offline evaluation to run. Either `"eval_loss"`
                for evaluating the validation loss of the policy, `"is"` for importance
                sampling, or `"pdis"` for per-decision importance sampling. If you want to
                implement your own offline evaluation method write an `OfflineEvaluationRunner`
                and use the `AlgorithmConfig.offline_eval_runner_class`.
            offline_eval_runner_class: An `OfflineEvaluationRunner` class that implements
                custom offline evaluation logic.
            offline_loss_for_module_fn: A callable to compute the loss per `RLModule` in
                offline evaluation. If not provided the training loss function (
                `Learner.compute_loss_for_module`) is used. The signature must be (
                runner: OfflineEvaluationRunner, module_id: ModuleID, config: AlgorithmConfig,
                batch: Dict[str, Any], fwd_out: Dict[str, TensorType]).
            offline_eval_batch_size_per_runner: Evaluation batch size per individual
                OfflineEvaluationRunner worker. This setting only applies to the new API
                stack. The number of OfflineEvaluationRunner workers can be set via
                `config.evaluation(num_offline_eval_runners=...)`. The total effective batch
                size is then `num_offline_eval_runners` x
                `offline_eval_batch_size_per_runner`.
            dataset_num_iters_per_offline_eval_runner: Number of batches to evaluate in each
                OfflineEvaluationRunner during a single evaluation. If None, each learner runs a
                complete epoch over its data block (the dataset is partitioned into
                at least as many blocks as there are runners). The default is `1`.
            offline_eval_rl_module_inference_only: If `True`, the module spec is used in an
                inference-only setting (no-loss) and the RLModule can thus be built in
                its light version (if available). For example, the `inference_only`
                version of an RLModule might only contain the networks required for
                computing actions, but misses additional target- or critic networks.
                Also, if `True`, the module does NOT contain those (sub) RLModules that have
                their `learner_only` flag set to True.
            num_cpus_per_offline_eval_runner: Number of CPUs to allocate per
                OfflineEvaluationRunner.
            num_gpus_per_offline_eval_runner: Number of GPUs to allocate per
                OfflineEvaluationRunner. This can be fractional. This is usually needed only if
                your (custom) loss function itself requires a GPU (i.e., it contains GPU-
                intensive computations), or model inference is unusually expensive.
            custom_resources_per_eval_runner: Any custom Ray resources to allocate per
                OfflineEvaluationRunner.
            offline_evaluation_timeout_s: The timeout in seconds for calling `run()` on remote
                OfflineEvaluationRunner workers. Results (episode list) from workers that take
                longer than this time are discarded.
            max_requests_in_flight_per_offline_eval_runner: Max number of in-flight requests
                to each OfflineEvaluationRunner (actor)). See the
                `ray.rllib.utils.actor_manager.FaultTolerantActorManager` class for more
                details.
                Tuning these values is important when running experiments with
                large evaluation batches, where there is the risk that the object store may
                fill up, causing spilling of objects to disk. This can cause any
                asynchronous requests to become very slow, making your experiment run
                slowly as well. You can inspect the object store during your experiment
                through a call to `ray memory` on your head node, and by using the Ray
                dashboard. If you're seeing that the object store is filling up,
                turn down the number of remote requests in flight or enable compression
                or increase the object store memory through, for example:
                `ray.init(object_store_memory=10 * 1024 * 1024 * 1024)  # =10 GB`.
            broadcast_offline_eval_runner_states: True, if merged OfflineEvaluationRunner
                states (from the central connector pipelines) should be broadcast back to
                all remote OfflineEvaluationRunner actors.
            validate_offline_eval_runners_after_construction: Whether to validate that each
                created remote OfflineEvaluationRunner is healthy after its construction process.
            restart_failed_offline_eval_runners: Whether - upon an OfflineEvaluationRunner
                failure - RLlib tries to restart the lost OfflineEvaluationRunner(s) as an
                identical copy of the failed one(s). You should set this to True when training
                on SPOT instances that may preempt any time and/or if you need to evaluate always a
                complete dataset b/c OfflineEvaluationRunner(s) evaluate through streaming split
                iterators on disjoint batches. The new, recreated OfflineEvaluationRunner(s) only
                differ from the failed one in their `self.recreated_worker=True` property value
                and have the same `worker_index` as the original(s). If this setting is True, the
                value of the `ignore_offline_eval_runner_failures` setting is ignored.
            ignore_offline_eval_runner_failures: Whether to ignore any OfflineEvalautionRunner
                failures and continue running with the remaining OfflineEvaluationRunners. This
                setting is ignored, if `restart_failed_offline_eval_runners=True`.
            max_num_offline_eval_runner_restarts: The maximum number of times any
                OfflineEvaluationRunner is allowed to be restarted (if
                `restart_failed_offline_eval_runners` is True).
            offline_eval_runner_health_probe_timeout_s: Max amount of time in seconds, we should
                spend waiting for OfflineEvaluationRunner health probe calls
                (`OfflineEvaluationRunner.ping.remote()`) to respond. Health pings are very cheap,
                however, we perform the health check via a blocking `ray.get()`, so the
                default value should not be too large.
            offline_eval_runner_restore_timeout_s: Max amount of time we should wait to restore
                states on recovered OfflineEvaluationRunner actors. Default is 30 mins.

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.evaluation(**kwargs)

    @rllib_config_mutator
    def offline_data(cfg, **kwargs) -> None:
        return cfg.offline_data(**kwargs)

    @rllib_config_mutator
    def multi_agent(cfg, **kwargs) -> None:
        """Sets the config's multi-agent settings.

        Validates the new multi-agent settings and translates everything into
        a unified multi-agent setup format. For example a `policies` list or set
        of IDs is properly converted into a dict mapping these IDs to PolicySpecs.

        Args:
            policies: Map of type MultiAgentPolicyConfigDict from policy ids to either
                4-tuples of (policy_cls, obs_space, act_space, config) or PolicySpecs.
                These tuples or PolicySpecs define the class of the policy, the
                observation- and action spaces of the policies, and any extra config.
            policy_map_capacity: Keep this many policies in the "policy_map" (before
                writing least-recently used ones to disk/S3).
            policy_mapping_fn: Function mapping agent ids to policy ids. The signature
                is: `(agent_id, episode, worker, **kwargs) -> PolicyID`.
            policies_to_train: Determines those policies that should be updated.
                Options are:
                - None, for training all policies.
                - An iterable of PolicyIDs that should be trained.
                - A callable, taking a PolicyID and a SampleBatch or MultiAgentBatch
                and returning a bool (indicating whether the given policy is trainable
                or not, given the particular batch). This allows you to have a policy
                trained only on certain data (e.g. when playing against a certain
                opponent).
            policy_states_are_swappable: Whether all Policy objects in this map can be
                "swapped out" via a simple `state = A.get_state(); B.set_state(state)`,
                where `A` and `B` are policy instances in this map. You should set
                this to True for significantly speeding up the PolicyMap's cache lookup
                times, iff your policies all share the same neural network
                architecture and optimizer types. If True, the PolicyMap doesn't
                have to garbage collect old, least recently used policies, but instead
                keeps them in memory and simply override their state with the state of
                the most recently accessed one.
                For example, in a league-based training setup, you might have 100s of
                the same policies in your map (playing against each other in various
                combinations), but all of them share the same state structure
                (are "swappable").
            observation_fn: Optional function that can be used to enhance the local
                agent observations to include more state. See
                rllib/evaluation/observation_function.py for more info.
            count_steps_by: Which metric to use as the "batch size" when building a
                MultiAgentBatch. The two supported values are:
                "env_steps": Count each time the env is "stepped" (no matter how many
                multi-agent actions are passed/how many multi-agent observations
                have been returned in the previous step).
                "agent_steps": Count each individual agent step as one step.
        """
        return cfg.multi_agent(**kwargs)

    @rllib_config_mutator
    def reporting(cfg, **kwargs) -> None:
        """Sets the config's reporting settings.

        Args:
            keep_per_episode_custom_metrics: Store raw custom metrics without
                calculating max, min, mean
            metrics_episode_collection_timeout_s: Wait for metric batches for at most
                this many seconds. Those that have not returned in time are collected
                in the next train iteration.
            metrics_num_episodes_for_smoothing: Smooth rollout metrics over this many
                episodes, if possible.
                In case rollouts (sample collection) just started, there may be fewer
                than this many episodes in the buffer and we'll compute metrics
                over this smaller number of available episodes.
                In case there are more than this many episodes collected in a single
                training iteration, use all of these episodes for metrics computation,
                meaning don't ever cut any "excess" episodes.
                Set this to 1 to disable smoothing and to always report only the most
                recently collected episode's return.
            min_time_s_per_iteration: Minimum time (in sec) to accumulate within a
                single `Algorithm.train()` call. This value does not affect learning,
                only the number of times `Algorithm.training_step()` is called by
                `Algorithm.train()`. If - after one such step attempt, the time taken
                has not reached `min_time_s_per_iteration`, performs n more
                `Algorithm.training_step()` calls until the minimum time has been
                consumed. Set to 0 or None for no minimum time.
            min_train_timesteps_per_iteration: Minimum training timesteps to accumulate
                within a single `train()` call. This value does not affect learning,
                only the number of times `Algorithm.training_step()` is called by
                `Algorithm.train()`. If - after one such step attempt, the training
                timestep count has not been reached, performs n more
                `training_step()` calls until the minimum timesteps have been
                executed. Set to 0 or None for no minimum timesteps.
            min_sample_timesteps_per_iteration: Minimum env sampling timesteps to
                accumulate within a single `train()` call. This value does not affect
                learning, only the number of times `Algorithm.training_step()` is
                called by `Algorithm.train()`. If - after one such step attempt, the env
                sampling timestep count has not been reached, performs n more
                `training_step()` calls until the minimum timesteps have been
                executed. Set to 0 or None for no minimum timesteps.
            log_gradients: Log gradients to results. If this is `True` the global norm
                of the gradients dictionary for each optimizer is logged to results.
                The default is `False`.
            custom_stats_cls_lookup: A dictionary mapping stat names to their corresponding Stats classes.
                The Stats classes should be subclasses of :py:class:`~ray.rllib.utils.metrics.stats.StatsBase`.
                The keys of the dictionary are the stat names, and the values are the corresponding Stats classes.
                This allows you to use your own Stats classes for logging metrics.
                You can replace existing values to override some behaviour of RLlib.
                You can add key-value-pairs to the dictionary to add new stats classes that will be available
                when logging values with the MetricsLogger throughout RLlib.

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.reporting(**kwargs)

    @rllib_config_mutator
    def checkpointing(cfg, **kwargs) -> None:
        return cfg.checkpointing(**kwargs)

    @rllib_config_mutator
    def fault_tolerance(cfg, **kwargs) -> None:
        """Sets the config's fault tolerance settings.

        Args:
            restart_failed_env_runners: Whether - upon an EnvRunner failure - RLlib
                tries to restart the lost EnvRunner(s) as an identical copy of the
                failed one(s). You should set this to True when training on SPOT
                instances that may preempt any time. The new, recreated EnvRunner(s)
                only differ from the failed one in their `self.recreated_worker=True`
                property value and have the same `worker_index` as the original(s).
                If this setting is True, the value of the `ignore_env_runner_failures`
                setting is ignored.
            ignore_env_runner_failures: Whether to ignore any EnvRunner failures
                and continue running with the remaining EnvRunners. This setting is
                ignored, if `restart_failed_env_runners=True`.
            max_num_env_runner_restarts: The maximum number of times any EnvRunner
                is allowed to be restarted (if `restart_failed_env_runners` is True).
            delay_between_env_runner_restarts_s: The delay (in seconds) between two
                consecutive EnvRunner restarts (if `restart_failed_env_runners` is
                True).
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, the
                EnvRunner tries to restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environment and without
                the EnvRunner crashing. You can raise
                `ray.rllib.env.env_runner.StepFailedRecreateEnvError` from your
                environment's `step` method to not log the error.
            num_consecutive_env_runner_failures_tolerance: The number of consecutive
                times an EnvRunner failure (also for evaluation) is tolerated before
                finally crashing the Algorithm. Only useful if either
                `ignore_env_runner_failures` or `restart_failed_env_runners` is True.
                Note that for `restart_failed_sub_environments` and sub-environment
                failures, the EnvRunner itself is NOT affected and won't throw any
                errors as the flawed sub-environment is silently restarted under the
                hood.
            env_runner_health_probe_timeout_s: Max amount of time in seconds, we should
                spend waiting for EnvRunner health probe calls
                (`EnvRunner.ping.remote()`) to respond. Health pings are very cheap,
                however, we perform the health check via a blocking `ray.get()`, so the
                default value should not be too large.
            env_runner_restore_timeout_s: Max amount of time we should wait to restore
                states on recovered EnvRunner actors. Default is 30 mins.

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.fault_tolerance(**kwargs)

    @rllib_config_mutator
    def rl_module(cfg, **kwargs) -> None:
        """Sets the config's RLModule settings.

        Args:
            model_config: The DefaultModelConfig object (or a config dictionary) passed
                as `model_config` arg into each RLModule's constructor. This is used
                for all RLModules, if not otherwise specified through `rl_module_spec`.
            rl_module_spec: The RLModule spec to use for this config. It can be either
                a RLModuleSpec or a MultiRLModuleSpec. If the
                observation_space, action_space, catalog_class, or the model config is
                not specified it is inferred from the env and other parts of the
                algorithm config object.
            algorithm_config_overrides_per_module: Only used if
                `enable_rl_module_and_learner=True`.
                A mapping from ModuleIDs to per-module AlgorithmConfig override dicts,
                which apply certain settings,
                e.g. the learning rate, from the main AlgorithmConfig only to this
                particular module (within a MultiRLModule).
                You can create override dicts by using the `AlgorithmConfig.overrides`
                utility. For example, to override your learning rate and (PPO) lambda
                setting just for a single RLModule with your MultiRLModule, do:
                config.multi_agent(algorithm_config_overrides_per_module={
                "module_1": PPOConfig.overrides(lr=0.0002, lambda_=0.75),
                })

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.rl_module(**kwargs)

    @rllib_config_mutator
    def experimental(cfg, **kwargs) -> None:
        """Sets the config's experimental settings.

        Args:
            _validate_config: Whether to run `validate()` on this config. True by
                default. If False, ignores any calls to `self.validate()`.
            _use_msgpack_checkpoints: Create state files in all checkpoints through
                msgpack rather than pickle.
            _torch_grad_scaler_class: Class to use for torch loss scaling (and gradient
                unscaling). The class must implement the following methods to be
                compatible with a `TorchLearner`. These methods/APIs match exactly those
                of torch's own `torch.amp.GradScaler` (see here for more details
                https://pytorch.org/docs/stable/amp.html#gradient-scaling):
                `scale([loss])` to scale the loss by some factor.
                `get_scale()` to get the current scale factor value.
                `step([optimizer])` to unscale the grads (divide by the scale factor)
                and step the given optimizer.
                `update()` to update the scaler after an optimizer step (for example to
                adjust the scale factor).
            _torch_lr_scheduler_classes: A list of `torch.lr_scheduler.LRScheduler`
                (see here for more details
                https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
                classes or a dictionary mapping module IDs to such a list of respective
                scheduler classes. Multiple scheduler classes can be applied in sequence
                and are stepped in the same sequence as defined here. Note, most
                learning rate schedulers need arguments to be configured, that is, you
                might have to partially initialize the schedulers in the list(s) using
                `functools.partial`.
            _tf_policy_handles_more_than_one_loss: Experimental flag.
                If True, TFPolicy handles more than one loss or optimizer.
                Set this to True, if you would like to return more than
                one loss term from your `loss_fn` and an equal number of optimizers
                from your `optimizer_fn`.
            _disable_preprocessor_api: Experimental flag.
                If True, no (observation) preprocessor is created and
                observations arrive in model as they are returned by the env.
            _disable_action_flattening: Experimental flag.
                If True, RLlib doesn't flatten the policy-computed actions into
                a single tensor (for storage in SampleCollectors/output files/etc..),
                but leave (possibly nested) actions as-is. Disabling flattening affects:
                - SampleCollectors: Have to store possibly nested action structs.
                - Models that have the previous action(s) as part of their input.
                - Algorithms reading from offline files (incl. action information).

        Returns:
            This updated AlgorithmConfig object.
        """
        return cfg.experimental(**kwargs)

    def _apply_agents_to_rllib(self) -> list[AgentID]:
        policies = {}
        agent_type_map = {}
        agents: list[AgentID] = []

        for agent_type, spec in self.agent_specs.items():
            policies[spec.get("policy")] = (
                None,
                spec.get("observation_space"),
                spec.get("action_space"),
                {},
            )

            for i in range(spec.get("count")):
                agent_id = f"{agent_type}:{i}"
                agents.append(agent_id)
                agent_type_map[agent_id] = spec.get("policy")

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return agent_type_map[agent_id]

        self.rllib_cfg = self.rllib_cfg.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(set(agent_type_map.values())),
        )
        return agents

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict[str, AgentSpec]) -> Self:
        self.agent_specs = agents
        return self

    @override(OptimizerConfig)
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        world_name: Optional[str] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ):
        if self.rllib_cfg is None:
            self.rllib_cfg = self.algo_class.get_default_config()
            for op in self._cfg_ops:
                self.rllib_cfg = op(self.rllib_cfg)

        cfg = self.rllib_cfg.copy(copy_frozen=True)
        if self.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        env_name = f"regulated_env_{uuid.uuid4().hex}"

        if world is not None:
            if world_name is None:
                raise ValueError(
                    "world_name must be provided when using Ray world actor"
                )
            self.world_name = world_name
            registry = ray.get(world.get_opt_registry.remote())
            # Register the new ID and get the result
            opt_id = ray.get(
                world._set_new_opt_id.remote(opt_id=generate_uuid(registry))
            )
        if self.agent_specs:
            agents = self._apply_agents_to_rllib()

        def env_creator(env_ctx):
            return self._env_creator(
                world=world, opt_id=opt_id, agents=agents, **dict(env_ctx)
            )

        register_env(env_name, env_creator)
        self.rllib_cfg = self.rllib_cfg.environment(
            env=env_name, env_config=self.env_config
        )

        algo = self.rllib_cfg.build_algo(**kwargs)
        opt = RayOptimizer(algo=algo, config=cfg)

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt.set_id(opt_id)

        return opt

    @rllib_config_mutator
    @override(OptimizerConfig)
    def freeze(cfg, **kwargs) -> None:
        return cfg.freeze(**kwargs)

    @rllib_config_mutator
    @override(OptimizerConfig)
    def training(cfg, **kwargs) -> Self:
        """Sets the training related configuration.

        Args:
            gamma: Float specifying the discount factor of the Markov Decision process.
            lr: The learning rate (float) or learning rate schedule in the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                In case of a schedule, intermediary timesteps are assigned to
                linearly interpolated learning rate values. A schedule config's first
                entry must start with timestep 0, i.e.: [[0, initial_value], [...]].
                Note: If you require a) more than one optimizer (per RLModule),
                b) optimizer types that are not Adam, c) a learning rate schedule that
                is not a linearly interpolated, piecewise schedule as described above,
                or d) specifying c'tor arguments of the optimizer that are not the
                learning rate (e.g. Adam's epsilon), then you must override your
                Learner's `configure_optimizer_for_module()` method and handle
                lr-scheduling yourself.
            grad_clip: If None, no gradient clipping is applied. Otherwise,
                depending on the setting of `grad_clip_by`, the (float) value of
                `grad_clip` has the following effect:
                If `grad_clip_by=value`: Clips all computed gradients individually
                inside the interval [-`grad_clip`, +`grad_clip`].
                If `grad_clip_by=norm`, computes the L2-norm of each weight/bias
                gradient tensor individually and then clip all gradients such that these
                L2-norms do not exceed `grad_clip`. The L2-norm of a tensor is computed
                via: `sqrt(SUM(w0^2, w1^2, ..., wn^2))` where w[i] are the elements of
                the tensor (no matter what the shape of this tensor is).
                If `grad_clip_by=global_norm`, computes the square of the L2-norm of
                each weight/bias gradient tensor individually, sum up all these squared
                L2-norms across all given gradient tensors (e.g. the entire module to
                be updated), square root that overall sum, and then clip all gradients
                such that this global L2-norm does not exceed the given value.
                The global L2-norm over a list of tensors (e.g. W and V) is computed
                via:
                `sqrt[SUM(w0^2, w1^2, ..., wn^2) + SUM(v0^2, v1^2, ..., vm^2)]`, where
                w[i] and v[j] are the elements of the tensors W and V (no matter what
                the shapes of these tensors are).
            grad_clip_by: See `grad_clip` for the effect of this setting on gradient
                clipping. Allowed values are `value`, `norm`, and `global_norm`.
            train_batch_size_per_learner: Train batch size per individual Learner
                worker. This setting only applies to the new API stack. The number
                of Learner workers can be set via `config.resources(
                num_learners=...)`. The total effective batch size is then
                `num_learners` x `train_batch_size_per_learner` and you can
                access it with the property `AlgorithmConfig.total_train_batch_size`.
            train_batch_size: Training batch size, if applicable. When on the new API
                stack, this setting should no longer be used. Instead, use
                `train_batch_size_per_learner` (in combination with
                `num_learners`).
            num_epochs: The number of complete passes over the entire train batch (per
                Learner). Each pass might be further split into n minibatches (if
                `minibatch_size` provided).
            minibatch_size: The size of minibatches to use to further split the train
                batch into.
            shuffle_batch_per_epoch: Whether to shuffle the train batch once per epoch.
                If the train batch has a time rank (axis=1), shuffling only takes
                place along the batch axis to not disturb any intact (episode)
                trajectories.
            model: Arguments passed into the policy model. See models/catalog.py for a
                full list of the available model options.
                TODO: Provide ModelConfig objects instead of dicts.
            optimizer: Arguments to pass to the policy optimizer. This setting is not
                used when `enable_rl_module_and_learner=True`.
        """
        return cfg.training(**kwargs)

    # @override(OptimizerConfig)
    # def environment(self, *, env=None, **kwargs) -> Self:
    #     super().__init__
    #     self.env = env
    #     return self
