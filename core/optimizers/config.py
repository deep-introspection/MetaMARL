from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Self, Type, Union

import numpy as np
import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.envs.base import BaseEnv
from core.metrics.schemas import MetricSchema
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import EnvConfigDict, EnvType
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    def to_dict(self) -> dict:
        """Converts this configuration to dict format."""
        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
    # TODO registry to allow opt_class str
    # TODO runtime checking of opt_class
    def __init__(self, opt_class: Optional[Type[Optimizer]] = None):
        """Initializes an OptimizerConfig instance.

        Args:
            optimizer_class: An optional Optimizer class that this config class belongs to.
                Used (if provided) to build a respective Optimizer instance from this
                config.
        """
        self.opt_class = opt_class

        # -- lifecycle --
        self._is_frozen = False

        # --- world / environment ---
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.horizon: int = None  # TODO default value

        # --- debugging ---
        self.base_seed: Optional[int] = None
        self.seeds: list[int] = []

        # --- eval ---
        self.evaluation_config: Optional["OptimizerConfig"] = None
        self.eval_seeds: Optional[list[int]] = None

        # TODO
        # --- reporting ---
        self.stats_cls_lookup = DEFAULT_STATS_CLS_LOOKUP
        self._reporter_cfg: Optional[ReporterConfig] = None
        self._reporting_schema: Optional[type[MetricSchema]] = None
        self._reporting_queries: Optional[tuple[Query, ...]] = None
        # Declared through .environment(queries=, schema=): the env's own metrics.
        self._reporting_schema_env: Optional[type[MetricSchema]] = None
        self._reporting_queries_env: Optional[tuple[Query, ...]] = None

    @property
    def reporter_cfg(self) -> Optional[ReporterConfig]:
        return self._reporter_cfg

    @reporter_cfg.setter
    def reporter_cfg(self, reporter_cfg: ReporterConfig) -> None:
        self._reporter_cfg = reporter_cfg

    def __setattr__(self, name, value):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if name not in ["_is_frozen"]:
                raise AttributeError(
                    f"Cannot set attribute ({name}) of an already frozen "
                    "OptimizerConfig!"
                )
        super().__setattr__(name, value)

    # TODO generalize this function
    def _merge_env_config(self, extra: dict) -> Self:
        self.env_config = {
            **(self.env_config or {}),
            **extra,
        }
        return self

    # TODO freezing for nested configs
    def freeze(self) -> None:
        """Freeze this config object, such that no attributes can be set anymore.

        Optimizers should use this method to make sure their config objects
        remain read-only after this.
        """
        if self._is_frozen:
            return
        self._is_frozen = True

    def copy(self, copy_frozen: Optional[bool] = None) -> Self:
        """Creates a deep copy of this config and (un)freezes if necessary.

        Args:
            copy_frozen: Whether the created deep copy is frozen or not, If None,
                keep the same frozen status that 'self' currently has.

        Returns:
            A deep copy of 'self' that is (un)frozen.
        """
        cp = copy.deepcopy(self)
        if copy_frozen is True:
            cp.freeze()
        elif copy_frozen is False:
            cp._is_frozen = False
            if isinstance(cp.evaluation_config, OptimizerConfig):
                cp.evaluation_config._is_frozen = False
        return cp

    # TODO review this
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Serialization from dict"""
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # TODO review this
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Serialization from yaml"""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def _env_creator(
        self,
        **env_ctx,
    ) -> BaseEnv:
        return self.env(**env_ctx)

    def _build_reporter(self, *, label: str):
        """Instantiate the optimizer-level reporter from ``reporter_cfg``, if any."""
        if self._reporter_cfg is None:
            return None
        reporter = self._reporter_cfg.build(label=label)
        reporter.schema = self._reporting_schema
        reporter.add_query(*(self._reporting_queries or ()))
        return reporter

    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    # TODO move optimizer registration to executor in future
    # TODO enable multiple world registration
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs,
    ) -> Optimizer:
        """Builds an Optimizer from this OptimizerConfig (or a copy thereof)."""
        cfg = self.copy(copy_frozen=True)
        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        # TODO remove this in the future and create registry for world and optimizer. keep for now as safety guard
        opt: Optimizer = cfg.opt_class(config=cfg)

        opt.world = world

        # Build the optimizer-level reporter (None when no reporting is configured)
        opt.reporting = self._build_reporter(label=self.opt_class.__name__)

        # register optimizer in world to link contexts to optimizers
        opt_id = None
        if world is not None:
            opt_id = ray.get(world._set_new_opt_id.remote(opt_id=opt.opt_id))
            opt.set_id(opt_id)

        env = cfg._env_creator(
            world=world,
            opt_id=opt_id,
            optimizer=inner_opt,
            reporter_cfg=cfg.reporter_cfg.copy()
            if cfg.reporter_cfg is not None
            else None,
            queries=cfg._reporting_queries_env,
            schema=cfg._reporting_schema_env,
            **self.env_config,
        )
        opt.env = env

        return opt

    # TODO EnvConfigDict
    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
        train_iters: Optional[int] = None,
        horizon: Optional[int] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[type[MetricSchema]] = None,
        *,
        env_config: Optional[EnvConfigDict] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        disable_env_checking: Optional[bool] = None,
    ):
        """Sets the config's RL-environment settings.

        Args:
            env: The environment specifier. This can either be a tune-registered env,
                via `tune.register_env([name], lambda env_ctx: [env object])`,
                or a string specifier of an RLlib supported type. In the latter case,
                RLlib tries to interpret the specifier as either an Farama-Foundation
                gymnasium env, a PyBullet env, or a fully qualified classpath to an Env
                class, e.g. "ray.rllib.examples.envs.classes.random_env.RandomEnv".
            env_config: Arguments dict passed to the env creator as an EnvContext
                object (which is a dict plus the properties: `num_env_runners`,
                `worker_index`, `vector_index`, and `remote`).
            observation_space: The observation space for the Policies of this Algorithm.
            action_space: The action space for the Policies of this Algorithm.
            horizon: Rollout steps taken by the environment before termination.
            render_env: If True, try to render the environment on the local worker or on
                worker 1 (if num_env_runners > 0). For vectorized envs, this usually
                means that only the first sub-environment is rendered.
                In order for this to work, your env has to implement the
                `render()` method which either:
                a) handles window generation and rendering itself (returning True) or
                b) returns a numpy uint8 image of shape [height x width x 3 (RGB)].
            clip_rewards: Whether to clip rewards during Policy's postprocessing.
                None (default): Clip for Atari only (r=sign(r)).
                True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
                False: Never clip.
                [float value]: Clip at -value and + value.
                Tuple[value1, value2]: Clip at value1 and value2.
            normalize_actions: If True, RLlib learns entirely inside a normalized
                action space (0.0 centered with small stddev; only affecting Box
                components). RLlib unsquashes actions (and clip, just in case) to the
                bounds of the env's action space before sending actions back to the env.
            clip_actions: If True, the RLlib default ModuleToEnv connector clips
                actions according to the env's bounds (before sending them into the
                `env.step()` call).
            disable_env_checking: Disable RLlib's env checks after a gymnasium.Env
                instance has been constructed in an EnvRunner. Note that the checks
                include an `env.reset()` and `env.step()` (with a random action), which
                might tinker with your env's logic and behavior and thus negatively
                influence sample collection- and/or learning behavior.
            is_atari: This config can be used to explicitly specify whether the env is
                an Atari env or not. If not specified, RLlib tries to auto-detect
                this.
            action_mask_key: If observation is a dictionary, expect the value by
                the key `action_mask_key` to contain a valid actions mask (`numpy.int8`
                array of zeros and ones). Defaults to "action_mask".

        Returns:
            This updated AlgorithmConfig object.
        """
        self.env_config: dict = {}
        if env is not None:
            self.env = env
        if train_iters is not None:
            self.env_config.update({"train_iters": train_iters})
        if observation_space is not None:
            self.env_config.update({"observation_space": observation_space})
        if action_space is not None:
            self.env_config.update({"action_space": action_space})
        if horizon is not None:
            self.env_config.update({"horizon": horizon})
        if env_config is not None:
            self.env_config.update(env_config)
        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking
        if queries is not None:
            self._reporting_queries_env = tuple(queries)
        if schema is not None:
            self._reporting_schema_env = schema

        return self

    @abstractmethod
    def training(self):
        raise NotImplementedError

    def debugging(
        self,
        *,
        seed: Optional[int] = None,  # base seed
        num_seeds: int = 3,
    ) -> Self:
        if seed is not None:
            self.base_seed = seed
            ss = np.random.SeedSequence(seed)
            self.seeds = ss.generate_state(num_seeds).tolist()
        else:
            self.seeds = []
        return self

    def reporting(
        self,
        queries: Optional[tuple[Query, ...]] = None,
        schema: Optional[type[MetricSchema]] = None,
    ) -> Self:
        """Declare the optimizer-level metric schema and the queries to render from it."""
        if schema is not None:
            self._reporting_schema = schema
        if queries is not None:
            self._reporting_queries = tuple(queries)
        return self

    # TODO Docstring explanation
    # @abstractmethod
    # def ressources(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def evaluation(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def reporting(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def checkpointing(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def fault_tolerance(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def experimental(self):
    #     raise NotImplementedError
