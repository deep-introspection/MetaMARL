import numpy as np
import yaml
from gymnasium import spaces
from ray.rllib.models import ModelCatalog

from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork
from core.registry import REGISTRY
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig

# Register custom MPS model
ModelCatalog.register_custom_model("mps_fcnet", MPSFullyConnectedNetwork)


class BilevelConfigLoader:
    @staticmethod
    def from_yaml(path: str, output_dir: str | None = None) -> BilevelConfig:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        mechanism_space_cls = REGISTRY["mechanism_space"][cfg["mechanism"]["space"]]
        scaling_cfg = cfg["mechanism"].get("scaling", {})
        mechanism_space = mechanism_space_cls(
            max_fine=scaling_cfg.get("max_fine", 5.0),
            max_ban=scaling_cfg.get("max_ban", 50),
        )

        mechanism_cls = REGISTRY["mechanism"]["WaterMechanism"]
        mechanism = mechanism_cls(
            **cfg["mechanism"]["default"],
            max_fine=scaling_cfg.get("max_fine", 5.0),
            max_ban=scaling_cfg.get("max_ban", 50),
        )

        bilevel = (
            BilevelConfig()
            .world(world_name=cfg["world"]["world_name"])
            .mechanism(space=mechanism_space_cls, default=mechanism)
            .training(outer_iters=cfg["training"]["outer_iters"], output_dir=output_dir)
            .ray(**cfg["ray"])
        )

        outer_env_cls = REGISTRY["env"][cfg["outer"]["environment"]["env"]]

        bilevel = bilevel.outer(
            ESConfig()
            .training(**cfg["outer"]["training"])
            .environment(
                env=outer_env_cls,
                env_config=cfg["outer"]["environment"].get("env_config", {}),
                horizon=cfg["outer"]["environment"].get("horizon"),
                train_iters=cfg["outer"]["environment"].get("train_iters", 1),
            )
        )

        inner_env_cls = REGISTRY["env"][cfg["inner"]["environment"]["env"]]

        utilizer_cfg = cfg["inner"]["agents"]["utilizer"]
        obs_dim = 5 + mechanism_space.dimension  # water_norm, placeholder, restriction, effective_quota, no_water_zone + θ

        # Build inner PPO config
        ppo_cfg = (
            PPOptimizerConfig()
            .resources(**cfg["inner"].get("resources", {}))
            .framework(**cfg["inner"].get("framework", {}))
            .model(**cfg["inner"].get("model", {}))
            .api_stack(**cfg["inner"].get("api_stack", {}))
        )

        if "learners" in cfg["inner"]:
            ppo_cfg = ppo_cfg.learners(**cfg["inner"]["learners"])

        action_cfg = utilizer_cfg["action_space"]

        ppo_cfg = (
            ppo_cfg
            .environment(
                env=inner_env_cls,
                env_config=cfg["inner"]["environment"].get("env_config", {}),
                horizon=cfg["inner"]["environment"].get("horizon"),
            )
            .env_runners(**cfg["inner"].get("env_runners", {}))
            .training(**cfg["inner"].get("training", {}))
            .evaluation(**cfg["inner"].get("evaluation", {}))
            .agents(
                {
                    "utilizer": {
                        "count": utilizer_cfg["count"],
                        "policy": utilizer_cfg["policy"],
                        "observation_space": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(obs_dim,),
                            dtype=np.float32,
                        ),
                        "action_space": spaces.Box(
                            low=float(action_cfg["low"]),
                            high=float(action_cfg["high"]),
                            shape=tuple(action_cfg["shape"]),
                            dtype=np.float32,
                        ),
                    }
                }
            )
            .fault_tolerance(**cfg["inner"].get("fault_tolerance", {}))
        )

        bilevel = bilevel.inner(ppo_cfg)

        return bilevel
