import numpy as np
import yaml
from gymnasium import spaces
from ray.rllib.models import ModelCatalog

from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.registry import REGISTRY

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

        mechanism_cls = REGISTRY["mechanism"]["FisheryMechanism"]
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
                env_config=cfg["outer"]["environment"]["env_config"],
                horizon=cfg["outer"]["environment"]["horizon"],
                train_iters=cfg["outer"]["environment"]["train_iters"],
            )
        )

        inner_env_cls = REGISTRY["env"][cfg["inner"]["environment"]["env"]]

        fisher_cfg = cfg["inner"]["agents"]["fisher"]
        obs_dim = 5 + mechanism_space.dimension  # fish, algae, ban, quota, no_fish_zone + θ
        action_cfg = fisher_cfg["action_space"]

        bilevel = bilevel.inner(
            PPOptimizerConfig()
            .resources(**cfg["inner"]["resources"])
            .framework(**cfg["inner"]["framework"])
            .model(**cfg["inner"]["model"])
            .api_stack(**cfg["inner"]["api_stack"])
            .learners(**cfg["inner"]["learners"])
            .environment(
                env=inner_env_cls,
                env_config=cfg["inner"]["environment"]["env_config"],
                horizon=cfg["inner"]["environment"]["horizon"],
            )
            .env_runners(**cfg["inner"]["env_runners"])
            .training(**cfg["inner"]["training"])
            .evaluation(**cfg["inner"]["evaluation"])
            .agents(
                {
                    "fisher": {
                        "count": fisher_cfg["count"],
                        "policy": fisher_cfg["policy"],
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
            .fault_tolerance(**cfg["inner"]["fault_tolerance"])
        )

        return bilevel
