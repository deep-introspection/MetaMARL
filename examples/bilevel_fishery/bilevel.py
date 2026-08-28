import yaml
import numpy as np
from gymnasium import spaces

from ray.rllib.models import ModelCatalog

from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig
from examples.registry import REGISTRY

# Register model once
ModelCatalog.register_custom_model("mps_fcnet", MPSFullyConnectedNetwork)


# Optional: central callback registry
from core.callbacks import tag_episode_with_env_idx

CALLBACKS = {
    "tag_episode_with_env_idx": tag_episode_with_env_idx,
}


class BilevelConfigLoader:
    @staticmethod
    def from_yaml(path: str, output_dir: str | None = None) -> BilevelConfig:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # =========================
        # MECHANISM
        # =========================
        mechanism_space_cls = REGISTRY["mechanism_space"][cfg["mechanism"]["space"]]

        scaling_cfg = cfg["mechanism"].get("scaling", {})
        default_cfg = cfg["mechanism"]["default"]

        mechanism_space = mechanism_space_cls(
            max_fine=scaling_cfg.get("max_fine", 10.0),
            default_fixed_quota=default_cfg.get("fixed_quota", 0.25),
            default_prop_quota=default_cfg.get("prop_quota", 0.25),
            default_min_stock=default_cfg.get("min_stock", 0.4),
            default_target_stock=default_cfg.get("target_stock", 0.6),
            default_fine_amount=default_cfg.get("fine_amount", 10.0),
            default_risk_penalty_scale=default_cfg.get("risk_penalty_scale", 8.0),
            default_risk_penalty_power=default_cfg.get("risk_penalty_power", 2.0),
        )

        # =========================
        # BASE CONFIG
        # =========================
        bilevel = (
            BilevelConfig()
            .world(world_name=cfg["world"]["world_name"])
            .mechanism(space=mechanism_space)
            .training(
                outer_iters=cfg["training"]["outer_iters"],
                output_dir=output_dir,
            )
            .ray(**cfg["ray"])
        )

        # =========================
        # REPORTING (optional)
        # =========================
        if "reporting" in cfg:
            bilevel = bilevel.reporting(**cfg["reporting"])

        # =========================
        # OUTER (ES)
        # =========================
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

        # =========================
        # INNER (APPO)
        # =========================
        optimizer_name = cfg["inner"].get("optimizer", "APPO")

        if optimizer_name != "APPO":
            raise ValueError(f"Unsupported inner optimizer: {optimizer_name}")

        inner_env_cls = REGISTRY["env"][cfg["inner"]["environment"]["env"]]

        fisher_cfg = cfg["inner"]["agents"]["fisher"]

        # Observation dimension
        base_obs_dim = fisher_cfg.get("observation_base_dim", 4)
        obs_dim = base_obs_dim + mechanism_space.full_dimension

        action_cfg = fisher_cfg["action_space"]

        # Resolve callbacks
        callbacks_cfg = cfg["inner"].get("callbacks", {})
        resolved_callbacks = {k: CALLBACKS[v] for k, v in callbacks_cfg.items()}

        bilevel = bilevel.inner(
            APPOptimizerConfig()
            .resources(**cfg["inner"]["resources"])
            .framework(**cfg["inner"]["framework"])
            .api_stack(**cfg["inner"]["api_stack"])
            .learners(**cfg["inner"]["learners"])
            .environment(
                env=inner_env_cls,
                env_config=cfg["inner"]["environment"]["env_config"],
                horizon=cfg["inner"]["environment"]["horizon"],
                disable_env_checking=cfg["inner"]["environment"].get(
                    "disable_env_checking", False
                ),
            )
            .env_runners(**cfg["inner"]["env_runners"])
            .training(**cfg["inner"]["training"])
            .evaluation(**cfg["inner"]["evaluation"])
            .callbacks(**resolved_callbacks)
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
