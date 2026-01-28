from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
import torch
from gymnasium.core import ActType, ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism, VectorMechanism
from core.mechanism.space import MechanismSpace
from core.optimizers.base import Optimizer
from core.types import OptimizerID
from core.world.base import World
from core.world.context import EnvStepContext, MechanismContext


class RegulatorEnv(BaseEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        optimizer: Optional[Optimizer] = None,
        train_iters: int = 5,
        eval_iters: int = 2,
        mechanism_space: MechanismSpace = None,
        **kwargs,
    ):
        super().__init__(world=world, opt_id=opt_id, **kwargs)
        self.inner: Optimizer = optimizer
        self.train_iters: int = train_iters
        self.eval_iters: int = eval_iters
        self.mechanism_space: MechanismSpace = mechanism_space

        self._validate()

    def _validate(self):
        if self.inner is None:
            return  # analytic override mode allowed

        if self.train_iters <= 0:
            raise ValueError("train_iters must be >= 1 when optimizer is provided")

        if self.eval_iters <= 0:
            raise ValueError("eval_iters must be >= 1 when optimizer is provided")

    @override(BaseEnv)
    def _reset(self):
        if self.observation_space is None:
            return 0.0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    # TODO
    # @override(BaseEnv)
    # def _step(
    #     self, theta: Mechanism
    # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

    #     # Always publish mechanism
    #     self._publish(MechanismContext(theta=theta))

    #     return self._inner_step(theta)

    # @abstractmethod
    # def _inner_step(
    #     self, theta: Mechanism
    # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #     ...

    @override(BaseEnv)
    def _step(
        self, thetas: list[Mechanism]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.inner is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no inner optimizer. "
                f"Override `_step()` for analytic reward computation."
            )

        if not isinstance(thetas, list):
            raise TypeError(
                f"{self.__class__.__name__} expected list[Mechanism] after action(), "
                f"got {type(thetas)}"
            )

        for theta in thetas:
            self._publish(MechanismContext(theta=theta))

        for _ in range(self.train_iters):
            self.inner.run()

        return None, 0.0, False, False, {}

    # @abstractmethod
    # @override(BaseEnv)
    # def observation(self, observation: ObsType) -> ObsType:
    #     # read downstream results from optimizer and compute aggregate
    #     raise NotImplementedError

    @abstractmethod
    def aggregate_rewards(self, rewards: list[SupportsFloat]) -> SupportsFloat:
        return NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat = 0.0) -> SupportsFloat:
        # analytic path
        if self.inner is None:
            return reward

        ctx_registry = ray.get(self.world.get_ctx_registry.remote())

        # TODO introduce windowing later to allow only to aggregate rewards from this mechanism
        # Keep only EnvStepContexts produced by inner optimizer
        inner_ctxs = [
            ctx
            for ctx in ctx_registry.values()
            if ctx.opt_id == self.inner.opt_id
            and isinstance(ctx.payload, EnvStepContext)
        ]

        if not inner_ctxs:
            raise RuntimeError("No EnvStepContext published by inner optimizer")

        rewards = [float(ctx.payload.reward) for ctx in inner_ctxs]

        return self.aggregate_rewards(rewards)

    # @abstractmethod
    @override(BaseEnv)
    def action(self, action: ActType) -> list[Mechanism]:
        # analytic path
        if self.inner is None:
            return action

        # 1) Mechanism space path (preferred)
        if self.mechanism_space is not None:
            if isinstance(action, (list, tuple)):
                action = np.asarray(action, dtype=np.float32)

            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    action = action[None, :]  # (d,) -> (1, d)
                return [
                    self.mechanism_space.project(self.mechanism_space.from_vector(v))
                    for v in action
                ]

            if torch.is_tensor(action):
                return self.action(action.detach().cpu().numpy())

            raise TypeError(f"Unsupported action type: {type(action)}")

        # 2) No mechanism_space: still guarantee Mechanism objects
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=np.float32)

        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = action[None, :]
            return [VectorMechanism(v) for v in action]

        if torch.is_tensor(action):
            return self.action(action.detach().cpu().numpy())

        # If someone passed an already-built Mechanism, allow it
        # (e.g., analytic tests).
        if hasattr(action, "to_vector"):
            return [action]  # type: ignore

        raise TypeError(
            f"Unsupported action type with no mechanism_space: {type(action)}"
        )
