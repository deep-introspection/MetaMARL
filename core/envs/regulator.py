from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
import torch
from gymnasium.core import ActType, ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism, VectorMechanism
from core.optimizers.base import Optimizer
from core.types import OptimizerID
from core.world.base import World
from core.world.context import (
    Context,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)


class RegulatorEnv(BaseEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        optimizer: Optional[Optimizer] = None,
        train_iters: int = 5,
        **kwargs,
    ):
        super().__init__(world=world, opt_id=opt_id, **kwargs)
        self.inner: Optimizer = optimizer
        self.train_iters: int = train_iters

        self._validate()

    def _validate(self):
        if self.inner is None:
            return  # analytic override mode allowed

        if self.train_iters <= 0:
            raise ValueError("train_iters must be >= 1 when optimizer is provided")

    @override(BaseEnv)
    def _reset(self):
        if self.observation_space is None:
            return 0.0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    @override(BaseEnv)
    def _step(
        self, mechanisms: list[Mechanism]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.inner is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no inner optimizer. "
                f"Override `_step()` for analytic reward computation."
            )

        if not isinstance(mechanisms, list) or not all(
            isinstance(t, Mechanism) for t in mechanisms
        ):
            raise TypeError(
                f"{self.__class__.__name__} expected list[Mechanism], got {type(mechanisms)}"
            )

        # TODO PARALLELIZE Vectorize environment across θ candidates and train one ppo policy over mehcanism candidates
        # TODO other techiniques can also speed this up
        # for theta in thetas:
        #     self._publish(MechanismContext(theta=theta))
        for idx, m in enumerate(mechanisms):
            self._publish(
                MechanismContext(
                    index=idx,
                    status=MechanismStatus.published,
                    job=MechanismStatus.train,
                    env_id=None,
                    mechanism=m,
                    metrics=None,
                )
            )

        # One policy conditioned on Theta (theta-conditioned RL)
        self.inner.run()

        # reset mechanism contexts
        for _ in range(self.inner.config.evaluation_duration):
            for idx, m in enumerate(mechanisms):
                self._publish(
                    MechanismContext(
                        index=idx,
                        status=MechanismStatus.published,
                        job=MechanismStatus.eval,
                        env_id=None,
                        mechanism=m,
                        metrics=None,
                    )
                )

        ctx_registry_before = set(ray.get(self.world.get_ctx_registry.remote()).keys())

        # TODO review env step geometry
        self.inner.evaluate()

        ctx_registry_after = ray.get(self.world.get_ctx_registry.remote())

        new_ctxs = [
            ctx
            for cid, ctx in ctx_registry_after.items()
            if cid not in ctx_registry_before
            and ctx.opt_id == self.inner.opt_id
            and isinstance(ctx.payload, EnvStepContext)
        ]

        # Evaluation metrics are defined by user and provided as a callable.
        # metrics = user_eval_fn(contexts)

        reward = self.aggregate_rewards(new_ctxs)

        return None, reward, False, False, {}

    # @abstractmethod
    # @override(BaseEnv)
    # def observation(self, observation: ObsType) -> ObsType:
    #     # read downstream results from optimizer and compute aggregate
    #     raise NotImplementedError

    @abstractmethod
    def aggregate_rewards(self, ctx: list[Context]) -> SupportsFloat:
        return NotImplementedError

    # @abstractmethod
    @override(BaseEnv)
    def action(self, action: ActType) -> list[Mechanism]:
        # analytic path
        if self.inner is None:
            return action

        # 1) Mechanism space path (preferred)
        if self.m_space is not None:
            if isinstance(action, (list, tuple)):
                action = np.asarray(action, dtype=np.float32)

            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    action = action[None, :]  # (d,) -> (1, d)

                # TODO parallelize
                return [self.m_space.decode(x) for x in action]

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
