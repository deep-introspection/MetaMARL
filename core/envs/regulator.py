"""Outer-level environment driving the inner optimizer.

``RegulatorEnv`` is the environment seen by the outer optimizer (Evolution
Strategies). One outer step evaluates a population of candidate mechanisms:

1. ``action(population)`` decodes each optimizer vector into a ``Mechanism``
   through the env's mechanism template;
2. ``_step(mechanisms)`` publishes one ``MechanismContext`` per (candidate,
   seed) to the ``World``, resets the inner policy, trains it for
   ``train_iters`` inner iterations, optionally evaluates it on ``eval_seeds``,
   then aggregates the inner ``EnvStepContext`` records into one fitness per
   candidate with :meth:`aggregate_rewards`;
3. consumed contexts are flushed from the World.

Without an inner optimizer the env runs in *analytic* mode: subclasses override
``_step`` with a closed-form fitness, which is how the ES is unit-tested.
"""

from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
import torch
from gymnasium.core import ActType, ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.optimizers.base import Optimizer
from core.world.context import (
    Context,
    MechanismContext,
    MechanismStatus,
)


class RegulatorEnv(BaseEnv):
    """Outer-loop environment: candidate mechanisms in, one fitness per candidate out.

    Parameters
    ----------
    optimizer : Optimizer, optional
        Inner optimizer (e.g. ``RayOptimizer`` wrapping APPO). ``None`` selects
        the analytic mode.
    train_iters : int
        Inner training iterations per outer step (``>= 1`` with an optimizer).
    seeds : list[int], optional
        Policy seeds; one ``MechanismContext`` is published per (candidate, seed).
    eval_seeds : list[int], optional
        If given, ``inner.evaluate()`` runs after training.
    **kwargs
        Forwarded to :class:`BaseEnv` (``world``, ``mechanism``, ``horizon``, ...).
    """

    def __init__(
        self,
        *,
        optimizer: Optional[Optimizer] = None,
        train_iters: int = 5,
        seeds: Optional[list[int]] = None,
        eval_seeds: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner: Optimizer = optimizer
        self.train_iters: int = train_iters
        self.seeds: list[int] = seeds or []
        self.eval_seeds: Optional[list[int]] = eval_seeds or None

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

        # Reset policy weights for fresh equilibrium search each ES iteration
        if hasattr(self.inner, "reset"):
            self.inner.reset()

        # TODO parallelize: vectorize the inner env across candidates
        for idx, m in enumerate(mechanisms):
            for seed in self.seeds:
                self._publish(
                    MechanismContext(
                        index=idx,
                        seed=seed,
                        status=MechanismStatus.published,
                        env_id=None,
                        mechanism=m,
                        metrics=None,
                    )
                )

        # TODO : why eval gets repeated ?
        # Train policy for train_iters iterations
        for _ in range(self.train_iters):
            ctx_registry = ray.get(self.world.get_ctx_registry.remote())
            ray.get(self.world.flush_ctx.remote(ctx_registry.keys()))
            ray.get(self.world.flush.remote(status=MechanismStatus.eval))
            self.inner.run()

        # TODO check if eval mechanisms published. If parallel and sequential eval both turned on
        # will be a problem
        if self.eval_seeds:
            # TODO flush all remote mechanisms and env_step ctx.
            # TODO initializing the envs with the seeds from eval_seeds
            # TODO flush all remote mechanisms and env_step ctx.
            self.inner.evaluate()

        ctx_registry = ray.get(self.world.get_ctx_registry.remote())

        # Aggregate rewards
        reward = self.aggregate_rewards(ctx_registry.values())

        # flush consumed contexts
        ray.get(self.world.flush_ctx.remote(ctx_registry.keys()))
        ray.get(self.world.flush.remote(status=MechanismStatus.eval))

        return None, reward, False, False, {}

    @abstractmethod
    def aggregate_rewards(self, ctx: list[Context]) -> SupportsFloat:
        """Reduce the World contexts produced during the inner loop to fitness values.

        Parameters
        ----------
        ctx : list[Context]
            Every context currently held by the World (``EnvStepContext`` and
            ``MechanismContext`` payloads).

        Returns
        -------
        SupportsFloat or list[float]
            One fitness per candidate index, in candidate order.
        """
        raise NotImplementedError

    @override(BaseEnv)
    def action(self, action: ActType) -> list[Mechanism]:
        """Decode optimizer vectors into mechanisms.

        Accepts a ``(d,)`` or ``(n, d)`` array-like (list, ndarray or torch
        tensor) and returns ``n`` mechanisms decoded through the template.
        Already-built mechanisms are passed through. In analytic mode the raw
        action is returned unchanged.
        """
        # analytic path
        if self.inner is None:
            return action

        # 1) Mechanism template path (preferred)
        if self.mechanism_template is not None:
            if isinstance(action, (list, tuple)):
                action = np.asarray(action, dtype=np.float32)

            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    action = action[None, :]  # (d,) -> (1, d)

                # TODO parallelize
                return [self.mechanism_template.decode(x) for x in action]

            if torch.is_tensor(action):
                return self.action(action.detach().cpu().numpy())

            raise TypeError(f"Unsupported action type: {type(action)}")

        # 2) No mechanism template: only already-built mechanisms are accepted
        if isinstance(action, Mechanism):
            return [action]
        if isinstance(action, (list, tuple)) and all(
            isinstance(m, Mechanism) for m in action
        ):
            return list(action)

        raise TypeError(
            "RegulatorEnv needs a mechanism template to decode optimizer vectors; "
            f"got {type(action)} with no template."
        )
