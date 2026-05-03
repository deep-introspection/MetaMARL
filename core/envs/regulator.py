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
    """Outer-loop environment wrapping the inner-loop MARL optimizer.

    Acts as the single outer-loop "step" from the perspective of the Evolution
    Strategy (ES).  When :meth:`_step` is called with a list of candidate
    mechanisms:

    1. Each mechanism is published to the shared ``World`` actor.
    2. The inner-loop optimizer (``self.inner``) is trained for
       ``train_iters`` iterations under those mechanisms.
    3. The inner-loop optimizer then *evaluates* the trained policy.
    4. New :class:`~core.world.context.EnvStepContext` objects produced during
       evaluation are collected and forwarded to :meth:`aggregate_rewards` to
       produce a scalar fitness signal for the ES.

    If ``optimizer`` is ``None`` the class operates in *analytic* mode: both
    :meth:`action` and :meth:`_step` pass raw values through unchanged,
    allowing subclasses to compute rewards analytically without a nested RL loop.

    Parameters
    ----------
    world : World
        Ray remote actor serving as the shared runtime state container.
    opt_id : OptimizerID or None, optional
        Identifier of the outer-loop (ES) optimizer.
    optimizer : Optimizer or None, optional
        Inner-loop optimizer instance (e.g. RLlib PPO/APPO runner).  When
        ``None`` the analytic override path is used.
    train_iters : int, optional
        Number of inner-loop training iterations per outer-loop step.  Must be
        ``>= 1`` when an optimizer is provided.  Default is ``5``.
    **kwargs
        Forwarded to :class:`~core.envs.base.BaseEnv`.
    """

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
        """Validate constructor arguments.

        Raises
        ------
        ValueError
            If ``train_iters <= 0`` when an optimizer is provided.
        """
        if self.inner is None:
            return  # analytic override mode allowed

        if self.train_iters <= 0:
            raise ValueError("train_iters must be >= 1 when optimizer is provided")

    @override(BaseEnv)
    def _reset(self):
        """Reset the regulator environment to its initial (zero) state.

        The regulator has no meaningful episode state to reinitialise — its
        "state" is the fitness landscape explored by the ES.  Returns a zero
        vector matching the observation space shape, or a scalar ``0.0`` if no
        observation space is defined.

        Returns
        -------
        np.ndarray or float
            Zero-filled initial observation.
        """
        if self.observation_space is None:
            return 0.0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    @override(BaseEnv)
    def _step(
        self, mechanisms: list[Mechanism]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one outer-loop step: publish mechanisms, train inner loop, evaluate.

        For each candidate mechanism in ``mechanisms``:

        1. Publish a :class:`~core.world.context.MechanismContext` with status
           ``published`` to the World.
        2. Run the inner-loop optimizer for ``train_iters`` iterations.
        3. Publish evaluation-phase mechanism contexts and run
           :pymeth:`~core.optimizers.base.Optimizer.evaluate`.
        4. Collect new :class:`~core.world.context.EnvStepContext` objects,
           compute the aggregate fitness via :meth:`aggregate_rewards`, and
           flush consumed contexts from the World.

        Parameters
        ----------
        mechanisms : list[Mechanism]
            Candidate mechanism objects decoded from the ES action vector.

        Returns
        -------
        obs : None
            The regulator does not produce a meaningful next observation.
        reward : SupportsFloat
            Scalar fitness signal aggregated from inner-loop evaluation episodes.
        terminated : bool
            Always ``False`` (the outer loop manages termination).
        truncated : bool
            Always ``False``.
        info : dict
            Empty auxiliary info dict.

        Raises
        ------
        NotImplementedError
            If no inner optimizer is set and :meth:`_step` has not been overridden.
        TypeError
            If ``mechanisms`` is not a ``list[Mechanism]``.
        """
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

        # Train policy for train_iters iterations
        for _ in range(self.train_iters):
            self.inner.run()

        # todo : the total episodes must be same as numer mechanisms
        # reset mechanism contexts
        for _ in range(self.inner.eval_episodes):
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
        consumed_ids = [ctx.id for ctx in new_ctxs]

        # Evaluation metrics are defined by user and provided as a callable.
        # metrics = user_eval_fn(contexts)

        reward = self.aggregate_rewards(new_ctxs)

        # flush consumed contexts

        ray.get(self.world.flush_ctx.remote(consumed_ids))
        ray.get(self.world.flush.remote(job=MechanismStatus.eval))

        return None, reward, False, False, {}

    # @abstractmethod
    # @override(BaseEnv)
    # def observation(self, observation: ObsType) -> ObsType:
    #     # read downstream results from optimizer and compute aggregate
    #     raise NotImplementedError

    @abstractmethod
    def aggregate_rewards(self, ctx: list[Context]) -> SupportsFloat:
        """Aggregate inner-loop evaluation contexts into an outer-loop fitness score.

        Called after each evaluation run.  Concrete implementations should
        extract per-episode metrics from the provided contexts (e.g. mean reward,
        sustainability rate, economic welfare) and return a single scalar that
        the ES uses to rank mechanism candidates.

        Parameters
        ----------
        ctx : list[Context]
            Evaluation-phase context objects collected from the World since the
            last flush.  Each context's payload is an
            :class:`~core.world.context.EnvStepContext`.

        Returns
        -------
        SupportsFloat
            Scalar fitness value (higher is better for maximisation problems).
        """
        return NotImplementedError

    # @abstractmethod
    @override(BaseEnv)
    def action(self, action: ActType) -> list[Mechanism]:
        """Decode a raw ES action into a list of typed Mechanism objects.

        Handles multiple input formats:

        - ``np.ndarray`` of shape ``(d,)`` (single mechanism) — reshaped to
          ``(1, d)`` and decoded via :attr:`m_space`.
        - ``np.ndarray`` of shape ``(k, d)`` (population batch) — each row is
          decoded independently.
        - ``torch.Tensor`` — converted to NumPy and re-dispatched.
        - ``list`` / ``tuple`` — cast to ``np.ndarray`` and re-dispatched.
        - Already-built :class:`~core.mechanism.base.Mechanism` instance — wrapped
          in a single-element list (analytic test path).
        - If no :attr:`m_space` is defined, raw vectors are wrapped in
          :class:`~core.mechanism.base.VectorMechanism` objects.
        - If ``self.inner is None`` (analytic mode) the action is returned
          unchanged.

        Parameters
        ----------
        action : ActType
            Raw action from the ES optimizer.  Typically an ``np.ndarray`` with
            one row per mechanism candidate.

        Returns
        -------
        list[Mechanism]
            Decoded mechanism objects ready to be published to the World.

        Raises
        ------
        TypeError
            If the action type is not supported.
        """
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
