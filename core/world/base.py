from __future__ import annotations

from typing import TYPE_CHECKING, KeysView, Optional

import ray
from core.reporting.wandb import WandbReporter
from core.types import ContextID, OptimizerID
from core.utils import generate_uuid
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


@ray.remote
class World:
    """
    Shared runtime container for optimizer-produced contexts.

    The World acts as the single source of truth for:
    - Context lifecycles (creation, update, removal)
    - Association between optimizers and their contexts
    - Enforcing global constraints (e.g. singleton contexts)

    The World does NOT own optimizers or environments.
    It only tracks identifiers and context payloads.
    """

    # TODO the reporting type annotation to add
    def __init__(self, reporting: WandbReporter = None):
        """Initialise the World actor with empty registries.

        Parameters
        ----------
        reporting : WandbReporter, optional
            Optional Weights & Biases reporter actor used for live experiment
            logging.  ``None`` disables all external reporting.
        """
        # Maps optimizer IDs to the list of context IDs they own
        # TODO replace with registry
        self._opt_ctx_map: dict[OptimizerID, list[ContextID]] = {}

        # Maps context IDs to Context objects
        self._contexts: dict[ContextID, Context] = {}

        # Mechanism registry
        self._mechanism_registry: dict[int, MechanismContext] = {}

        # TODO wrap this into generic logger class - for now use W&B
        self.reporting: WandbReporter = reporting

    def __deepcopy__(self, memo):
        """Return self to prevent Ray actors from being deep-copied.

        Ray remote actors must not be serialised by value.  Returning ``self``
        ensures that any inadvertent ``copy.deepcopy`` call leaves the actor
        handle intact.
        """
        return self

    def __copy__(self):
        """Return self to prevent Ray actors from being shallow-copied."""
        return self

    # Accessors
    def get_ctx_registry(self) -> dict[ContextID, Context]:
        """Return the full context registry.

        Returns
        -------
        dict[ContextID, Context]
            Mapping of all context IDs to their :class:`~core.world.context.Context` objects.
        """
        return self._contexts

    def get_mechanism_registry(self) -> dict[int, MechanismContext]:
        """Return the mechanism registry.

        Returns
        -------
        dict[int, MechanismContext]
            Mapping of context IDs to active
            :class:`~core.world.context.MechanismContext` objects.
        """
        return self._mechanism_registry

    def get_opt_registry(self) -> KeysView[OptimizerID]:
        """Return the set of registered optimizer IDs.

        Returns
        -------
        KeysView[OptimizerID]
            Live view of all optimizer IDs known to the World.
        """
        return self._opt_ctx_map.keys()

    def get_context(self, ctx_id: ContextID) -> Context | None:
        """Access a context stored in world with an ID"""
        return self._contexts.get(ctx_id, None)

    def get_opt_ctx_ids(self, opt_id: OptimizerID) -> list[ContextID]:
        """
        Return all context IDs registered under a given optimizer.
        """
        return list(self._opt_ctx_map.get(opt_id, []))

    def get_ctx_ids(self) -> set[ContextID]:
        """
        Return all context IDs registered in the world.
        """
        return set(self._contexts.keys())

    def get_opt_ids(self) -> set[OptimizerID]:
        """
        Return all optimizer IDs known to the world.
        """
        return set(self._opt_ctx_map.keys())
    
        # ADDED: helpers for reduced env plotting
    def get_env_step_contexts(
        self,
        opt_id: Optional[OptimizerID] = None,
    ) -> list[Context]:
        """
        Return all EnvStepContext objects for a given optimizer.
        If opt_id is None, return all EnvStepContext objects in the world.
        Order is preserved according to insertion order.
        """
        if opt_id is None:
            ctxs = list(self._contexts.values())
        else:
            ctx_ids = self._opt_ctx_map.get(opt_id, [])
            ctxs = [self._contexts[cid] for cid in ctx_ids if cid in self._contexts]

        return [
            ctx for ctx in ctxs
            if ctx is not None and isinstance(ctx.payload, EnvStepContext)
        ]

    # ADDED: helpers for reduced env plotting
    def get_latest_env_step_contexts(
        self,
        opt_id: Optional[OptimizerID] = None,
    ) -> list[Context]:
        """
        Return only the latest contiguous env-step episode for an optimizer.

        We assume env-step contexts are appended in order and that step resets
        to 0 at the beginning of a new episode. We walk backward from the most
        recent EnvStepContext until we hit step == 0.
        """
        env_ctxs = self.get_env_step_contexts(opt_id=opt_id)
        if not env_ctxs:
            return []

        latest: list[Context] = []
        for ctx in reversed(env_ctxs):
            latest.append(ctx)
            if int(ctx.step) == 0:
                break

        latest.reverse()
        return latest

    def get_mechanism(self) -> MechanismContext:
        """Claim the first published mechanism (blocking).

        Iterates over the mechanism registry and atomically transitions the
        first mechanism with status ``published`` to status ``assigned``.

        Returns
        -------
        MechanismContext
            The claimed mechanism context.

        Raises
        ------
        RuntimeError
            If no mechanism with status ``published`` is currently available.
        """
        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned
                return m_ctx

        raise RuntimeError("no available mechanisms to train")

    def try_get_mechanism(self) -> MechanismContext | None:
        """Try to get a published mechanism, return None if none available."""
        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned
                return m_ctx
        return None

    # TODO fix this function. now the primary key is ctx_id
    def get_mechanism_by_index(self, index: int) -> MechanismContext:
        """Retrieve a mechanism context by its integer index.

        Parameters
        ----------
        index : int
            The mechanism index as stored in the registry (currently the
            context ID key).

        Returns
        -------
        MechanismContext
            The corresponding mechanism context.

        Raises
        ------
        KeyError
            If no mechanism with the given index exists.
        """
        return self._mechanism_registry[index]

    def _validate_ctx_schema_exists(self, schema: type[ContextSchema]) -> None:
        """
        Ensure a singleton ContextSchema is not already present in the world.
        """
        for ctx in self._contexts.values():
            if type(ctx.payload) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in world."
                )

    # Mutators
    def append_context(self, ctx: Context, *, singleton: bool = False):
        """Append a new context to the World, assigning it a unique ID.

        Generates a UUID for the context, stores it in the context registry,
        updates the optimizer-to-context mapping, and (if the payload is a
        :class:`~core.world.context.MechanismContext`) also registers it in the
        mechanism registry.

        Parameters
        ----------
        ctx : Context
            The context to register.  Its ``id`` field must be ``None`` or a
            value not already present in the registry.
        singleton : bool, optional
            If ``True``, raises :exc:`ValueError` if a context with the same
            payload schema type already exists.

        Returns
        -------
        ContextID
            The newly assigned context ID.

        Raises
        ------
        ValueError
            If ``ctx.id`` is already registered, or if ``singleton=True`` and
            a context with the same schema type already exists.
        """
        # Enforce singleton schemas if requested
        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None and ctx.id in self._contexts:
            raise ValueError(f"ContextID '{ctx.id}' already exists")

        ctx.id = generate_uuid(self._contexts)
        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            self._mechanism_registry[ctx.id] = ctx.payload

        # Track optimizer → context mapping
        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)
            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        # if env-step-context call reporter actor
        # if self.reporting is not None and isinstance(ctx.payload, EnvStepContext):
        #     self.reporting.plot_env_step.remote(
        #         ctx=ctx,
        #         obs_keys_skip=(
        #             "fixed_quota",
        #             "prop_quota",
        #             "min_stock",
        #             "target_stock",
        #             "fine_amount",
        #             "risk_penalty_scale",
        #             "risk_penalty_power",
        #         ),
        #     )

        return ctx.id

    def register_optimizer(self, opt: Optimizer) -> OptimizerID:
        """
        Register a new optimizer ID in the world.

        This initializes an empty context set for the optimizer.
        """
        return self._set_new_opt_id(opt_id=opt.opt_id)

    def _set_new_opt_id(self, opt_id: OptimizerID) -> OptimizerID:
        """Register a new optimizer ID and initialise its context list.

        If ``opt_id`` is ``None`` a UUID is generated automatically.

        Parameters
        ----------
        opt_id : OptimizerID or None
            Desired optimizer ID.  Pass ``None`` to auto-generate.

        Returns
        -------
        OptimizerID
            The registered (possibly auto-generated) optimizer ID.
        """
        if opt_id is None:
            opt_id = generate_uuid(registry=self._opt_ctx_map.keys())

        if opt_id not in self._opt_ctx_map:
            self._opt_ctx_map[opt_id] = []

        return opt_id

    def set_new_context(self, ctx: Context, singleton: bool = False) -> ContextID:
        """Register a new context in the World with explicit ID assignment.

        Similar to :meth:`append_context` but allows pre-assigned context IDs.
        If ``ctx.id`` is ``None`` a UUID is generated; otherwise the provided
        value is used as-is (and must not already be registered).

        Parameters
        ----------
        ctx : Context
            Context object containing the optimizer ID and schema payload.
        singleton : bool, optional
            If ``True``, raises :exc:`ValueError` if a context with the same
            payload schema type already exists.

        Returns
        -------
        ContextID
            The (possibly auto-generated) context ID.

        Raises
        ------
        ValueError
            If ``ctx.id`` is already registered, if ``singleton=True`` and a
            duplicate schema type exists, or if a :class:`~core.world.context.MechanismContext`
            payload is missing ``env_id``.
        """

        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None:
            if ctx.id in self._contexts:
                raise ValueError(f"ContextID '{ctx.id}' already exists")
        else:
            ctx.id = generate_uuid(registry=self._contexts.keys())

        self._contexts[ctx.id] = ctx

        # Track latest mechanism globally
        # Track per-env mechanism
        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")
            self._mechanism_registry[ctx.id] = ctx.payload

        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)

            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        return ctx.id

    def update_context(self, ctx: Context) -> None:
        """Replace the payload of an existing context in-place.

        Parameters
        ----------
        ctx : Context
            Updated context.  ``ctx.id`` must already be registered.

        Raises
        ------
        KeyError
            If ``ctx.id`` is not found in the context registry.
        ValueError
            If the payload is a :class:`~core.world.context.MechanismContext`
            without a populated ``env_id``.
        """
        if ctx.id not in self._contexts:
            raise KeyError(f"Context {ctx.id} not registered")

        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")
            self._mechanism_registry[ctx.id] = ctx.payload

    def remove_context(self, ctx: Context) -> None:
        """Remove a context from the World and clean up optimizer mappings.

        If the optimizer no longer owns any contexts after removal its entry is
        deleted from the optimizer-to-context map.

        Parameters
        ----------
        ctx : Context
            The context to remove.  If ``ctx.id`` is not registered, the call
            is a no-op.
        """
        if ctx.id in self._contexts:
            self._contexts.pop(ctx.id)

        if ctx.opt_id in self._opt_ctx_map:
            lst = self._opt_ctx_map[ctx.opt_id]
            if ctx.id in lst:
                lst.remove(ctx.id)

            if not lst:
                del self._opt_ctx_map[ctx.opt_id]

    # TODO fix this function. now the primary key is ctx_id
    def flush(self, job: Optional[MechanismStatus] = None) -> None:
        """Remove mechanism contexts from the registry, optionally filtered by job type.

        Parameters
        ----------
        job : MechanismStatus or None, optional
            When provided, only mechanism contexts whose ``job`` field matches
            this status are deleted.  When ``None``, all mechanism contexts are
            removed.
        """
        to_delete = []

        for ctx_id, m_ctx in self._mechanism_registry.items():
            if job is not None and m_ctx.job != job:
                continue
            to_delete.append(ctx_id)

        for ctx_id in to_delete:
            del self._mechanism_registry[ctx_id]

    def flush_ctx(self, ctx_ids: list[ContextID]):
        """Remove a specific set of context IDs from the context registry.

        Missing IDs are silently ignored.  Used after evaluation to discard
        consumed :class:`~core.world.context.EnvStepContext` entries.

        Parameters
        ----------
        ctx_ids : list[ContextID]
            List of context IDs to delete from the registry.
        """
        for cid in ctx_ids:
            self._contexts.pop(cid, None)
