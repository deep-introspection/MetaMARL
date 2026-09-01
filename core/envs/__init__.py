"""Environments of both optimisation levels.

``base`` defines the single-agent ``BaseEnv`` that publishes every step to the
``World``, ``regulator`` builds the outer ``RegulatorEnv`` on it,
``marl_regulated`` provides the inner ``MultiAgentRegulatedEnv`` that concrete
benchmarks subclass, ``hooks`` holds the decorators those benchmarks use to
declare their dynamics, and ``schema`` the metric schemas of an inner episode.
This ``__init__`` re-exports nothing; import from the submodules directly.
"""
