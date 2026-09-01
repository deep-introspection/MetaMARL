"""Regulatory mechanisms and their composition.

``base`` defines the abstract ``Mechanism`` (three transform channels plus an
optimizer-space encoding), ``algorithms`` the concrete mechanisms (quota,
subsidy, threshold penalty, social observation) and ``composition`` the
chained and parallel combinators. This ``__init__`` re-exports nothing; import
from the submodules directly.
"""
