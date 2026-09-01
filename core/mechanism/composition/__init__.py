"""Combinators building a mechanism out of several children.

``chained_mechanism`` applies the children one after the other on every
channel; ``parallel_mechanism`` applies them to the same input and merges the
outputs. This ``__init__`` re-exports nothing; import from the submodules
directly.
"""
