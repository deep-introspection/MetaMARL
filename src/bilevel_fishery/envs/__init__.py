"""Gymnasium environments wrapping the bilevel-fishery domain model.

For now this subpackage exposes only the single-agent :class:`FisheryEnv`.
Multi-agent and regulated variants will be added in subsequent bricks.
"""

from bilevel_fishery.envs.fishery_env import FisheryEnv

__all__ = ["FisheryEnv"]
