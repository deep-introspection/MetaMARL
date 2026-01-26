from asyncio import Protocol

from core.mechanism.base import Mechanism


class MechanismSpace(Protocol):
    def sample(self) -> Mechanism:
        pass

    def project(self, theta: Mechanism) -> Mechanism:
        pass

    def clip(self, theta: Mechanism) -> Mechanism:
        pass
