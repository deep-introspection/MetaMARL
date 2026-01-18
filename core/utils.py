import uuid
from typing import AbstractSet, Any

# TODO restric Any type annotation.


def generate_uuid(registry: AbstractSet[Any]) -> str:
    """
    Generate a UUID that is guaranteed not to exist in the given registry.
    """
    while True:
        id = str(uuid.uuid4())
        if id not in registry:
            return id
