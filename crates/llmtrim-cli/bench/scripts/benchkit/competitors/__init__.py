"""Competitor registry. Register adapters with @register; resolve by name with get()."""
from .base import Competitor

REGISTRY: dict = {}


def register(cls):
    """Class decorator: add a Competitor subclass to the registry under its `name`."""
    REGISTRY[cls.name] = cls
    return cls


def get(name):
    """Instantiate the registered competitor `name`, or raise with the known names."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none registered)"
        raise SystemExit(f"unknown competitor '{name}'. Known competitors: {known}")
    return REGISTRY[name]()


# Import the built-in adapters so importing this package populates the registry.
from . import headroom  # noqa: E402,F401
from . import caveman  # noqa: E402,F401
from . import leanctx  # noqa: E402,F401
from . import entroly  # noqa: E402,F401
from . import rtk  # noqa: E402,F401

__all__ = ["Competitor", "REGISTRY", "register", "get"]
