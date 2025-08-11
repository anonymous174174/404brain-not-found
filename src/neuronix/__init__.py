__all__ = [
    "custom_tensor",
    "losses",
    "module",
    "autograd_graph",
    "optimizers",
    "lr_scheduler",
    "__version__"
]

def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod  # cache so future lookups are fast
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import (
        custom_tensor,
        autograd_graph,
        module,
        optimizers,
        losses,
        lr_scheduler
    )
__version__ = "0.0.1"