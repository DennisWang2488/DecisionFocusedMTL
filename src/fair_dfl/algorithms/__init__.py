"""Algorithm components — gradient handlers, utilities, and legacy trainers.

Active modules (used by training/loop.py):
  - mo_handler.py   — MOO gradient handlers (PCGrad, MGDA, CAGrad, FAMO, etc.)
  - torch_utils.py  — gradient manipulation utilities

Legacy trainer wrappers (used by runner.py's old code path):
  - core_methods.py      — 5-method trainer (fpto, dfl, fdfl, plg, fplg)
  - advanced_methods.py  — FFO/NCE/LANCER trainer

The unified training pipeline in training/loop.py supersedes the legacy trainers.
"""

from importlib import import_module

__all__ = [
    "METHOD_SPECS",
    "METHOD_ALIASES",
    "REVERSE_ALIASES",
    "ADVANCED_METHODS",
    "run_core_methods",
    "run_advanced_methods",
]

_EXPORT_MAP = {
    "METHOD_SPECS": ".core_methods",
    "METHOD_ALIASES": ".core_methods",
    "REVERSE_ALIASES": ".core_methods",
    "run_core_methods": ".core_methods",
    "ADVANCED_METHODS": ".advanced_methods",
    "run_advanced_methods": ".advanced_methods",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
