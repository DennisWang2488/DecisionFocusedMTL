"""Algorithm components — gradient handlers, utilities, and legacy trainers.

Active modules (used by training/loop.py):
  - mo_handler.py   — MOO gradient handlers (PCGrad, MGDA, CAGrad, FAMO, etc.)
  - torch_utils.py  — gradient manipulation utilities

Legacy trainer wrappers (used by runner.py's old code path):
  - core_methods.py      — 5-method trainer (fpto, dfl, fdfl, plg, fplg)
  - advanced_methods.py  — FFO/NCE/LANCER trainer

The unified training pipeline in training/loop.py supersedes the legacy trainers.
"""

from .core_methods import METHOD_SPECS, METHOD_ALIASES, REVERSE_ALIASES, run_core_methods
from .advanced_methods import ADVANCED_METHODS, run_advanced_methods

__all__ = [
    "METHOD_SPECS",
    "METHOD_ALIASES",
    "REVERSE_ALIASES",
    "ADVANCED_METHODS",
    "run_core_methods",
    "run_advanced_methods",
]
