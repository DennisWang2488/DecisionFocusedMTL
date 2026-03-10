"""Algorithm entry points for the 8 active methods.

Core methods (`fpto`, `dfl`, `fdfl`, `plg`, `fplg`) run via `core_methods`,
while advanced methods (`ffo`, `nce`, `lancer`) run via `advanced_methods`.
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
