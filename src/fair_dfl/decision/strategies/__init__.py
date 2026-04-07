"""Decision gradient strategies."""

from .analytic import AnalyticStrategy
from .finite_diff import FiniteDiffStrategy
from .spsa import SPSAStrategy

__all__ = [
    "AnalyticStrategy",
    "FiniteDiffStrategy",
    "SPSAStrategy",
]

# Lazy imports for optional strategies (require additional dependencies)


def get_fold_opt_strategy(**kwargs):
    from .fold_opt import FoldOptStrategy
    return FoldOptStrategy(**kwargs)


def get_nce_strategy(**kwargs):
    from .nce import NCEStrategy
    return NCEStrategy(**kwargs)


def get_lancer_strategy(**kwargs):
    from .lancer import LancerStrategy
    return LancerStrategy(**kwargs)
