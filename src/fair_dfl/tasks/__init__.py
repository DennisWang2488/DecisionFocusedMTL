"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from importlib import import_module

__all__ = [
    "PortfolioQPTask",
    "PortfolioQPSimplexTask",
    "PortfolioQPMultiConstraintTask",
    "PyEPONonlinearKnapsackTask",
    "PyEPOPortfolioQPTask",
    "PyEPOPortfolioQPSimplexTask",
    "ResourceAllocationTask",
    "MedicalResourceAllocationTask",
]

_EXPORT_MAP = {
    "PortfolioQPTask": ".portfolio_qp",
    "PortfolioQPSimplexTask": ".portfolio_qp_simplex",
    "PortfolioQPMultiConstraintTask": ".portfolio_qp_multi_constraint",
    "PyEPONonlinearKnapsackTask": ".pyepo_synthetic",
    "PyEPOPortfolioQPTask": ".pyepo_synthetic",
    "PyEPOPortfolioQPSimplexTask": ".pyepo_synthetic",
    "ResourceAllocationTask": ".resource_allocation",
    "MedicalResourceAllocationTask": ".medical_resource_allocation",
}
_OPTIONAL_EXPORTS = {
    "PyEPONonlinearKnapsackTask",
    "PyEPOPortfolioQPTask",
    "PyEPOPortfolioQPSimplexTask",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name, __name__)
    except ImportError:
        if name in _OPTIONAL_EXPORTS:
            return None
        raise
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
