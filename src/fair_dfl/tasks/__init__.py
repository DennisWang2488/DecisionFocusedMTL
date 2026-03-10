"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from .portfolio_qp import PortfolioQPTask
from .portfolio_qp_simplex import PortfolioQPSimplexTask
from .portfolio_qp_multi_constraint import PortfolioQPMultiConstraintTask
try:
    from .pyepo_synthetic import PyEPONonlinearKnapsackTask, PyEPOPortfolioQPTask, PyEPOPortfolioQPSimplexTask
except ImportError:
    PyEPONonlinearKnapsackTask = None
    PyEPOPortfolioQPTask = None
    PyEPOPortfolioQPSimplexTask = None
from .resource_allocation import ResourceAllocationTask
from .medical_resource_allocation import MedicalResourceAllocationTask

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
