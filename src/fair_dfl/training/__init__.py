"""Unified training package for all DFL methods."""

from .eval import eval_split, eval_split_medical, evaluate_model
from .loop import run_method_seed, run_methods, train_single_stage
from .method_spec import MethodSpec, resolve_method_spec, resolve_method_backend

__all__ = [
    "eval_split",
    "eval_split_medical",
    "evaluate_model",
    "run_method_seed",
    "run_methods",
    "train_single_stage",
    "MethodSpec",
    "resolve_method_spec",
    "resolve_method_backend",
]
