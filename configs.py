"""Shared experiment configuration — method registry, training defaults, and plot styling."""

from __future__ import annotations

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
ALPHA_VALUES = [0.5, 2.0]
N_SAMPLE_SMALL = 500  # quick verification
N_SAMPLE_FULL = 0     # 0 = use all patients (48,784)

# ---------------------------------------------------------------------------
# Method registry
#
# Each method config declares its objective flags explicitly:
#   use_dec:  use decision regret gradient
#   use_pred: use prediction (MSE) gradient
#   use_fair: use fairness gradient
#
# The "method" key maps to the training loop backend:
#   - Core backends: fpto, dfl, fdfl, plg, fplg, saa, var_dro, wass_dro
#   - Advanced backends: ffo, nce, lancer
#
# MOO methods set "mo_method" to override the default gradient combination.
# ---------------------------------------------------------------------------
ALL_METHOD_CONFIGS = {
    # ----- Base methods -----
    "FPTO":   {"method": "fpto",  "use_dec": False, "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "fixed1"},
    "DFL":    {"method": "dfl",   "use_dec": True,  "use_pred": False, "use_fair": False,
               "pred_weight_mode": "zero"},
    "FDFL":   {"method": "fdfl",  "use_dec": True,  "use_pred": False, "use_fair": True,
               "pred_weight_mode": "zero"},
    "PLG":    {"method": "plg",   "use_dec": True,  "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "schedule"},
    "FPLG":   {"method": "fplg",  "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True},

    # ----- Advanced decision gradient methods -----
    "FFO":    {"method": "ffo",    "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "decision_grad_backend": "ffo"},
    "NCE":    {"method": "nce",    "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "decision_grad_backend": "nce"},
    "LANCER": {"method": "lancer", "use_dec": True,  "use_pred": True,  "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "decision_grad_backend": "lancer"},

    # ----- Data-driven optimization baselines -----
    "SAA":    {"method": "saa",  "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1"},
    "VarDRO": {"method": "var_dro", "use_dec": False, "use_pred": True,  "use_fair": False,
               "pred_weight_mode": "fixed1", "dro_epsilon": 0.1},
    "WassDRO": {"method": "wass_dro", "use_dec": False, "use_pred": True,  "use_fair": False,
                "pred_weight_mode": "fixed1", "wdro_epsilon": 0.1},

    # ----- MOO methods (3-objective with MOO handler) -----
    "WS-equal": {
        "method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.333, "pred_loss": 0.333, "pred_fairness": 0.333},
    },
    "WS-dec": {
        "method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.6, "pred_loss": 0.2, "pred_fairness": 0.2},
    },
    "WS-fair": {
        "method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
        "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
        "mo_method": "weighted_sum",
        "mo_weights": {"decision_regret": 0.2, "pred_loss": 0.2, "pred_fairness": 0.6},
    },
    "MGDA":   {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "mgda"},
    "PCGrad": {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "pcgrad"},
    "CAGrad": {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "cagrad"},
    "FAMO":   {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "famo"},
    "PLG-FP": {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "plg_fp"},
    "PLG-PP": {"method": "fplg", "use_dec": True, "use_pred": True, "use_fair": True,
               "pred_weight_mode": "schedule", "continuation": True, "allow_orthogonalization": True,
               "mo_method": "plg_pp"},

    # ----- No-fairness variants (2-objective) -----
    "PTO":        {"method": "fpto", "use_dec": False, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "fixed1",
                   "lambdas": [0.0], "force_lambda_path_all_methods": False},
    "PCGrad-nf":  {"method": "plg", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "schedule", "mo_method": "pcgrad"},
    "MGDA-nf":    {"method": "plg", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "schedule", "mo_method": "mgda"},
    "CAGrad-nf":  {"method": "plg", "use_dec": True, "use_pred": True, "use_fair": False,
                   "pred_weight_mode": "schedule", "mo_method": "cagrad"},
}


def describe_method(name: str, spec: dict) -> str:
    """Return a human-readable description of a method's objectives and handler."""
    objectives = []
    if spec.get("use_dec", False):
        objectives.append("dec")
    if spec.get("use_pred", False):
        objectives.append("pred")
    if spec.get("use_fair", False):
        objectives.append("fair")
    obj_str = "+".join(objectives) if objectives else "none"
    parts = [f"objectives={obj_str}"]
    mo = spec.get("mo_method", "")
    if mo:
        parts.append(f"mo={mo}")
    dgb = spec.get("decision_grad_backend", "")
    if dgb:
        parts.append(f"dec_grad={dgb}")
    return ", ".join(parts)

# ---------------------------------------------------------------------------
# Default training config
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_CFG = {
    "lambdas": [0.0, 0.5],
    "seeds": [11, 22, 33],
    "steps_per_lambda": 70,
    "batch_size": -1,
    "lr": 0.0005,
    "lr_decay": 0.0005,
    "alpha_schedule": {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0},
    "force_lambda_path_all_methods": False,
    "grad_clip_norm": 10000.0,
    "explode_threshold": 1000000.0,
    "fairness_smoothing": 1e-6,
    "log_every": 5,
    "pareto_sweep_mode": True,
    "lambda_train": 0.0,
    "model": {
        "arch": "mlp",
        "hidden_dim": 64,
        "n_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "init_mode": "default",
    },
    "device": DEVICE,
}


# ---------------------------------------------------------------------------
# Task config builder
# ---------------------------------------------------------------------------
def make_task_cfg(
    data_csv: str,
    n_sample: int,
    alpha_fair: float,
    fairness_type: str = "mad",
    val_fraction: float = 0.2,
) -> dict:
    return {
        "name": "medical_resource_allocation",
        "data_csv": data_csv,
        "n_sample": n_sample,
        "data_seed": 42,
        "split_seed": 2,
        "test_fraction": 0.5,
        "val_fraction": val_fraction,
        "alpha_fair": alpha_fair,
        "budget": -1,
        "budget_rho": 0.35,
        "decision_mode": "group",
        "fairness_type": fairness_type,
    }


def compute_full_batch_size(data_csv: str, n_sample: int,
                            test_fraction: float = 0.5,
                            val_fraction: float = 0.2) -> int:
    """Compute the full training set size for use as batch_size.

    Full-batch training is required because the allocation solver needs to see
    all patients simultaneously to respect the global budget constraint.
    """
    df = pd.read_csv(data_csv)
    n_total = n_sample if (n_sample > 0 and n_sample < len(df)) else len(df)
    n_test = int(round(test_fraction * n_total))
    n_remaining = n_total - n_test
    n_val = int(round(val_fraction * n_remaining))
    n_train = n_remaining - n_val
    return n_train


# ---------------------------------------------------------------------------
# Plot styling — shared across all plots
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "FPTO": "#1f77b4", "FDFL": "#ff7f0e", "FFO": "#2ca02c", "LANCER": "#d62728",
    "WS-equal": "#9467bd", "WS-dec": "#8c564b", "WS-fair": "#e377c2", "WS-balanced": "#7f7f7f",
    "MGDA": "#bcbd22", "PCGrad": "#17becf", "PLG-FP": "#aec7e8", "PLG-PP": "#ffbb78",
    "CAGrad": "#98df8a", "FAMO": "#ff9896",
    "DFL": "#c5b0d5", "PLG": "#c49c94", "FPLG": "#f7b6d2", "NCE": "#dbdb8d",
    "SAA": "#e6550d", "VarDRO": "#756bb1", "WassDRO": "#393b79",
    "PTO": "#636363", "PCGrad-nf": "#17becf", "MGDA-nf": "#bcbd22", "CAGrad-nf": "#98df8a",
}

MARKER_MAP = {
    "FPTO": "o", "FDFL": "s", "FFO": "^", "LANCER": "D",
    "WS-equal": "v", "WS-dec": "<", "WS-fair": ">", "WS-balanced": "p",
    "MGDA": "h", "PCGrad": "*", "PLG-FP": "X", "PLG-PP": "P",
    "CAGrad": "d", "FAMO": "H",
    "DFL": "8", "PLG": "+", "FPLG": "x", "NCE": "1",
    "SAA": "D", "VarDRO": "p", "WassDRO": "2",
    "PTO": "o", "PCGrad-nf": "*", "MGDA-nf": "h", "CAGrad-nf": "d",
}
