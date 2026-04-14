"""Reusable driver functions for the advisor-review experiment series.

Imported from short shell-style invocations in the worker session — keeps
the experiment scripts terse and the configurations centralised.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from .runner import (
    make_healthcare_task_cfg,
    make_md_task_cfg,
    make_train_cfg,
    run_one,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ADV_DIR = REPO_ROOT / "results" / "advisor_review"


# ---------------------------------------------------------------------------
# HP sweep driver (Step 2)
# ---------------------------------------------------------------------------

def hp_sweep_one_axis(
    *,
    sweep_name: str,
    axis_name: str,
    axis_values: list,
    methods: list[str],
    seeds: list[int],
    lambdas: list[float],
    steps: int,
    base_task: dict | None = None,
    base_train: dict | None = None,
    overwrite: bool = True,
) -> None:
    """Sweep a single task-config axis, holding everything else fixed."""
    base_task = dict(base_task) if base_task else {}
    base_train = dict(base_train) if base_train else {}
    out_dir = ADV_DIR / "hp_tuning" / "md_knapsack" / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for v in axis_values:
        task_overrides = dict(base_task)
        task_overrides[axis_name] = v
        task_cfg = make_md_task_cfg(**task_overrides)
        train_cfg = make_train_cfg(
            seeds=seeds, lambdas=lambdas, steps=steps,
            **base_train,
        )
        sub = out_dir / f"{axis_name}_{v}"
        run_one(
            out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
            methods=methods, label=f"{sweep_name}_{axis_name}={v}",
            overwrite=overwrite,
        )
    print(f"  TOTAL {sweep_name}: {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Hypothesis 3a: DP vs AP on healthcare
# ---------------------------------------------------------------------------

def healthcare_dp_vs_ap(
    *,
    methods: list[str],
    seeds: list[int],
    lambdas: list[float],
    alphas: list[float],
    n_sample: int = 5000,
    steps: int = 70,
    overwrite: bool = True,
) -> None:
    out_root = ADV_DIR / "hypothesis" / "dp_vs_ap"
    out_root.mkdir(parents=True, exist_ok=True)
    train_cfg = make_train_cfg(
        seeds=seeds, lambdas=lambdas, steps=steps,
        lr=5e-4, hidden_dim=64, n_layers=2, arch="mlp",
        decision_grad_backend="analytic",
    )
    for fairness_type in ("mad", "dp"):
        for alpha in alphas:
            sub = out_root / fairness_type / f"alpha_{alpha}"
            task_cfg = make_healthcare_task_cfg(
                n_sample=n_sample,
                alpha_fair=alpha,
                fairness_type=fairness_type,
                val_fraction=0.2,
                test_fraction=0.5,
            )
            run_one(
                out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
                methods=methods, label=f"dp_vs_ap_{fairness_type}_a{alpha}",
                overwrite=overwrite,
            )


# ---------------------------------------------------------------------------
# Hypothesis 3b: benefit vs cost imbalance grid (MD)
# ---------------------------------------------------------------------------

def md_benefit_cost_grid(
    *,
    chosen_md_cfg: dict,
    train_cfg_overrides: dict,
    methods: list[str],
    seeds: list[int],
    lambdas: list[float],
    benefit_biases: list[float],
    cost_biases: list[float],
    alphas: list[float],
    steps: int,
    cells: list[tuple[float, float]] | None = None,
    overwrite: bool = True,
) -> None:
    out_root = ADV_DIR / "hypothesis" / "benefit_cost_imbalance"
    out_root.mkdir(parents=True, exist_ok=True)
    grid = cells if cells is not None else [(b, c) for b in benefit_biases for c in cost_biases]
    for alpha in alphas:
        for bb, cb in grid:
            sub = out_root / f"alpha{alpha}_bb{bb}_cb{cb}"
            task_cfg_dict = dict(chosen_md_cfg)
            task_cfg_dict["benefit_group_bias"] = bb
            task_cfg_dict["cost_group_bias"] = cb
            task_cfg_dict["alpha_fair"] = alpha
            task_cfg = make_md_task_cfg(**task_cfg_dict)
            train_cfg = make_train_cfg(seeds=seeds, lambdas=lambdas, steps=steps,
                                       **train_cfg_overrides)
            run_one(out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
                    methods=methods, label=f"bb={bb}_cb={cb}_a={alpha}",
                    overwrite=overwrite)


# ---------------------------------------------------------------------------
# Step 4: budget sweep (MD + healthcare)
# ---------------------------------------------------------------------------

def md_budget_sweep(
    *,
    chosen_md_cfg: dict,
    train_cfg_overrides: dict,
    methods: list[str],
    seeds: list[int],
    lambdas: list[float],
    budgets: list[float],
    alpha: float,
    steps: int,
    overwrite: bool = True,
) -> None:
    out_root = ADV_DIR / "budget_sweep" / "md_small"
    out_root.mkdir(parents=True, exist_ok=True)
    for b in budgets:
        sub = out_root / f"budget_{b}"
        task_cfg_dict = dict(chosen_md_cfg)
        task_cfg_dict["budget_tightness"] = b
        task_cfg_dict["alpha_fair"] = alpha
        task_cfg = make_md_task_cfg(**task_cfg_dict)
        train_cfg = make_train_cfg(seeds=seeds, lambdas=lambdas, steps=steps,
                                   **train_cfg_overrides)
        run_one(out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
                methods=methods, label=f"budget={b}", overwrite=overwrite)


def healthcare_budget_sweep(
    *,
    methods: list[str],
    seeds: list[int],
    lambdas: list[float],
    budgets: list[float],
    alpha: float = 2.0,
    n_sample: int = 5000,
    steps: int = 70,
    overwrite: bool = True,
) -> None:
    out_root = ADV_DIR / "budget_sweep" / "healthcare_full"
    out_root.mkdir(parents=True, exist_ok=True)
    train_cfg = make_train_cfg(
        seeds=seeds, lambdas=lambdas, steps=steps,
        lr=5e-4, hidden_dim=64, n_layers=2, arch="mlp",
        decision_grad_backend="analytic",
    )
    for b in budgets:
        sub = out_root / f"budget_{b}"
        task_cfg = make_healthcare_task_cfg(
            n_sample=n_sample, alpha_fair=alpha, budget_rho=b,
        )
        run_one(out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
                methods=methods, label=f"hc_budget={b}", overwrite=overwrite)
