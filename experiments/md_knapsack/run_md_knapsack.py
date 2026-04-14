#!/usr/bin/env python3
"""Run synthetic multi-dimensional knapsack experiments on the current codebase."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402
from fair_dfl.runner import run_experiment_unified  # noqa: E402


DEFAULT_TASK_CFG = {
    "name": "md_knapsack",
    # --- Population (per-individual layout) ---
    "n_samples_train": 120,
    "n_samples_val": 24,
    "n_samples_test": 48,
    "n_features": 5,
    "n_resources": 2,
    # --- Objective ---
    "scenario": "alpha_fair",
    "alpha_fair": 2.0,
    "poly_degree": 2,
    # --- Signal-to-noise (decoupled from group bias) ---
    "snr": 5.0,
    # --- Group imbalance knobs (separate for benefit vs cost) ---
    "benefit_group_bias": 0.3,
    "benefit_noise_ratio": 1.0,
    "cost_group_bias": 0.0,
    "cost_noise_ratio": 1.0,
    "cost_mean": 1.0,
    "cost_std": 0.2,
    # --- Constraint ---
    "budget_tightness": 0.5,
    # --- Misc ---
    "fairness_type": "mad",
    "data_seed": 42,
    "group_ratio": 0.5,
    "decision_mode": "group",
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
UNSUPPORTED_METHOD_BACKENDS: set[str] = set()
SUPPORTED_METHOD_CONFIGS = {
    name: spec
    for name, spec in ALL_METHOD_CONFIGS.items()
    if str(spec.get("method", "")).strip().lower() not in UNSUPPORTED_METHOD_BACKENDS
}


def _resolve_methods(names: list[str]) -> dict[str, dict]:
    lookup = {name.lower(): (name, spec) for name, spec in SUPPORTED_METHOD_CONFIGS.items()}
    resolved: dict[str, dict] = {}
    for name in names:
        key = name.strip().lower()
        if key not in lookup:
            supported = ", ".join(sorted(SUPPORTED_METHOD_CONFIGS))
            raise ValueError(f"Unknown or unsupported method {name!r}. Supported methods: {supported}")
        canonical, spec = lookup[key]
        resolved[canonical] = copy.deepcopy(spec)
    return resolved


def _make_train_cfg(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    cfg["seeds"] = list(args.seeds)
    cfg["lambdas"] = list(args.lambdas)
    cfg["steps_per_lambda"] = int(args.steps)
    cfg["lr"] = float(args.lr)
    cfg["batch_size"] = -1
    cfg["decision_grad_backend"] = str(args.decision_grad_backend)
    cfg["decision_grad_spsa_eps"] = float(args.spsa_eps)
    cfg["decision_grad_spsa_n_dirs"] = int(args.spsa_n_dirs)
    cfg["device"] = str(args.device)
    cfg["log_every"] = int(args.log_every)
    cfg["model"] = {
        "arch": args.model_arch,
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "init_mode": "default",
    }
    return cfg


def _make_task_cfg(args: argparse.Namespace, alpha: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_TASK_CFG)
    cfg["scenario"] = str(args.scenario)
    cfg["alpha_fair"] = float(alpha)
    cfg["n_resources"] = int(args.n_resources)
    cfg["n_features"] = int(args.n_features)
    cfg["poly_degree"] = int(args.poly_degree)
    cfg["snr"] = float(args.snr)
    cfg["benefit_group_bias"] = float(args.benefit_group_bias)
    cfg["benefit_noise_ratio"] = float(args.benefit_noise_ratio)
    cfg["cost_group_bias"] = float(args.cost_group_bias)
    cfg["cost_noise_ratio"] = float(args.cost_noise_ratio)
    cfg["cost_mean"] = float(args.cost_mean)
    cfg["cost_std"] = float(args.cost_std)
    cfg["budget_tightness"] = float(args.budget_tightness)
    cfg["fairness_type"] = str(args.fairness_type)
    cfg["n_samples_train"] = int(args.n_train)
    cfg["n_samples_val"] = int(args.n_val)
    cfg["n_samples_test"] = int(args.n_test)
    cfg["data_seed"] = int(args.data_seed)
    cfg["group_ratio"] = float(args.group_ratio)
    return cfg


def _run_single(
    task_cfg: dict,
    train_cfg: dict,
    method_name: str,
    method_spec: dict,
    alpha: float,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = {"task": task_cfg, "training": copy.deepcopy(train_cfg)}
    for key, value in method_spec.items():
        if key not in {
            "method",
            "use_dec",
            "use_pred",
            "use_fair",
            "pred_weight_mode",
            "continuation",
            "allow_orthogonalization",
        }:
            cfg["training"][key] = value

    t0 = perf_counter()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs={method_name: method_spec})
    elapsed = perf_counter() - t0

    if not stage_df.empty:
        stage_df["alpha_fair"] = alpha
        stage_df["config_name"] = method_name
        stage_df["scenario"] = task_cfg["scenario"]
        stage_df["poly_degree"] = task_cfg["poly_degree"]
        stage_df["benefit_group_bias"] = task_cfg["benefit_group_bias"]
        stage_df["cost_group_bias"] = task_cfg["cost_group_bias"]
        stage_df["snr"] = task_cfg["snr"]
        stage_df["n_resources"] = task_cfg["n_resources"]
        stage_df["dec_grad_backend"] = cfg["training"].get("decision_grad_backend", "finite_diff")
        if tag:
            stage_df["tag"] = tag

    if not iter_df.empty:
        iter_df["alpha_fair"] = alpha
        iter_df["config_name"] = method_name
        if tag:
            iter_df["tag"] = tag

    print(f"  {method_name} alpha={alpha}: {len(stage_df)} stages, {elapsed:.1f}s")
    return stage_df, iter_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dimensional knapsack synthetic experiment")
    parser.add_argument("--scenario", default="alpha_fair", choices=["lp", "alpha_fair"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[2.0])
    parser.add_argument("--n-resources", type=int, default=2,
                        help="Number of resource types per individual (default: 2 — 'a' and 'b').")
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--snr", type=float, default=5.0,
                        help="Signal-to-noise ratio for benefit (decoupled from group bias).")
    parser.add_argument("--benefit-group-bias", type=float, default=0.3,
                        help="Per-resource additive shift to the benefit signal (+ for g0, - for g1).")
    parser.add_argument("--benefit-noise-ratio", type=float, default=1.0,
                        help="Per-group noise scaling for benefit (group-1 std relative to group-0).")
    parser.add_argument("--cost-group-bias", type=float, default=0.0,
                        help="Per-resource additive shift to the cost (+ for g0, - for g1). Affects decisions only.")
    parser.add_argument("--cost-noise-ratio", type=float, default=1.0)
    parser.add_argument("--cost-mean", type=float, default=1.0)
    parser.add_argument("--cost-std", type=float, default=0.2)
    parser.add_argument("--budget-tightness", type=float, default=0.5)
    parser.add_argument("--group-ratio", type=float, default=0.5,
                        help="Fraction of individuals in group 0 (default: 0.5 = equal groups)")
    parser.add_argument("--fairness-type", default="mad",
                        choices=["mad", "gap", "atkinson", "dp", "demographic_parity"])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-val", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=48)
    parser.add_argument("--methods", nargs="+", default=["FPTO", "FDFL", "PCGrad"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--decision-grad-backend",
        default="spsa",
        choices=["spsa", "finite_diff"],
        help="Decision-gradient backend. SPSA (default) is ~20x faster than "
             "finite_diff on the per-individual MD task at the cost of slightly "
             "noisier gradients; finite_diff still available for full-FD checks.",
    )
    parser.add_argument("--spsa-n-dirs", type=int, default=8,
                        help="Number of Rademacher directions averaged per SPSA step. "
                             "Higher = lower variance, linearly more solver calls.")
    parser.add_argument("--spsa-eps", type=float, default=5e-3,
                        help="SPSA perturbation magnitude.")
    parser.add_argument(
        "--model-arch",
        default="mlp",
        choices=["linear", "mlp", "resnet_tabular", "ft_transformer"],
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()

    if args.list_methods:
        for name in sorted(SUPPORTED_METHOD_CONFIGS):
            print(name)
        return

    method_configs = _resolve_methods(args.methods)
    train_cfg = _make_train_cfg(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.alphas) * len(method_configs) * len(args.seeds) * len(args.lambdas)
    print("=" * 70)
    print("Multi-Dimensional Knapsack Experiment")
    print("=" * 70)
    print(f"Scenario: {args.scenario}")
    print(f"Alphas:   {args.alphas}")
    print(f"Population: {args.n_train} train / {args.n_val} val / {args.n_test} test")
    print(f"Resources: {args.n_resources} (one budget each), SNR={args.snr}")
    print(f"Imbalance: benefit_bias={args.benefit_group_bias}, cost_bias={args.cost_group_bias}")
    print(f"Methods:  {list(method_configs.keys())}")
    print(f"Backend:  {args.decision_grad_backend}")
    print(f"Seeds:    {args.seeds}")
    print(f"Lambdas:  {args.lambdas}")
    print(f"Steps:    {args.steps}")
    print(f"Results:  {results_dir}")
    print(f"Total stage runs: {total}")
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN - exiting.")
        return

    all_stage: list[pd.DataFrame] = []
    all_iter: list[pd.DataFrame] = []
    for alpha in args.alphas:
        task_cfg = _make_task_cfg(args, alpha)
        for method_name, method_spec in method_configs.items():
            stage_df, iter_df = _run_single(
                task_cfg=task_cfg,
                train_cfg=train_cfg,
                method_name=method_name,
                method_spec=method_spec,
                alpha=alpha,
                tag=args.tag,
            )
            if not stage_df.empty:
                all_stage.append(stage_df)
            if not iter_df.empty:
                all_iter.append(iter_df)

    if all_stage:
        stage_path = results_dir / "stage_results.csv"
        merged = pd.concat(all_stage, ignore_index=True)
        if stage_path.exists() and not args.overwrite:
            merged = pd.concat([pd.read_csv(stage_path), merged], ignore_index=True)
        merged.to_csv(stage_path, index=False)
        print(f"\nSaved {len(merged)} stage rows to {stage_path}")

    if all_iter:
        iter_path = results_dir / "iter_logs.csv"
        merged = pd.concat(all_iter, ignore_index=True)
        if iter_path.exists() and not args.overwrite:
            merged = pd.concat([pd.read_csv(iter_path), merged], ignore_index=True)
        merged.to_csv(iter_path, index=False)
        print(f"Saved {len(merged)} iter rows to {iter_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
