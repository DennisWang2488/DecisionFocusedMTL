"""Backend sanity check for healthcare decision gradients.

Purpose
-------
Diagnose whether the surprising Phase 1 healthcare SPSA result
(`FPTO` winning all SPSA pilot cells) is specific to SPSA noise or
persists under a different approximate decision-gradient backend.

Default experiment
------------------
- task: healthcare (`medical_resource_allocation`)
- fairness_type: `mad`
- alphas: {0.5, 1.5, 2.0, 3.0}
- seed: {1}
- lambdas: {0.0, 1.0}
- steps_per_lambda: 70
- methods: Phase 1 pilot methods
- backends: analytic, finite_diff, spsa
- device: cuda
- n_sample: 0 (full cohort)

Outputs
-------
`results/pilot_alignmo/backend_sanity/<backend>/alpha_<a>/seed_<s>/`
containing `stage_results.csv`, `iter_logs.csv`, and `config.json`.

Important
---------
The current full-cohort healthcare `finite_diff` path is extremely
expensive: the generic finite-difference strategy performs O(n_train)
decision solves per step. This launcher therefore requires an explicit
`--allow-expensive-fd` flag before running `finite_diff` together with
`n_sample=0`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.runner import (  # noqa: E402
    make_healthcare_task_cfg,
    make_train_cfg,
    run_one,
)


OUT_DIR = REPO_ROOT / "results" / "pilot_alignmo" / "backend_sanity"
ALPHAS = [0.5, 1.5, 2.0, 3.0]
SEEDS = [1]
LAMBDAS = [0.0, 1.0]
METHODS = [
    "FPTO",
    "FDFL-Scal",
    "FDFL-0.1",
    "FDFL-0.5",
    "FDFL",
    "FPLG",
    "PCGrad",
    "MGDA",
]
BACKENDS = {
    "analytic": {"decision_grad_backend": "analytic"},
    "finite_diff": {
        "decision_grad_backend": "finite_diff",
        "decision_grad_fd_eps": 1e-3,
    },
    "spsa": {
        "decision_grad_backend": "spsa",
        "decision_grad_spsa_eps": 1e-3,
        "decision_grad_spsa_n_dirs": 8,
    },
}


def _healthcare_task_cfg(*, alpha_fair: float, split_seed: int, n_sample: int) -> dict:
    return make_healthcare_task_cfg(
        n_sample=int(n_sample),
        val_fraction=0.0,
        test_fraction=0.5,
        alpha_fair=float(alpha_fair),
        fairness_type="mad",
        budget_rho=0.30,
        split_seed=int(split_seed),
        data_seed=42,
    )


def _train_cfg(
    *,
    backend: str,
    seeds: Iterable[int],
    lambdas: Iterable[float],
    steps: int,
    device: str,
) -> dict:
    cfg = make_train_cfg(
        seeds=list(seeds),
        lambdas=list(lambdas),
        steps=int(steps),
        lr=1e-3,
        hidden_dim=64,
        n_layers=2,
        arch="mlp",
        decision_grad_backend=BACKENDS[backend]["decision_grad_backend"],
        spsa_eps=float(BACKENDS[backend].get("decision_grad_spsa_eps", 5e-3)),
        spsa_n_dirs=int(BACKENDS[backend].get("decision_grad_spsa_n_dirs", 4)),
        batch_size=-1,
        device=str(device),
        eval_train=True,
        extra={
            "lr_decay": 5e-4,
            "decision_grad_fd_eps": float(BACKENDS[backend].get("decision_grad_fd_eps", 1e-3)),
        },
    )
    return cfg


def _run_backend_grid(
    *,
    backend: str,
    alphas: Iterable[float],
    seeds: Iterable[int],
    lambdas: Iterable[float],
    steps: int,
    n_sample: int,
    device: str,
    methods: Iterable[str],
    out_root: Path,
    overwrite: bool,
) -> list[dict]:
    summary: list[dict] = []
    for alpha in alphas:
        for seed in seeds:
            task_cfg = _healthcare_task_cfg(alpha_fair=float(alpha), split_seed=int(seed), n_sample=int(n_sample))
            train_cfg = _train_cfg(
                backend=backend,
                seeds=[int(seed)],
                lambdas=lambdas,
                steps=int(steps),
                device=device,
            )
            subdir = out_root / backend / f"alpha_{alpha}" / f"seed_{seed}"
            label = f"hc_backend_{backend}_a{alpha}_s{seed}"
            stage_df, _, elapsed = run_one(
                out_dir=subdir,
                task_cfg=task_cfg,
                train_cfg=train_cfg,
                methods=list(methods),
                label=label,
                overwrite=overwrite,
            )
            summary.append(
                {
                    "backend": backend,
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "elapsed_sec": float(elapsed),
                    "n_rows": int(len(stage_df)),
                }
            )
            print(
                f"[backend_sanity] backend={backend} alpha={alpha} seed={seed}: "
                f"{elapsed:.1f}s, {len(stage_df)} rows"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare backend sanity comparison.")
    parser.add_argument(
        "--backend",
        choices=["analytic", "finite_diff", "spsa", "all"],
        default="all",
        help="Which backend(s) to run.",
    )
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--n-sample", type=int, default=0, help="0 means full cohort.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-expensive-fd",
        action="store_true",
        help="Required to run finite_diff with n_sample=0, since that path is extremely expensive.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_DIR,
        help="Output root directory.",
    )
    args = parser.parse_args()

    backends = ["analytic", "finite_diff", "spsa"] if args.backend == "all" else [args.backend]
    if args.n_sample == 0 and "finite_diff" in backends and not args.allow_expensive_fd:
        raise SystemExit(
            "Refusing to run full-cohort healthcare finite_diff without --allow-expensive-fd. "
            "That path is O(n_train) decision solves per step and is likely intractable."
        )

    args.out_root.mkdir(parents=True, exist_ok=True)

    print("=== Healthcare backend sanity comparison ===")
    print(f"Output root:     {args.out_root}")
    print(f"Backends:        {backends}")
    print(f"Alphas:          {ALPHAS}")
    print(f"Seeds:           {SEEDS}")
    print(f"Methods:         {METHODS}")
    print(f"Lambdas:         {LAMBDAS}")
    print(f"Steps/lambda:    {args.steps}")
    print(f"n_sample:        {args.n_sample}")
    print(f"Device:          {args.device}")
    print("")

    t0 = time.time()
    summary: list[dict] = []
    for backend in backends:
        summary.extend(
            _run_backend_grid(
                backend=backend,
                alphas=ALPHAS,
                seeds=SEEDS,
                lambdas=LAMBDAS,
                steps=args.steps,
                n_sample=args.n_sample,
                device=args.device,
                methods=METHODS,
                out_root=args.out_root,
                overwrite=args.overwrite,
            )
        )
    elapsed = time.time() - t0

    summary_path = args.out_root / "grid_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "n_runs": len(summary),
                "alphas": ALPHAS,
                "seeds": SEEDS,
                "methods": METHODS,
                "lambdas": LAMBDAS,
                "steps_per_lambda": int(args.steps),
                "n_sample": int(args.n_sample),
                "device": str(args.device),
                "backends": backends,
            },
            f,
            indent=2,
        )
    print("")
    print("=== backend sanity summary ===")
    print(f"  runs:           {len(summary)}")
    print(f"  total elapsed:  {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
