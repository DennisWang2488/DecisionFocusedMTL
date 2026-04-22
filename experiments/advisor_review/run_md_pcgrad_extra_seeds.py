"""Extra-seeds robustness check for normalized PCGrad on MD knapsack.

Same task/training config as run_md_knapsack_mu_sweep.py, but runs only
PCGrad on seeds {66, 77, 88, 99} to check how often normalize=True
still produces an exploded seed under SPSA.

Output:
    results/advisor_review/md_knapsack_mu_sweep/extra_seeds/
        seed_{66,77,88,99}/stage_results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.runner import (  # noqa: E402
    make_md_task_cfg,
    make_train_cfg,
    run_one,
)

OUT_ROOT = REPO_ROOT / "results" / "advisor_review" / "md_knapsack_mu_sweep" / "extra_seeds"
METHODS = ["PCGrad"]
SEEDS = [66, 77, 88, 99]
LAMBDAS = [0.0, 0.5, 1.0, 2.0]


def main() -> None:
    print(f"MD knapsack PCGrad extra-seed robustness ({SEEDS})")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        task_cfg = make_md_task_cfg(
            n_train=300, n_val=20, n_test=40, n_features=5, n_resources=2,
            scenario="alpha_fair", alpha_fair=2.0, poly_degree=2, snr=5.0,
            benefit_group_bias=0.4, benefit_noise_ratio=1.0,
            cost_group_bias=0.0, cost_noise_ratio=1.0,
            cost_mean=1.0, cost_std=0.2, budget_tightness=0.35,
            fairness_type="mad", group_ratio=0.5, decision_mode="group",
            data_seed=42,
        )
        train_cfg = make_train_cfg(
            seeds=[s], lambdas=LAMBDAS, steps=30, lr=5e-4,
            hidden_dim=32, n_layers=2, arch="mlp",
            decision_grad_backend="spsa", spsa_eps=5e-3, spsa_n_dirs=8,
            batch_size=-1,
        )
        sub = OUT_ROOT / f"seed_{s}"
        t0 = time.time()
        run_one(
            out_dir=sub, task_cfg=task_cfg, train_cfg=train_cfg,
            methods=METHODS, label=f"pcgrad_extra_s{s}", overwrite=True,
        )
        print(f"  seed {s}: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
