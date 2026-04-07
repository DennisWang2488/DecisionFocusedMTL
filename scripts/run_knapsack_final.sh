#!/usr/bin/env bash
# run_knapsack_final.sh — Full synthetic knapsack experiment grid
#
# Runs the complete experimental grid for the multi-dimensional knapsack:
#   Methods:     PTO, FPTO(3 lambdas), SAA, WDRO, FDFL-Scal(3 lambdas),
#                FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad
#   Alpha:       {0.5, 2.0}
#   Unfairness:  {mild, medium, high}
#   Seeds:       [11, 22, 33, 44, 55]
#
# Total: 12 method-configs x 2 alphas x 3 unfairness x 5 seeds = 360 runs
#
# Usage:
#   chmod +x scripts/run_knapsack_final.sh
#   ./scripts/run_knapsack_final.sh              # full run
#   ./scripts/run_knapsack_final.sh --dry-run    # preview
#   ./scripts/run_knapsack_final.sh --overwrite  # re-run all
#
# Idempotent: skips runs where results already exist.

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS="${@}"

echo "============================================================"
echo "Knapsack Final Experiment Grid"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python experiments/run_knapsack_final.py ${EXTRA_ARGS}

echo ""
echo "End time: $(date)"
echo "Results in: results/final/knapsack/"
