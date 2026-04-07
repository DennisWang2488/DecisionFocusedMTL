#!/usr/bin/env bash
# run_healthcare_final.sh — Full healthcare experiment grid
#
# Runs the complete experimental grid for the Obermeyer healthcare experiment:
#   Methods:     PTO, FPTO(3 lambdas), SAA, WDRO, FDFL-Scal(3 lambdas),
#                FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad
#   Alpha:       {0.5, 2.0}
#   Hidden dim:  {64, 128}
#   Seeds:       [11, 22, 33, 44, 55]
#
# Total: 12 method-configs x 2 alphas x 2 hidden_dims x 5 seeds = 240 runs
#
# Usage:
#   chmod +x scripts/run_healthcare_final.sh
#   ./scripts/run_healthcare_final.sh              # full run
#   ./scripts/run_healthcare_final.sh --dry-run    # preview
#   ./scripts/run_healthcare_final.sh --overwrite  # re-run all
#
# Idempotent: skips runs where results already exist.

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS="${@}"

echo "============================================================"
echo "Healthcare Final Experiment Grid"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python experiments/run_healthcare_final.py ${EXTRA_ARGS}

echo ""
echo "End time: $(date)"
echo "Results in: results/final/healthcare/"
