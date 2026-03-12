#!/usr/bin/env bash
# run_sample.sh — Small-sample sanity check (n=1000)
#
# Runs SAA, WDRO, PTO, DFL, PCGrad across:
#   - prediction fairness types: mad, gap, atkinson
#   - alpha-fair decision values: 0.5, 1.0, 2.0
#
# Results land in results/sample/ (under the unified results dir).
#
# Usage (run from project root):
#   chmod +x scripts/run_sample.sh
#   ./scripts/run_sample.sh
#   ./scripts/run_sample.sh --dry-run   # preview plan without running

set -e

METHODS="SAA WDRO PTO DFL PCGrad"
ALPHAS="0.5 1.0 2.0"
N=1000
RESULTS_DIR="results/sample"
DATA_CSV="${DATA_CSV:-data/data_processed.csv}"   # override via env if needed
EXTRA_ARGS="${@}"                             # pass --dry-run, --overwrite, etc.

echo "============================================================"
echo "Small-sample experiment: n=${N}"
echo "Methods:  ${METHODS}"
echo "Alphas:   ${ALPHAS}"
echo "Fairness: mad  gap  atkinson"
echo "Results:  ${RESULTS_DIR}/"
echo "============================================================"

for FT in mad gap atkinson; do
    echo ""
    echo ">>> fairness_type=${FT}"
    python run_methods.py \
        --methods ${METHODS} \
        --alphas ${ALPHAS} \
        --fairness-type ${FT} \
        --n-sample ${N} \
        --results-dir "${RESULTS_DIR}" \
        --data-csv "${DATA_CSV}" \
        ${EXTRA_ARGS}
done

echo ""
echo "Done. Results in ${RESULTS_DIR}/stage_results_full.csv"
echo "  (stage_results_full.csv + iter_logs_full.csv)"
