#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python experiments/md_knapsack/run_md_knapsack.py \
  --scenario alpha_fair \
  --alphas 0.5 1.0 2.0 \
  --methods FPTO FDFL MGDA PCGrad PTO \
  --n-items 10 \
  --n-constraints 2 \
  --n-train 30 \
  --n-val 10 \
  --n-test 30 \
  --steps 10 \
  --seeds 11 \
  --lambdas 0.0 0.1 0.5 \
  --decision-grad-backend finite_diff \
  --tag main
