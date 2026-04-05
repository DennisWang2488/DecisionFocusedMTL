# Multi-Dimensional Knapsack Experiment

This experiment ports the older synthetic multi-dimensional knapsack task onto the newer codebase.

Key differences from the older snapshot:

- The current repo only supports `finite_diff` decision gradients for this task.
- Advanced backends such as `FFO`, `NCE`, and `LANCER` remain medical-task specific and are intentionally excluded here.
- The task is wired directly into `fair_dfl.runner._build_task()` as `name: "md_knapsack"`.

Quick smoke run:

```bash
python experiments/md_knapsack/run_md_knapsack.py \
  --methods FPTO FDFL \
  --alphas 2.0 \
  --n-items 6 \
  --n-constraints 2 \
  --n-train 12 \
  --n-val 4 \
  --n-test 4 \
  --steps 1 \
  --seeds 11 \
  --lambdas 0.0 \
  --device cpu \
  --overwrite
```
