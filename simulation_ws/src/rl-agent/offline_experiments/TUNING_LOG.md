# Offline Model Tuning Log

This log tracks iterative hyperparameter changes for existing model profiles in `offline_experiments/agents.py`.

| Iteration | Model | Key Changes | Train Episodes | Eval Avg Reward | Eval Success Rate | Result File |
|---|---|---|---:|---:|---:|---|
| 1 | dqn | alpha 0.16, epsilon 0.28->0.02, decay 0.99965, replay 5000, batch 48, target sync 170 | 1700 | 27.47 | 0.97 | `results/tuning/dqn_tune_01.json` |
