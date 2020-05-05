# Offline Model Tuning Log

This log tracks iterative hyperparameter changes for existing model profiles in `offline_experiments/agents.py`.

| Iteration | Model | Key Changes | Train Episodes | Eval Avg Reward | Eval Success Rate | Result File |
|---|---|---|---:|---:|---:|---|
| 1 | dqn | alpha 0.16, epsilon 0.28->0.02, decay 0.99965, replay 5000, batch 48, target sync 170 | 1700 | 27.47 | 0.97 | `results/tuning/dqn_tune_01.json` |
| 2 | double_dqn | alpha 0.19, epsilon 0.30->0.015, decay 0.9996, replay 6000, batch 56, target sync 160 | 1700 | 29.06 | 0.99 | `results/tuning/double_dqn_tune_01.json` |
| 3 | dueling_dqn | alpha 0.17, epsilon 0.22->0.02, decay 0.99955, replay 5500, batch 52, target sync 140 | 1600 | 19.24 | 0.83 | `results/tuning/dueling_dqn_tune_01.json` |
