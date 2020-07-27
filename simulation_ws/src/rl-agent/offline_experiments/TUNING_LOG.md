# Offline Model Tuning Log

This log tracks iterative hyperparameter changes for existing model profiles in `offline_experiments/agents.py`.

| Iteration | Model | Key Changes | Train Episodes | Eval Avg Reward | Eval Success Rate | Result File |
|---|---|---|---:|---:|---:|---|
| 1 | dqn | alpha 0.16, epsilon 0.28->0.02, decay 0.99965, replay 5000, batch 48, target sync 170 | 1700 | 27.47 | 0.97 | `results/tuning/dqn_tune_01.json` |
| 2 | double_dqn | alpha 0.19, epsilon 0.30->0.015, decay 0.9996, replay 6000, batch 56, target sync 160 | 1700 | 29.06 | 0.99 | `results/tuning/double_dqn_tune_01.json` |
| 3 | dueling_dqn | alpha 0.17, epsilon 0.22->0.02, decay 0.99955, replay 5500, batch 52, target sync 140 | 1600 | 19.24 | 0.83 | `results/tuning/dueling_dqn_tune_01.json` |
| 4 | noisy_dqn | alpha 0.20, epsilon 0.50->0.06, decay 0.99935, replay 6500, batch 56, target sync 110 | 1600 | 11.99 | 0.95 | `results/tuning/noisy_dqn_tune_01.json` |
| 5 | c51 | alpha 0.14, epsilon 0.25->0.015, decay 0.99965, replay 6500, batch 56, target sync 180 | 1700 | 25.36 | 0.92 | `results/tuning/c51_tune_01.json` |
| 6 | qr_dqn | alpha 0.18, epsilon 0.28->0.02, decay 0.99955, replay 7000, batch 56, target sync 170 | 1700 | 18.62 | 0.86 | `results/tuning/qr_dqn_tune_01.json` |
| 7 | iqn | alpha 0.16, epsilon 0.24->0.015, decay 0.99962, replay 7000, batch 56, target sync 180 | 1700 | -7.56 | 0.80 | `results/tuning/iqn_tune_01.json` |
| 8 | a2c | alpha_policy 0.07, alpha_value 0.22, gamma 0.99, temperature 0.95 | 2200 | 28.55 | 0.98 | `results/tuning/a2c_tune_01.json` |
| 9 | a3c | alpha_policy 0.065, alpha_value 0.20, gamma 0.99, temperature 1.05 | 2200 | 26.84 | 0.95 | `results/tuning/a3c_tune_01.json` |
| 10 | ppo | alpha_policy 0.055, alpha_value 0.18, gamma 0.995, temperature 0.90 | 2200 | 26.01 | 0.94 | `results/tuning/ppo_tune_01.json` |
| 11 | trpo | alpha 0.012, gamma 0.995, temperature 0.85 | 3000 | -28.41 | 0.00 | `results/tuning/trpo_tune_01.json` |
| 12 | sac | alpha_policy 0.085, alpha_value 0.16, gamma 0.995, temperature 1.15 | 2200 | 29.86 | 1.00 | `results/tuning/sac_tune_01.json` |
| 13 | ddpg | alpha 0.18, epsilon 0.14->0.005, decay 0.99975, replay 7000, batch 56, target sync 140 | 1700 | 24.22 | 0.92 | `results/tuning/ddpg_tune_01.json` |
| 14 | td3 | alpha 0.16, epsilon 0.16->0.01, decay 0.9997, replay 7500, batch 56, target sync 130 | 1700 | -5.70 | 0.84 | `results/tuning/td3_tune_01.json` |
| 15 | impala | alpha_policy 0.070, alpha_value 0.21, gamma 0.995, temperature 0.98 | 2200 | 22.24 | 0.89 | `results/tuning/impala_tune_01.json` |
| 16 | r2d2 | alpha 0.15, epsilon 0.22->0.01, decay 0.99975, replay 9000, batch 72, target sync 200 | 1700 | 15.13 | 0.83 | `results/tuning/r2d2_tune_01.json` |
| 17 | bootstrapped_dqn | alpha 0.18, epsilon 0.42->0.03, decay 0.99945, replay 8500, batch 64, target sync 180 | 1700 | 21.05 | 0.86 | `results/tuning/bootstrapped_dqn_tune_01.json` |
| 18 | apex_dqn | alpha 0.20, epsilon 0.35->0.02, decay 0.9996, replay 10000, batch 72, target sync 120 | 1700 | -7.03 | 0.83 | `results/tuning/apex_dqn_tune_01.json` |
| 19 | actor_critic | alpha_policy 0.09, alpha_value 0.17, gamma 0.99, temperature 0.92 | 2200 | 28.51 | 0.98 | `results/tuning/actor_critic_tune_01.json` |
| 20 | reinforce | alpha 0.018, gamma 0.99, temperature 0.88 | 3000 | -28.65 | 0.00 | `results/tuning/reinforce_tune_01.json` |
| 21 | dqn | alpha 0.17, epsilon 0.24->0.01, decay 0.99972, replay 6500, batch 56, target sync 150 | 1800 | 27.84 | 0.97 | `results/tuning/dqn_tune_02.json` |
| 22 | double_dqn | alpha 0.18, epsilon 0.26->0.01, decay 0.99968, replay 7000, batch 64, target sync 140 | 1800 | 29.91 | 0.99 | `results/tuning/double_dqn_tune_02.json` |
| 23 | dueling_dqn | alpha 0.15, epsilon 0.20->0.015, decay 0.99960, replay 6000, batch 56, target sync 130 | 1700 | 26.53 | 0.97 | `results/tuning/dueling_dqn_tune_02.json` |
| 24 | noisy_dqn | alpha 0.19, epsilon 0.55->0.05, decay 0.99940, replay 7000, batch 64, target sync 100 | 1700 | -3.67 | 0.75 | `results/tuning/noisy_dqn_tune_02.json` |
| 25 | c51 | alpha 0.13, epsilon 0.22->0.01, decay 0.99970, replay 7000, batch 64, target sync 160 | 1800 | 21.15 | 0.86 | `results/tuning/c51_tune_02.json` |
| 26 | qr_dqn | alpha 0.17, epsilon 0.24->0.015, decay 0.99965, replay 8000, batch 64, target sync 150 | 1800 | 26.43 | 0.98 | `results/tuning/qr_dqn_tune_02.json` |
| 27 | iqn | alpha 0.14, epsilon 0.20->0.01, decay 0.99970, replay 8000, batch 64, target sync 160 | 1800 | 26.36 | 0.99 | `results/tuning/iqn_tune_02.json` |
| 28 | a2c | alpha_policy 0.08, alpha_value 0.20, gamma 0.995, temperature 0.90 | 2400 | 22.76 | 0.89 | `results/tuning/a2c_tune_02.json` |
| 29 | a3c | alpha_policy 0.070, alpha_value 0.19, gamma 0.995, temperature 1.00 | 2400 | 28.60 | 0.98 | `results/tuning/a3c_tune_02.json` |
| 30 | ppo | alpha_policy 0.050, alpha_value 0.20, gamma 0.995, temperature 0.85 | 2400 | 27.38 | 0.99 | `results/tuning/ppo_tune_02.json` |
| 31 | trpo | alpha 0.010, gamma 0.997, temperature 0.80 | 3200 | -28.54 | 0.00 | `results/tuning/trpo_tune_02.json` |
| 32 | sac | alpha_policy 0.090, alpha_value 0.15, gamma 0.997, temperature 1.05 | 2400 | 27.88 | 0.97 | `results/tuning/sac_tune_02.json` |
| 33 | ddpg | alpha 0.16, epsilon 0.12->0.003, decay 0.99980, replay 8000, batch 64, target sync 120 | 1800 | -13.61 | 0.60 | `results/tuning/ddpg_tune_02.json` |
| 34 | td3 | alpha 0.15, epsilon 0.14->0.005, decay 0.99975, replay 8500, batch 64, target sync 110 | 1800 | 23.48 | 0.93 | `results/tuning/td3_tune_02.json` |
| 35 | impala | alpha_policy 0.072, alpha_value 0.22, gamma 0.995, temperature 0.95 | 2400 | 25.11 | 0.93 | `results/tuning/impala_tune_02.json` |
| 36 | r2d2 | alpha 0.14, epsilon 0.20->0.005, decay 0.99980, replay 10000, batch 80, target sync 170 | 1800 | 24.49 | 0.91 | `results/tuning/r2d2_tune_02.json` |
| 37 | bootstrapped_dqn | alpha 0.17, epsilon 0.36->0.02, decay 0.99955, replay 9000, batch 72, target sync 160 | 1800 | 23.23 | 0.89 | `results/tuning/bootstrapped_dqn_tune_02.json` |
| 38 | apex_dqn | alpha 0.18, epsilon 0.30->0.015, decay 0.99965, replay 12000, batch 80, target sync 100 | 1800 | 27.54 | 0.96 | `results/tuning/apex_dqn_tune_02.json` |
| 39 | actor_critic | alpha_policy 0.085, alpha_value 0.18, gamma 0.995, temperature 0.88 | 2400 | 26.18 | 0.94 | `results/tuning/actor_critic_tune_02.json` |
| 40 | reinforce | alpha 0.015, gamma 0.995, temperature 0.82 | 3200 | -28.79 | 0.00 | `results/tuning/reinforce_tune_02.json` |
| 41 | dqn | alpha 0.16, epsilon 0.21->0.008, decay 0.99976, replay 7000, batch 64, target sync 135 | 1900 | 21.34 | 0.88 | `results/tuning/dqn_tune_03.json` |
| 42 | double_dqn | alpha 0.17, epsilon 0.23->0.008, decay 0.99972, replay 8000, batch 72, target sync 125 | 1900 | 27.37 | 0.97 | `results/tuning/double_dqn_tune_03.json` |
| 43 | dueling_dqn | alpha 0.14, epsilon 0.18->0.01, decay 0.99965, replay 6500, batch 64, target sync 120 | 1800 | 21.69 | 0.86 | `results/tuning/dueling_dqn_tune_03.json` |
| 44 | noisy_dqn | alpha 0.18, epsilon 0.48->0.04, decay 0.99948, replay 7500, batch 72, target sync 90 | 1800 | 19.81 | 0.84 | `results/tuning/noisy_dqn_tune_03.json` |
| 45 | c51 | alpha 0.12, epsilon 0.20->0.008, decay 0.99974, replay 7500, batch 72, target sync 145 | 1900 | -32.09 | 0.40 | `results/tuning/c51_tune_03.json` |
| 46 | qr_dqn | alpha 0.16, epsilon 0.22->0.01, decay 0.99970, replay 9000, batch 72, target sync 135 | 1900 | 20.87 | 0.87 | `results/tuning/qr_dqn_tune_03.json` |
