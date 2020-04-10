# Offline RL Experiments

This folder contains lightweight RL agent experiments that run without ROS/Gazebo.

## Run

```bash
cd simulation_ws/src/rl-agent
python3 offline_experiments/run_experiment.py \
  --agent random \
  --train-episodes 0 \
  --eval-episodes 50 \
  --seed 7 \
  --output offline_experiments/results/random_baseline.json
```

Results are written as JSON to `offline_experiments/results/`.

## Agent Options

- `random`
- `q_learning`
- `sarsa`
- `expected_sarsa`
- `monte_carlo_control`
- `double_q_learning`
- `dyna_q`
- `actor_critic`
- `reinforce`
- `dqn`
- `double_dqn`
- `dueling_dqn`
- `noisy_dqn`
- `c51`
- `qr_dqn`
- `iqn`
- `a2c`
- `a3c`
