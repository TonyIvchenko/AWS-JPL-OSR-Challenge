"""Training/evaluation runner for offline rover experiments."""

from __future__ import annotations

import statistics
from typing import Dict, List

from .agents import build_agent
from .env import RoverMiniGridEnv


def _run_episode(env: RoverMiniGridEnv, agent, training: bool) -> Dict[str, float | bool | str]:
    state = env.reset()
    done = False
    total_reward = 0.0
    event = ""

    while not done:
        state_idx = env.state_to_index(state)
        action = agent.choose_action(state_idx, training=training)
        step = env.step(action)
        next_state_idx = env.state_to_index(step.state)
        agent.observe(state_idx, action, step.reward, next_state_idx, step.done)

        state = step.state
        done = step.done
        total_reward += step.reward
        if done:
            event = step.info.get("event", "")

    return {
        "reward": total_reward,
        "steps": float(env.steps),
        "success": event == "goal",
        "event": event,
    }


def _aggregate_metrics(rows: List[Dict[str, float | bool | str]]) -> Dict[str, float]:
    rewards = [float(r["reward"]) for r in rows]
    steps = [float(r["steps"]) for r in rows]
    successes = [1.0 if bool(r["success"]) else 0.0 for r in rows]

    return {
        "episodes": float(len(rows)),
        "avg_reward": statistics.fmean(rewards) if rewards else 0.0,
        "avg_steps": statistics.fmean(steps) if steps else 0.0,
        "success_rate": statistics.fmean(successes) if successes else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
    }


def run_experiment(
    agent_name: str,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    slip_probability: float,
) -> Dict[str, object]:
    env = RoverMiniGridEnv(seed=seed, slip_probability=slip_probability)
    agent = build_agent(agent_name=agent_name, num_actions=env.num_actions, seed=seed)

    train_rows = []
    for _ in range(train_episodes):
        train_rows.append(_run_episode(env, agent, training=True))

    eval_rows = []
    for _ in range(eval_episodes):
        eval_rows.append(_run_episode(env, agent, training=False))

    return {
        "agent": agent_name,
        "seed": seed,
        "environment": {
            "width": env.width,
            "height": env.height,
            "start": list(env.start),
            "goal": list(env.goal),
            "obstacle_count": len(env.obstacles),
            "slip_probability": env.slip_probability,
            "max_steps": env.max_steps,
        },
        "train": _aggregate_metrics(train_rows),
        "eval": _aggregate_metrics(eval_rows),
    }
