"""Agents used by offline rover experiments."""

from __future__ import annotations

import random
from typing import Protocol


class Agent(Protocol):
    def choose_action(self, state_idx: int, training: bool) -> int:
        ...

    def observe(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        ...


class RandomAgent:
    """Policy baseline that samples actions uniformly."""

    def __init__(self, num_actions: int, seed: int = 0) -> None:
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def choose_action(self, state_idx: int, training: bool) -> int:
        del state_idx, training
        return self.rng.randrange(self.num_actions)

    def observe(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        del state_idx, action, reward, next_state_idx, done
        return None


def build_agent(agent_name: str, num_actions: int, seed: int = 0) -> Agent:
    if agent_name == "random":
        return RandomAgent(num_actions=num_actions, seed=seed)

    raise ValueError(f"Unknown agent '{agent_name}'")
