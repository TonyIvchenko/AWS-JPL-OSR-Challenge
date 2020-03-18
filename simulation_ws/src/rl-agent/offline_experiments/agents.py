"""Agents used by offline rover experiments."""

from __future__ import annotations

import random
from typing import List
from typing import Protocol


class Agent(Protocol):
    def choose_action(self, state_idx: int, training: bool) -> int:
        ...

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        ...


class RandomAgent:
    """Policy baseline that samples actions uniformly."""

    def __init__(self, num_actions: int, seed: int = 0) -> None:
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def choose_action(self, state_idx: int, training: bool) -> int:
        del state_idx, training
        return self.rng.randrange(self.num_actions)

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        del state_idx, action, reward, next_state_idx, done, next_action
        return None


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.98,
        epsilon: float = 0.2,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self.q_table[state_idx])

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        del next_action
        current_q = self.q_table[state_idx][action]
        best_next_q = max(self.q_table[next_state_idx])
        td_target = reward + (0.0 if done else self.gamma * best_next_q)
        self.q_table[state_idx][action] = current_q + self.alpha * (td_target - current_q)


class SarsaAgent:
    """Tabular SARSA with epsilon-greedy exploration."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.98,
        epsilon: float = 0.2,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self.q_table[state_idx])

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        current_q = self.q_table[state_idx][action]
        next_q = 0.0 if done or next_action is None else self.q_table[next_state_idx][next_action]
        td_target = reward + (self.gamma * next_q)
        self.q_table[state_idx][action] = current_q + self.alpha * (td_target - current_q)


class ExpectedSarsaAgent:
    """Expected SARSA using epsilon-greedy expectation for the bootstrap value."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.98,
        epsilon: float = 0.2,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self.q_table[state_idx])

    def _expected_next_q(self, next_state_idx: int) -> float:
        values = self.q_table[next_state_idx]
        max_value = max(values)
        greedy_actions = [i for i, value in enumerate(values) if value == max_value]
        greedy_prob = (1.0 - self.epsilon) / len(greedy_actions)
        random_prob = self.epsilon / self.num_actions
        expectation = 0.0
        for action, q_value in enumerate(values):
            prob = random_prob + (greedy_prob if action in greedy_actions else 0.0)
            expectation += prob * q_value
        return expectation

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        del next_action
        current_q = self.q_table[state_idx][action]
        expected_q = 0.0 if done else self._expected_next_q(next_state_idx)
        td_target = reward + (self.gamma * expected_q)
        self.q_table[state_idx][action] = current_q + self.alpha * (td_target - current_q)


def build_agent(
    agent_name: str, num_actions: int, num_states: int, seed: int = 0
) -> Agent:
    if agent_name == "random":
        return RandomAgent(num_actions=num_actions, seed=seed)
    if agent_name == "q_learning":
        return QLearningAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
    if agent_name == "sarsa":
        return SarsaAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
    if agent_name == "expected_sarsa":
        return ExpectedSarsaAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )

    raise ValueError(f"Unknown agent '{agent_name}'")
