"""Agents used by offline rover experiments."""

from __future__ import annotations

import math
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


class MonteCarloControlAgent:
    """On-policy Monte Carlo control with epsilon-greedy policy improvement."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        gamma: float = 0.98,
        epsilon: float = 0.2,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]
        self.returns_count: List[List[int]] = [
            [0 for _ in range(num_actions)] for _ in range(num_states)
        ]
        self.episode: List[tuple[int, int, float]] = []

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self.q_table[state_idx])

    def _update_from_episode(self) -> None:
        g_return = 0.0
        for state_idx, action, reward in reversed(self.episode):
            g_return = reward + (self.gamma * g_return)
            count = self.returns_count[state_idx][action] + 1
            self.returns_count[state_idx][action] = count
            current = self.q_table[state_idx][action]
            self.q_table[state_idx][action] = current + ((g_return - current) / count)
        self.episode.clear()

    def observe(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        del next_state_idx, next_action
        self.episode.append((state_idx, action, reward))
        if done:
            self._update_from_episode()


class DoubleQLearningAgent:
    """Double Q-learning with epsilon-greedy action selection on Q1+Q2."""

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
        self.q1: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]
        self.q2: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]

    def _combined_values(self, state_idx: int) -> List[float]:
        return [self.q1[state_idx][a] + self.q2[state_idx][a] for a in range(self.num_actions)]

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self._combined_values(state_idx))

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
        if self.rng.random() < 0.5:
            next_action_idx = self._argmax_action(self.q1[next_state_idx])
            bootstrap = 0.0 if done else self.q2[next_state_idx][next_action_idx]
            current = self.q1[state_idx][action]
            td_target = reward + (self.gamma * bootstrap)
            self.q1[state_idx][action] = current + self.alpha * (td_target - current)
        else:
            next_action_idx = self._argmax_action(self.q2[next_state_idx])
            bootstrap = 0.0 if done else self.q1[next_state_idx][next_action_idx]
            current = self.q2[state_idx][action]
            td_target = reward + (self.gamma * bootstrap)
            self.q2[state_idx][action] = current + self.alpha * (td_target - current)


class DynaQAgent:
    """Dyna-Q with tabular model rollouts for planning updates."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.98,
        epsilon: float = 0.2,
        planning_steps: int = 10,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.rng = random.Random(seed)
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)
        return self._argmax_action(self.q_table[state_idx])

    def _q_learning_update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        current_q = self.q_table[state_idx][action]
        best_next_q = max(self.q_table[next_state_idx])
        td_target = reward + (0.0 if done else self.gamma * best_next_q)
        self.q_table[state_idx][action] = current_q + self.alpha * (td_target - current_q)

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
        self._q_learning_update(state_idx, action, reward, next_state_idx, done)
        self.model[(state_idx, action)] = (reward, next_state_idx, done)

        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            replay_state, replay_action = self.rng.choice(keys)
            replay_reward, replay_next_state, replay_done = self.model[(replay_state, replay_action)]
            self._q_learning_update(
                replay_state,
                replay_action,
                replay_reward,
                replay_next_state,
                replay_done,
            )


class ActorCriticAgent:
    """Tabular actor-critic using softmax policy preferences and state values."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: int = 0,
        alpha_policy: float = 0.08,
        alpha_value: float = 0.15,
        gamma: float = 0.98,
        temperature: float = 1.0,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha_policy = alpha_policy
        self.alpha_value = alpha_value
        self.gamma = gamma
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.values = [0.0 for _ in range(num_states)]
        self.preferences: List[List[float]] = [
            [0.0 for _ in range(num_actions)] for _ in range(num_states)
        ]

    def _softmax_probs(self, state_idx: int) -> List[float]:
        prefs = self.preferences[state_idx]
        scaled = [value / self.temperature for value in prefs]
        max_scaled = max(scaled)
        exp_vals = [math.exp(value - max_scaled) for value in scaled]
        total = sum(exp_vals)
        return [value / total for value in exp_vals]

    def _argmax_action(self, values: List[float]) -> int:
        max_value = max(values)
        max_actions = [i for i, value in enumerate(values) if value == max_value]
        return self.rng.choice(max_actions)

    def choose_action(self, state_idx: int, training: bool) -> int:
        probs = self._softmax_probs(state_idx)
        if not training:
            return self._argmax_action(probs)
        sample = self.rng.random()
        cumulative = 0.0
        for action, prob in enumerate(probs):
            cumulative += prob
            if sample <= cumulative:
                return action
        return self.num_actions - 1

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
        v_state = self.values[state_idx]
        v_next = 0.0 if done else self.values[next_state_idx]
        td_error = reward + (self.gamma * v_next) - v_state

        self.values[state_idx] = v_state + (self.alpha_value * td_error)

        probs = self._softmax_probs(state_idx)
        for candidate_action in range(self.num_actions):
            grad = (1.0 - probs[candidate_action]) if candidate_action == action else -probs[candidate_action]
            self.preferences[state_idx][candidate_action] += self.alpha_policy * td_error * grad


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
    if agent_name == "monte_carlo_control":
        return MonteCarloControlAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
    if agent_name == "double_q_learning":
        return DoubleQLearningAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
    if agent_name == "dyna_q":
        return DynaQAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
    if agent_name == "actor_critic":
        return ActorCriticAgent(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )

    raise ValueError(f"Unknown agent '{agent_name}'")
