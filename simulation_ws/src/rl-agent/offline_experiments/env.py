"""Minimal rover-like grid environment for offline RL experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Tuple


Action = int
State = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool
    info: dict


class RoverMiniGridEnv:
    """A small grid map with rocks (terminal collisions) and a goal checkpoint."""

    ACTIONS = {
        0: (0, 1),   # north
        1: (1, 0),   # east
        2: (0, -1),  # south
        3: (-1, 0),  # west
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        start: State = (0, 0),
        goal: State = (9, 9),
        obstacles: Iterable[State] | None = None,
        slip_probability: float = 0.1,
        max_steps: int = 80,
        seed: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or {
            (2, 2), (2, 3), (2, 4),
            (5, 5), (5, 6),
            (7, 2), (7, 3),
            (3, 7), (4, 7),
        })
        self.slip_probability = slip_probability
        self.max_steps = max_steps
        self.num_actions = len(self.ACTIONS)
        self.rng = random.Random(seed)

        self.position: State = self.start
        self.steps = 0

    def num_states(self) -> int:
        return self.width * self.height

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self.rng.seed(seed)
        self.position = self.start
        self.steps = 0
        return self.position

    def state_to_index(self, state: State) -> int:
        x, y = state
        return y * self.width + x

    def _clip_position(self, x: int, y: int) -> State:
        return max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y))

    def _maybe_slip(self, action: Action) -> Action:
        if self.rng.random() >= self.slip_probability:
            return action
        options = [a for a in self.ACTIONS if a != action]
        return self.rng.choice(options)

    def step(self, action: Action) -> StepResult:
        self.steps += 1
        actual_action = self._maybe_slip(action)
        dx, dy = self.ACTIONS[actual_action]

        x, y = self.position
        nx, ny = self._clip_position(x + dx, y + dy)
        next_state = (nx, ny)

        reward = -1.0
        done = False
        info = {"slipped": actual_action != action}

        if next_state in self.obstacles:
            self.position = next_state
            reward = -25.0
            done = True
            info["event"] = "collision"
            return StepResult(self.position, reward, done, info)

        self.position = next_state

        if self.position == self.goal:
            reward = 50.0
            done = True
            info["event"] = "goal"
            return StepResult(self.position, reward, done, info)

        if self.steps >= self.max_steps:
            reward = -10.0
            done = True
            info["event"] = "max_steps"

        return StepResult(self.position, reward, done, info)
