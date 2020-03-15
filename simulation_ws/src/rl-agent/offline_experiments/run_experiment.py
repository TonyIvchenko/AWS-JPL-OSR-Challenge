#!/usr/bin/env python3
"""CLI entrypoint for offline RL experiments."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from offline_experiments.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="random")
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slip-probability", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        agent_name=args.agent,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        slip_probability=args.slip_probability,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, sort_keys=True)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
