#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grader for the MangaTracker Environment.

This module evaluates an agent's efficiency over 100 steps by running
the environment and computing performance metrics.

Usage:
    python grader.py
    python grader.py --steps 100 --trials 5
"""

import argparse
import random
from dataclasses import dataclass
from typing import List

try:
    from server.environment import MangaTrackerEnvironment
    from models import ActionType, MangaTrackerAction
except ImportError:
    from manga_tracker.server.environment import MangaTrackerEnvironment
    from manga_tracker.models import ActionType, MangaTrackerAction


@dataclass
class GradingResult:
    """Results from grading an agent's performance."""
    total_steps: int
    total_reward: float
    chapters_found: int
    rate_limit_hits: int
    db_updates: int
    idle_actions: int
    check_actions: int
    avg_reward_per_step: float
    efficiency_score: float
    passed: bool


class SimpleAgent:
    """
    A simple heuristic agent for testing the environment.

    Strategy:
        - Prioritize checking sources with pending chapters
        - Avoid checking all sources at once (rate limit risk)
        - Update database periodically
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.last_check_all_step = -10

    def act(self, env: MangaTrackerEnvironment) -> MangaTrackerAction:
        """Decide on an action based on current state."""
        state = env.get_full_state()

        if state is None:
            return MangaTrackerAction(action_type=ActionType.IDLE)

        # Find manga with pending chapters
        pending_indices = [
            i for i, entry in enumerate(state.watchlist)
            if entry.chapters_pending > 0
        ]

        if not pending_indices:
            # No pending chapters - idle or check all for new releases
            if self.rng.random() < 0.3:
                return MangaTrackerAction(action_type=ActionType.IDLE)
            else:
                return MangaTrackerAction(
                    action_type=ActionType.CHECK_SOURCE,
                    check_all=True
                )

        # Check if we should update database
        if state.db_updates_count % 5 == 0 and state.db_updates_count > 0:
            if self.rng.random() < 0.3:
                return MangaTrackerAction(action_type=ActionType.UPDATE_DB)

        # Decide between checking specific source or all
        steps_since_check_all = env.state.step_count - self.last_check_all_step

        if len(pending_indices) >= 3 and steps_since_check_all > 10:
            # Many pending and haven't checked all recently
            if self.rng.random() < 0.2:
                self.last_check_all_step = env.state.step_count
                return MangaTrackerAction(
                    action_type=ActionType.CHECK_SOURCE,
                    check_all=True
                )

        # Check a specific source with pending chapters
        idx = self.rng.choice(pending_indices)
        return MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE,
            manga_index=idx
        )


class RandomAgent:
    """A random agent for baseline comparison."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, env: MangaTrackerEnvironment) -> MangaTrackerAction:
        """Select a random action."""
        action_type = self.rng.choice([
            ActionType.CHECK_SOURCE,
            ActionType.IDLE,
            ActionType.UPDATE_DB,
        ])

        if action_type == ActionType.CHECK_SOURCE:
            if self.rng.random() < 0.3:
                return MangaTrackerAction(
                    action_type=action_type,
                    check_all=True
                )
            else:
                return MangaTrackerAction(
                    action_type=action_type,
                    manga_index=self.rng.randint(0, 9)
                )
        else:
            return MangaTrackerAction(action_type=action_type)


def run_evaluation(
    agent: object,
    steps: int = 100,
    verbose: bool = False
) -> GradingResult:
    """
    Run the environment evaluation for a given number of steps.

    Args:
        agent: Agent object with act(env) method
        steps: Number of steps to run
        verbose: Whether to print detailed logs

    Returns:
        GradingResult with performance metrics
    """
    env = MangaTrackerEnvironment()
    env.reset()

    total_reward = 0.0
    chapters_found = 0
    rate_limit_hits = 0
    db_updates = 0
    idle_actions = 0
    check_actions = 0

    for step in range(steps):
        # Get action from agent
        action = agent.act(env)  # type: ignore

        # Execute action
        observation = env.step(action)

        # Track metrics
        total_reward += observation.reward
        chapters_found += observation.new_chapters_found

        if observation.rate_limited:
            rate_limit_hits += 1

        if action.action_type == ActionType.IDLE:
            idle_actions += 1
        elif action.action_type == ActionType.CHECK_SOURCE:
            check_actions += 1
        elif action.action_type == ActionType.UPDATE_DB:
            db_updates += 1

        if verbose:
            print(f"Step {step + 1}/{steps}: "
                  f"Reward={observation.reward:.1f}, "
                  f"Chapters={observation.new_chapters_found}, "
                  f"RateLimited={observation.rate_limited}")

    # Calculate efficiency score strictly bounded to (0, 1)
    
    raw_efficiency = (chapters_found * 100 - rate_limit_hits * 50) / max(1, steps)
    
    # Sigmoid normalization mapped strictly into (0.01, 0.99)
    import math
    try:
        normalized_efficiency = 1.0 / (1.0 + math.exp(-raw_efficiency / 100.0))
    except OverflowError:
        normalized_efficiency = 0.99 if raw_efficiency > 0 else 0.01
        
    efficiency_score = max(0.01, min(0.99, normalized_efficiency))

    # Determine if agent passed (raw efficiency > 50 and chapters_found > 10)
    passed = raw_efficiency > 50 and chapters_found > 10

    return GradingResult(
        total_steps=steps,
        total_reward=total_reward,
        chapters_found=chapters_found,
        rate_limit_hits=rate_limit_hits,
        db_updates=db_updates,
        idle_actions=idle_actions,
        check_actions=check_actions,
        avg_reward_per_step=total_reward / max(1, steps),
        efficiency_score=efficiency_score,
        passed=passed,
    )


def run_multiple_trials(
    agent_class: type,
    steps: int = 100,
    trials: int = 5,
    verbose: bool = False
) -> List[GradingResult]:
    """Run multiple evaluation trials with different seeds."""
    results = []

    for trial in range(trials):
        if verbose:
            print(f"\n=== Trial {trial + 1}/{trials} ===")

        agent = agent_class(seed=trial * 100)
        result = run_evaluation(agent, steps=steps, verbose=verbose)
        results.append(result)

    return results


def compute_statistics(results: List[GradingResult]) -> dict:
    """Compute statistics across multiple trials."""
    if not results:
        return {}

    return {
        "trials": len(results),
        "avg_reward": sum(r.total_reward for r in results) / len(results),
        "avg_chapters_found": sum(r.chapters_found for r in results) / len(results),
        "avg_rate_limits": sum(r.rate_limit_hits for r in results) / len(results),
        "avg_efficiency": sum(r.efficiency_score for r in results) / len(results),
        "pass_rate": sum(1 for r in results if r.passed) / len(results),
        "min_reward": min(r.total_reward for r in results),
        "max_reward": max(r.total_reward for r in results),
    }


def print_report(results: List[GradingResult]) -> None:
    """Print a detailed grading report."""
    stats = compute_statistics(results)

    print("\n" + "=" * 60)
    print("MANGA TRACKER ENVIRONMENT - GRADING REPORT")
    print("=" * 60)
    print(f"Total Trials: {stats.get('trials', 0)}")
    print(f"Steps per Trial: {results[0].total_steps if results else 0}")
    print("-" * 60)
    print("Performance Metrics:")
    print(f"  Average Reward: {stats.get('avg_reward', 0):.2f}")
    print(f"  Average Chapters Found: {stats.get('avg_chapters_found', 0):.1f}")
    print(f"  Average Rate Limits: {stats.get('avg_rate_limits', 0):.2f}")
    print(f"  Average Efficiency Score: {stats.get('avg_efficiency', 0):.2f}")
    print("-" * 60)
    print("Results:")
    print(f"  Pass Rate: {stats.get('pass_rate', 0) * 100:.1f}%")
    print(f"  Min Reward: {stats.get('min_reward', 0):.2f}")
    print(f"  Max Reward: {stats.get('max_reward', 0):.2f}")
    print("=" * 60)

    # Individual trial results
    print("\nIndividual Trial Results:")
    print("-" * 60)
    for i, result in enumerate(results):
        status = "PASS" if result.passed else "FAIL"
        print(f"Trial {i + 1}: [{status}] "
              f"Reward={result.total_reward:.2f}, "
              f"Chapters={result.chapters_found}, "
              f"Efficiency={result.efficiency_score:.2f}")
    print("=" * 60)


def main():
    """Main entry point for the grader."""
    parser = argparse.ArgumentParser(
        description="Grade MangaTracker agent performance"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps per evaluation (default: 100)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of evaluation trials (default: 5)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["simple", "random"],
        default="simple",
        help="Agent type to evaluate (default: simple)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step logs"
    )

    args = parser.parse_args()

    # Select agent class
    agent_class = SimpleAgent if args.agent == "simple" else RandomAgent

    print(f"Evaluating {args.agent} agent over {args.steps} steps, {args.trials} trials...")

    # Run evaluation
    results = run_multiple_trials(
        agent_class,
        steps=args.steps,
        trials=args.trials,
        verbose=args.verbose
    )

    # Print report
    print_report(results)

    # Output single aggregated score for OpenEnv parser
    # Format: [STEP] task_name score action=...
    stats = compute_statistics(results)
    avg_efficiency = stats.get("avg_efficiency", 0.5)
    # Ensure strictly bounded to (0, 1)
    avg_efficiency = max(0.01, min(0.99, avg_efficiency))
    print(f"[STEP] MangaTracker_Evaluation {avg_efficiency:.6f} action=COMPLETE trials={len(results)} pass_rate={stats.get('pass_rate', 0):.2f}")
    print("[END] MangaTracker_Evaluation")

    # Exit with appropriate code
    all_passed = all(r.passed for r in results)
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
