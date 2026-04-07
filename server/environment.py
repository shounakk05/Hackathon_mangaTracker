# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Manga Tracker Environment Implementation.

A web-scraping simulation environment where the agent manages a manga watchlist,
checking sources for new chapters while avoiding rate limits.

Reward Function:
    +100 for finding new chapters
    -50 for 429 rate-limit errors
    -1 for idle actions (encourages active engagement)
    +10 for successful database updates
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionType,
        MangaEntry,
        MangaTrackerAction,
        MangaTrackerObservation,
        MangaTrackerState,
        SourceHealth,
    )
except ImportError:
    from models import (  # type: ignore
        ActionType,
        MangaEntry,
        MangaTrackerAction,
        MangaTrackerObservation,
        MangaTrackerState,
        SourceHealth,
    )


# Initial manga watchlist data
INITIAL_MANGA_LIST: List[Dict] = [
    {"title": "One Piece", "last_chapter": 1100, "health": SourceHealth.HEALTHY},
    {"title": "Jujutsu Kaisen", "last_chapter": 245, "health": SourceHealth.HEALTHY},
    {"title": "Chainsaw Man", "last_chapter": 150, "health": SourceHealth.DEGRADED},
    {"title": "My Hero Academia", "last_chapter": 410, "health": SourceHealth.HEALTHY},
    {"title": "Demon Slayer", "last_chapter": 205, "health": SourceHealth.UNRELIABLE},
    {"title": "Attack on Titan", "last_chapter": 139, "health": SourceHealth.HEALTHY},
    {"title": "Tokyo Revengers", "last_chapter": 278, "health": SourceHealth.DEGRADED},
    {"title": "Spy x Family", "last_chapter": 90, "health": SourceHealth.HEALTHY},
    {"title": "Kaiju No. 8", "last_chapter": 100, "health": SourceHealth.HEALTHY},
    {"title": "Blue Lock", "last_chapter": 250, "health": SourceHealth.UNRELIABLE},
]

# Reward constants
REWARD_NEW_CHAPTER = 100.0
REWARD_RATE_LIMIT = -50.0
REWARD_IDLE = -1.0
REWARD_DB_UPDATE = 10.0


class MangaTrackerEnvironment(Environment):
    """
    Manga Tracker RL Environment.

    Simulates a web-scraping task where the agent manages a watchlist of 10 manga titles.
    Each manga has 'Last Chapter Seen' and 'Source Health' metrics.

    Action Space:
        - CheckSource: Check a manga source for new chapters
        - Idle: Skip this step (small penalty)
        - UpdateDB: Update the database with found chapters

    The environment features:
        - Dynamic chapter releases based on source health
        - Rate limiting (429 errors) when checking too many sources
        - Custom reward function (+100 for new chapters, -50 for rate limits)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the MangaTracker environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._manga_state: Optional[MangaTrackerState] = None
        self._rate_limit_counter: int = 0
        self._rate_limit_window: List[int] = []
        self._chapter_release_rng = random.Random()

    def _initialize_watchlist(self) -> List[MangaEntry]:
        """Create initial watchlist with 10 manga entries."""
        watchlist = []
        for manga_data in INITIAL_MANGA_LIST:
            entry = MangaEntry(
                title=manga_data["title"],
                last_chapter_seen=manga_data["last_chapter"],
                latest_available_chapter=manga_data["last_chapter"],
                source_health=manga_data["health"],
            )
            watchlist.append(entry)
        return watchlist

    def _simulate_chapter_releases(self) -> None:
        """Simulate new chapter releases based on source health."""
        if self._manga_state is None:
            return

        for entry in self._manga_state.watchlist:
            # Probability of new chapter based on source health
            health_probs = {
                SourceHealth.HEALTHY: 0.3,
                SourceHealth.DEGRADED: 0.15,
                SourceHealth.UNRELIABLE: 0.05,
                SourceHealth.DOWN: 0.0,
            }
            prob = health_probs.get(entry.source_health, 0.0)

            if self._chapter_release_rng.random() < prob:
                entry.latest_available_chapter += 1

    def _check_rate_limit(self, check_all: bool) -> bool:
        """
        Check if the action would trigger rate limiting.

        Rate limiting is designed to encourage smart agent behavior:
        - Checking individual sources is safe
        - Checking ALL sources at once has a chance of rate limiting
        - Consecutive check_all actions increase rate limit probability

        Returns True if rate limited, False otherwise.
        """
        if not check_all:
            # Individual source checks are never rate limited
            return False

        # check_all has a base 30% chance of rate limiting
        # This increases with consecutive check_all actions
        base_rate = 0.3
        consecutive_bonus = min(0.2 * self._rate_limit_counter, 0.5)
        rate_limit_prob = base_rate + consecutive_bonus

        if self._chapter_release_rng.random() < rate_limit_prob:
            self._rate_limit_counter += 1
            return True
        else:
            # Reset counter on successful check_all
            self._rate_limit_counter = 0
            return False

    def _execute_check_source(
        self, action: MangaTrackerAction
    ) -> Tuple[int, bool, str]:
        """
        Execute CheckSource action.

        Returns:
            Tuple of (new_chapters_found, rate_limited, result_message)
        """
        if self._manga_state is None:
            return 0, False, "No watchlist initialized"

        # Check for rate limiting
        if self._check_rate_limit(action.check_all):
            self._manga_state.rate_limit_hits += 1
            return 0, True, "Rate limit (429) - Too many requests"

        new_chapters = 0
        result_parts = []

        if action.check_all:
            # Check all sources
            for idx, entry in enumerate(self._manga_state.watchlist):
                found = self._check_single_source(idx)
                new_chapters += found
                if found > 0:
                    result_parts.append(f"{entry.title}: +{found} chapters")
        elif action.manga_index is not None:
            # Check specific source
            idx = action.manga_index
            if 0 <= idx < len(self._manga_state.watchlist):
                found = self._check_single_source(idx)
                new_chapters += found
                entry = self._manga_state.watchlist[idx]
                if found > 0:
                    result_parts.append(f"{entry.title}: +{found} chapters")
                else:
                    result_parts.append(f"{entry.title}: No new chapters")
            else:
                return 0, False, f"Invalid manga index: {idx}"
        else:
            return 0, False, "No manga index specified for CheckSource"

        if new_chapters > 0:
            self._manga_state.total_chapters_found += new_chapters
            result = f"Found {new_chapters} new chapters! " + "; ".join(result_parts)
        else:
            result = "No new chapters found"

        return new_chapters, False, result

    def _check_single_source(self, index: int) -> int:
        """Check a single manga source for new chapters."""
        if self._manga_state is None:
            return 0

        entry = self._manga_state.watchlist[index]

        # Check if there are pending chapters
        pending = entry.chapters_pending
        if pending <= 0:
            return 0

        # Source health affects reliability of finding chapters
        health_success = {
            SourceHealth.HEALTHY: 1.0,
            SourceHealth.DEGRADED: 0.8,
            SourceHealth.UNRELIABLE: 0.5,
            SourceHealth.DOWN: 0.0,
        }

        if self._chapter_release_rng.random() < health_success[entry.source_health]:
            # Found chapters - update last_chapter_seen
            entry.last_chapter_seen = entry.latest_available_chapter
            return pending

        return 0

    def _execute_idle(self) -> str:
        """Execute Idle action."""
        return "Idle - No action taken"

    def _execute_update_db(self) -> Tuple[int, str]:
        """
        Execute UpdateDB action.

        Returns:
            Tuple of (db_updates_count, result_message)
        """
        if self._manga_state is None:
            return 0, "No watchlist initialized"

        updates = 0
        for entry in self._manga_state.watchlist:
            if entry.chapters_pending > 0:
                updates += 1

        self._manga_state.db_updates_count += updates
        return updates, f"Database updated with {updates} manga entries"

    def _calculate_reward(
        self, new_chapters: int, rate_limited: bool, action_type: ActionType, db_updates: int = 0
    ) -> float:
        """
        Calculate reward for the current step.

        Reward Function:
            +100 for each new chapter found
            -50 for rate-limit (429) errors
            -1 for idle actions
            +10 for each database update
        """
        reward = 0.0

        # Reward for finding new chapters
        reward += new_chapters * REWARD_NEW_CHAPTER

        # Penalty for rate limiting
        if rate_limited:
            reward += REWARD_RATE_LIMIT

        # Small penalty for idle to encourage active engagement
        if action_type == ActionType.IDLE:
            reward += REWARD_IDLE

        # Reward for database updates
        reward += db_updates * REWARD_DB_UPDATE

        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MangaTrackerObservation:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation with the watchlist ready
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._manga_state = MangaTrackerState(
            watchlist=self._initialize_watchlist(),
            total_chapters_found=0,
            rate_limit_hits=0,
            db_updates_count=0,
        )
        self._rate_limit_counter = 0
        self._chapter_release_rng = random.Random()

        return MangaTrackerObservation(
            state=self._manga_state,
            action_result="Environment initialized with 10 manga titles",
            new_chapters_found=0,
            rate_limited=False,
            reward=0.0,
        )

    def step(
        self,
        action: MangaTrackerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MangaTrackerObservation:
        """
        Execute an action in the environment.

        Args:
            action: MangaTrackerAction specifying what to do

        Returns:
            MangaTrackerObservation with results and reward
        """
        self._state.step_count += 1

        # Simulate chapter releases over time
        self._simulate_chapter_releases()

        new_chapters = 0
        rate_limited = False
        db_updates = 0
        result_message = ""

        # Execute the action
        if action.action_type == ActionType.CHECK_SOURCE:
            new_chapters, rate_limited, result_message = self._execute_check_source(
                action
            )
        elif action.action_type == ActionType.IDLE:
            result_message = self._execute_idle()
        elif action.action_type == ActionType.UPDATE_DB:
            db_updates, result_message = self._execute_update_db()
        else:
            result_message = f"Unknown action type: {action.action_type}"

        # Calculate reward
        reward = self._calculate_reward(
            new_chapters, rate_limited, action.action_type, db_updates
        )

        # Update state if we have valid manga state
        if self._manga_state:
            observation = MangaTrackerObservation(
                state=self._manga_state,
                action_result=result_message,
                new_chapters_found=new_chapters,
                rate_limited=rate_limited,
                reward=reward,
            )
        else:
            observation = MangaTrackerObservation(
                state=MangaTrackerState(),
                action_result=result_message,
                new_chapters_found=new_chapters,
                rate_limited=rate_limited,
                reward=reward,
            )

        return observation

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_full_state(self) -> Optional[MangaTrackerState]:
        """Get the full manga tracker state (for debugging/grading)."""
        return self._manga_state
