# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Manga Tracker Environment."""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Available actions in the MangaTracker environment."""
    CHECK_SOURCE = "CheckSource"
    IDLE = "Idle"
    UPDATE_DB = "UpdateDB"


class SourceHealth(str, Enum):
    """Source health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    DOWN = "down"


class MangaEntry(BaseModel):
    """Represents a single manga entry in the watchlist."""
    title: str = Field(..., description="Manga title")
    last_chapter_seen: int = Field(..., description="Last chapter the agent has seen")
    latest_available_chapter: int = Field(..., description="Latest chapter available from source")
    source_health: SourceHealth = Field(..., description="Health status of the source")

    @property
    def chapters_pending(self) -> int:
        """Number of chapters pending to be read."""
        return max(0, self.latest_available_chapter - self.last_chapter_seen)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "last_chapter_seen": self.last_chapter_seen,
            "latest_available_chapter": self.latest_available_chapter,
            "source_health": self.source_health.value,
            "chapters_pending": self.chapters_pending,
        }


class MangaTrackerState(BaseModel):
    """State representation for the MangaTracker environment."""
    watchlist: List[MangaEntry] = Field(default_factory=list, description="List of manga being tracked")
    total_chapters_found: int = Field(default=0, description="Total chapters found across all sessions")
    rate_limit_hits: int = Field(default=0, description="Number of 429 rate limit errors encountered")
    db_updates_count: int = Field(default=0, description="Number of database updates performed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            "watchlist": [entry.to_dict() for entry in self.watchlist],
            "total_chapters_found": self.total_chapters_found,
            "rate_limit_hits": self.rate_limit_hits,
            "db_updates_count": self.db_updates_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MangaTrackerState":
        """Create state from dictionary."""
        watchlist = [
            MangaEntry(
                title=item["title"],
                last_chapter_seen=item["last_chapter_seen"],
                latest_available_chapter=item["latest_available_chapter"],
                source_health=SourceHealth(item["source_health"]),
            )
            for item in data.get("watchlist", [])
        ]
        return cls(
            watchlist=watchlist,
            total_chapters_found=data.get("total_chapters_found", 0),
            rate_limit_hits=data.get("rate_limit_hits", 0),
            db_updates_count=data.get("db_updates_count", 0),
        )


class MangaTrackerAction(Action):
    """Action for the MangaTracker environment."""
    action_type: ActionType = Field(default=ActionType.IDLE, description="Type of action to take")
    manga_index: Optional[int] = Field(default=None, description="Index of manga in watchlist to check")
    check_all: bool = Field(default=False, description="Check all sources (may trigger rate limits)")


class MangaTrackerObservation(Observation):
    """Observation from the MangaTracker environment."""
    state: MangaTrackerState = Field(default_factory=MangaTrackerState, description="Current environment state")
    action_result: str = Field(default="", description="Result of the last action taken")
    new_chapters_found: int = Field(default=0, description="Number of new chapters found this step")
    rate_limited: bool = Field(default=False, description="Whether the action was rate limited")
    reward: float = Field(default=0.0, description="Reward received for this step")
