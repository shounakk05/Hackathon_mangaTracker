# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manga Tracker Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import MangaTrackerAction, MangaTrackerObservation, MangaTrackerState  # type: ignore


class MangaTrackerClient(EnvClient[MangaTrackerAction, MangaTrackerObservation, State]):
    """
    Client for the Manga Tracker Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MangaTrackerClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.state.watchlist[0].title)
        ...
        ...     action = MangaTrackerAction(action_type=ActionType.CHECK_SOURCE, manga_index=0)
        ...     result = client.step(action)
        ...     print(f"Reward: {result.reward}, New chapters: {result.observation.new_chapters_found}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MangaTrackerClient.from_docker_image("manga_tracker:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MangaTrackerAction())
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MangaTrackerAction) -> Dict[str, Any]:
        """
        Convert MangaTrackerAction to JSON payload for step message.

        Args:
            action: MangaTrackerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "manga_index": action.manga_index,
            "check_all": action.check_all,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MangaTrackerObservation]:
        """
        Parse server response into StepResult[MangaTrackerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MangaTrackerObservation
        """
        obs_data = payload.get("observation", {})
        state_data = obs_data.get("state", {})

        # Parse the state
        state = MangaTrackerState.from_dict(state_data) if state_data else MangaTrackerState()

        observation = MangaTrackerObservation(
            state=state,
            action_result=obs_data.get("action_result", ""),
            new_chapters_found=obs_data.get("new_chapters_found", 0),
            rate_limited=obs_data.get("rate_limited", False),
            reward=obs_data.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
