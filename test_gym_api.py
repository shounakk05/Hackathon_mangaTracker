#!/usr/bin/env python3
"""Test script to verify the Gymnasium API for MangaTracker."""

from manga_tracker.client import MangaTrackerClient
from manga_tracker.models import ActionType, MangaTrackerAction

def main():
    print("=" * 60)
    print("Testing MangaTracker Gymnasium API")
    print("=" * 60)

    # Connect to the running server using sync wrapper
    with MangaTrackerClient(base_url="http://localhost:8000").sync() as client:
        # Test reset()
        print("\n[1] Testing reset()...")
        result = client.reset()
        print(f"    Observation type: {type(result.observation).__name__}")
        print(f"    Reward: {result.reward}")
        print(f"    Done: {result.done}")
        print(f"    Watchlist size: {len(result.observation.state.watchlist)}")
        print(f"    First manga title: {result.observation.state.watchlist[0].title if result.observation.state.watchlist else 'N/A'}")

        # Test step() with IDLE action
        print("\n[2] Testing step() with IDLE action...")
        idle_action = MangaTrackerAction(action_type=ActionType.IDLE)
        result = client.step(idle_action)
        print(f"    Reward: {result.reward}")
        print(f"    Done: {result.done}")
        print(f"    Action result: {result.observation.action_result}")
        print(f"    New chapters found: {result.observation.new_chapters_found}")

        # Test step() with CHECK_SOURCE action
        print("\n[3] Testing step() with CHECK_SOURCE action...")
        check_action = MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE,
            manga_index=0
        )
        result = client.step(check_action)
        print(f"    Reward: {result.reward}")
        print(f"    Done: {result.done}")
        print(f"    Action result: {result.observation.action_result}")
        print(f"    New chapters found: {result.observation.new_chapters_found}")
        print(f"    Rate limited: {result.observation.rate_limited}")

        # Test step() with CHECK_SOURCE (check_all=True)
        print("\n[4] Testing step() with CHECK_SOURCE (check_all=True)...")
        check_all_action = MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE,
            check_all=True
        )
        result = client.step(check_all_action)
        print(f"    Reward: {result.reward}")
        print(f"    Done: {result.done}")
        print(f"    Action result: {result.observation.action_result}")
        print(f"    New chapters found: {result.observation.new_chapters_found}")
        print(f"    Rate limited: {result.observation.rate_limited}")

        # Test step() with UPDATE_DB action
        print("\n[5] Testing step() with UPDATE_DB action...")
        update_action = MangaTrackerAction(action_type=ActionType.UPDATE_DB)
        result = client.step(update_action)
        print(f"    Reward: {result.reward}")
        print(f"    Done: {result.done}")
        print(f"    Action result: {result.observation.action_result}")
        print(f"    DB updates count: {result.observation.state.db_updates_count}")

        print("\n" + "=" * 60)
        print("All Gymnasium API tests passed successfully!")
        print("=" * 60)

if __name__ == "__main__":
    main()
