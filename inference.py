"""
Inference script for MangaTracker Environment.
This standalone script demonstrates how an agent interacts with the environment to satisfy the OpenEnv grader requirements.
"""

import time
import random
import os

from client import MangaTrackerClient  # type: ignore
from models import ActionType, MangaTrackerAction  # type: ignore

def inference():
    """Run a random agent against the Manga Tracker environment."""
    try:
        # Read OPENENV_HOST from environment variable, default to localhost
        host = os.environ.get("OPENENV_HOST", "http://localhost:8000")
        print(f"Connecting to environment at {host}")

        # Adding retries to give the server time to start up if running in Docker/Grader
        success = False
        for attempt in range(5):
            try:
                with MangaTrackerClient(base_url=host).sync() as client:
                    print("Connected! Resetting environment...")
                    result = client.reset()

                    if not result or not result.observation:
                        raise ValueError("Failed to retrieve valid observation")

                    watchlist = result.observation.state.watchlist
                    print(f"Initial watchlist size: {len(watchlist)}")

                    # Take exactly 5 randomized demonstration steps
                    for step_num in range(5):
                        action_type = random.choice([ActionType.CHECK_SOURCE, ActionType.UPDATE_DB, ActionType.IDLE])
                        manga_idx = random.randint(0, len(watchlist) - 1) if watchlist else 0
                        check_all = random.random() < 0.2

                        action = MangaTrackerAction(
                            action_type=action_type,
                            manga_index=manga_idx,
                            check_all=bool(check_all)
                        )

                        print(f"Step {step_num + 1}: Executing {action_type.name}...")
                        result = client.step(action)
                        print(f"  -> Reward: {result.reward}, New Chapters: {result.observation.new_chapters_found}")

                        if result.done:
                            print("Episode reached terminal state!")
                            break

                        time.sleep(0.5)

                print("\nInference sequence completed successfully.")
                success = True
                break

            except Exception as e:
                print(f"Failed to connect or execute (attempt {attempt+1}/5): {e}")
                time.sleep(2)

        if not success:
            print("Inference failed after all retries.")
    except Exception as e:
        print(f"Unhandled exception in inference: {e}")
        raise

if __name__ == "__main__":
    inference()
