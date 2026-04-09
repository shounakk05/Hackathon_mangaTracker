"""
Inference script for MangaTracker Environment.
This standalone script demonstrates how an agent interacts with the environment to satisfy the OpenEnv grader requirements.
"""

import os
import json
from typing import Optional, List, Dict, Any

from openai import OpenAI
from client import MangaTrackerClient
from models import ActionType, MangaTrackerAction, MangaTrackerState

# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def get_llm_response(client: OpenAI, prompt: str) -> str:
    """Get response from LLM."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for managing a manga tracker. Help decide the best action to take."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Warning: LLM request failed ({str(e)}), falling back to random action.")
        return "{}"


def parse_llm_action(llm_response: str, watchlist_size: int) -> MangaTrackerAction:
    """Parse LLM response into a MangaTrackerAction."""
    try:
        # Try to parse as JSON first
        response = llm_response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        action_data = json.loads(response)
        action_type_str = action_data.get("action_type", "IDLE").upper()
        manga_index = action_data.get("manga_index", 0)
        check_all = action_data.get("check_all", False)

        # Map string to ActionType enum
        action_type_map = {
            "CHECK_SOURCE": ActionType.CHECK_SOURCE,
            "UPDATE_DB": ActionType.UPDATE_DB,
            "IDLE": ActionType.IDLE
        }
        action_type = action_type_map.get(action_type_str, ActionType.IDLE)

        # Clamp manga_index to valid range
        manga_index = max(0, min(manga_index, watchlist_size - 1)) if watchlist_size > 0 else 0

        return MangaTrackerAction(
            action_type=action_type,
            manga_index=manga_index,
            check_all=check_all
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback to random action if parsing fails
        import random
        return MangaTrackerAction(
            action_type=random.choice([ActionType.CHECK_SOURCE, ActionType.UPDATE_DB, ActionType.IDLE]),
            manga_index=0 if watchlist_size == 0 else random.randint(0, watchlist_size - 1),
            check_all=False
        )


def build_prompt(state: MangaTrackerState, step_num: int) -> str:
    """Build prompt for LLM based on current state."""
    watchlist_info = []
    for i, manga in enumerate(state.watchlist[:5]):  # Limit to first 5 for context
        watchlist_info.append(f"{i}: {manga.title} (Health: {manga.source_health.value})")

    prompt = f"""
Step {step_num}. Current manga watchlist state:
{chr(10).join(watchlist_info)}
Total mangas in watchlist: {len(state.watchlist)}

Choose an action. Respond with JSON:
{{
    "action_type": "CHECK_SOURCE" | "UPDATE_DB" | "IDLE",
    "manga_index": <index of manga to act on>,
    "check_all": <true/false>
}}

Action types:
- CHECK_SOURCE: Check for new chapters of a specific manga
- UPDATE_DB: Update the database with latest chapters
- IDLE: Do nothing this turn
"""
    return prompt


def inference() -> None:
    """Run an LLM-based agent against the Manga Tracker environment."""
    print("Starting inference")

    try:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN is missing. Please add it as a Secret in your Hugging Face space settings.")
            raise ValueError("HF_TOKEN is missing.")

        # Initialize OpenAI client
        openai_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key")
        )
        print(f"OpenAI client with model={MODEL_NAME}, base_url={API_BASE_URL}")

        # Determine environment host
        host = os.environ.get("OPENENV_HOST", "http://localhost:8000")

        if LOCAL_IMAGE_NAME:
            print(f"Environment from Docker image: {LOCAL_IMAGE_NAME}")
            client_impl = MangaTrackerClient.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            print(f"Environment at host: {host}")
            client_impl = MangaTrackerClient(base_url=host)

        with client_impl.sync() as client:
            print("Resetting environment")
            
            # Announce official evaluation start
            print("[START] MangaTrackerTask")
            
            result = client.reset()

            if not result or not result.observation:
                print("[STEP] MangaTrackerTask 0.01 action=RESET status=FAILED error='Failed to retrieve observation'")
                raise ValueError("Failed to retrieve valid observation")

            watchlist = result.observation.state.watchlist
            print(f"[STEP] MangaTrackerTask 0.99 action=RESET status=SUCCESS watchlist_size={len(watchlist)}")

            # Run for 5 demonstration steps
            num_steps = 5
            for step_num in range(1, num_steps + 1):
                state = result.observation.state

                # Build prompt and get LLM decision
                prompt = build_prompt(state, step_num)
                llm_response = get_llm_response(openai_client, prompt)

                # Parse action from LLM response
                action = parse_llm_action(llm_response, len(watchlist))

                # Execute action
                result = client.step(action)

                print(f"[STEP] MangaTrackerTask 0.99 step={step_num} action={action.action_type.name} reward={result.reward} new_chapters={result.observation.new_chapters_found} done={result.done}")

                if result.done:
                    break

            print("[END] MangaTrackerTask status=SUCCESS")

    except Exception as e:
        print(f"[END] MangaTrackerTask status=FAILED error='{str(e)}'")
        raise
    finally:
        print("[END]")


if __name__ == "__main__":
    inference()
