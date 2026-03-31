---
title: MangaTracker Environment
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
  - reinforcement-learning
  - agentic-workflows
---

# MangaTracker Environment

A reinforcement learning environment for simulating web-scraping task management. The agent maintains a watchlist of 10 manga titles, checking sources for new chapters while avoiding rate limits.

## Overview

MangaTracker is an OpenEnv-compatible RL environment that simulates the challenges of managing automated web scrapers. The agent must:

- Monitor multiple manga sources for new chapter releases
- Balance thoroughness with rate-limit avoidance
- Maintain an efficient update schedule

This environment is ideal for testing RL algorithms, agent training, and demonstrating autonomous decision-making in resource-constrained scenarios.

## Quick Start

### Using the Python Client

```python
from manga_tracker import MangaTrackerClient, ActionType, MangaTrackerAction

# Connect to a running server
with MangaTrackerClient(base_url="http://localhost:8000") as client:
    # Reset the environment
    result = client.reset()
    print(f"Watchlist: {[m.title for m in result.observation.state.watchlist]}")

    # Take an action - check a specific manga source
    action = MangaTrackerAction(
        action_type=ActionType.CHECK_SOURCE,
        manga_index=0,
        check_all=False
    )
    result = client.step(action)
    print(f"Reward: {result.reward}, Chapters found: {result.observation.new_chapters_found}")
```

### Running the Server

```bash
# Development mode
uvicorn manga_tracker.server.app:app --reload --host 0.0.0.0 --port 8000

# Using uv
uv run --project . server

# Docker
docker run -d --name manga-tracker -p 8000:8000 manga_tracker-env:latest
```

## Installation

```bash
# Install with uv
uv pip install openenv-manga-tracker

# Or from source
cd manga_tracker
uv sync
```

## Action Space

| Action | Parameters | Description |
|--------|------------|-------------|
| `CheckSource` | `manga_index: int`, `check_all: bool` | Check a manga source for new chapters. Use `check_all=true` to scan all sources (higher rate-limit risk). |
| `Idle` | None | Skip this step. Incurs a small penalty (-1) to encourage active engagement. |
| `UpdateDB` | None | Update the database with newly found chapters. Rewards +10 per update. |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `state.watchlist` | `List[MangaEntry]` | List of tracked manga with their current status |
| `state.total_chapters_found` | `int` | Cumulative chapters discovered across all sessions |
| `state.rate_limit_hits` | `int` | Number of 429 errors encountered |
| `state.db_updates_count` | `int` | Number of database updates performed |
| `new_chapters_found` | `int` | Chapters discovered in this step |
| `rate_limited` | `bool` | Whether the action triggered rate limiting |
| `reward` | `float` | Reward for this step |
| `done` | `bool` | Whether the episode has terminated |

### MangaEntry Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Manga title |
| `last_chapter_seen` | `int` | Last chapter the agent has processed |
| `latest_available_chapter` | `int` | Latest chapter available from the source |
| `source_health` | `SourceHealth` | Health status: `healthy`, `degraded`, `unreliable`, `down` |

## Reward Function

| Event | Reward |
|-------|--------|
| New chapter found | +100 |
| Rate limit (429 error) | -50 |
| Idle action | -1 |
| Database update | +10 |

## Source Health Levels

| Health | Chapter Release Probability | Check Success Rate |
|--------|----------------------------|-------------------|
| `healthy` | 30% per step | 100% |
| `degraded` | 15% per step | 80% |
| `unreliable` | 5% per step | 50% |
| `down` | 0% | 0% |

## HTTP API Reference

```bash
# Reset the environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "CheckSource", "manga_index": 0, "check_all": false}'

# Get current state
curl http://localhost:8000/state

# Get action/observation schemas
curl http://localhost:8000/schema

# View interactive API docs
curl http://localhost:8000/docs
```

## Grading

Run the built-in grader to evaluate agent performance:

```bash
cd manga_tracker
uv run python grader.py
```

The grader runs 100-step trials and reports:
- Pass rate (episodes with positive reward)
- Average reward per trial
- Average chapters found
- Efficiency score

## Deployment

### Hugging Face Spaces

Deploy to Hugging Face Spaces using the OpenEnv CLI:

```bash
openenv push
```

### Manual Docker Deployment

```bash
# Build the image
docker build -t manga_tracker-env:latest -f server/Dockerfile .

# Run locally
docker run -d --name manga-tracker -p 8000:8000 manga_tracker-env:latest

# Push to registry
docker tag manga_tracker-env:latest <registry>/manga-tracker:latest
docker push <registry>/manga-tracker:latest
```

## Project Structure

```
manga_tracker/
├── __init__.py           # Package exports
├── client.py             # WebSocket/HTTP client
├── models.py             # Pydantic data models
├── grader.py             # Evaluation scripts
├── server/
│   ├── app.py            # FastAPI application
│   ├── environment.py    # Core environment logic
│   └── Dockerfile        # Container build config
├── pyproject.toml        # Project dependencies
├── openenv.yaml          # OpenEnv deployment config
└── README.md             # This file
```

## License

BSD-style license - see LICENSE file in the repository root.
