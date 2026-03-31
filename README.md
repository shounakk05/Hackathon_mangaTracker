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

# 📚 MangaTracker - Meta OpenEnv Hackathon

Welcome to **MangaTracker**, a reinforcement learning environment designed for simulating web-scraping task management. This project was built as a submission for the Meta OpenEnv Hackathon.

## 🌟 Project Overview

MangaTracker is an OpenEnv-compatible RL environment that simulates the challenges of managing automated web scrapers. The AI agent acts as a task manager that must:
- Monitor a watchlist of 10 manga titles across multiple sources for new chapter releases.
- Balance the thoroughness of checking for updates with rate-limit avoidance (handling 429 errors).
- Maintain an efficient schedule for updating a database without getting blocked or missing releases.

This environment acts as an excellent benchmark for testing RL algorithms, demonstrating autonomous decision-making and optimal resource scheduling under strict API constraints.

## 🚀 Quick Start

To quickly get started with evaluating the agent or running the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shounakk05/Hackathon_mangaTracker.git
   cd Hackathon_mangaTracker
   ```

2. **Run the evaluation grader:**
   ```bash
   uv run python grader.py
   ```

3. **Start the FastAPI Server:**
   ```bash
   uv run --project . server
   ```

## 🎮 Action Space

| Action | Parameters | Description |
|--------|------------|-------------|
| `CheckSource` | `manga_index: int`, `check_all: bool` | Check a manga source for new chapters. Use `check_all=true` to scan all sources (higher rate-limit risk). |
| `Idle` | None | Skip this step. Incurs a small penalty (-1) to encourage active engagement. |
| `UpdateDB` | None | Update the database with newly found chapters. Rewards +10 per update. |

## 👁️ Observation Space

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

## 💰 Reward Function

| Event | Reward |
|-------|--------|
| New chapter found | +100 |
| Rate limit (429 error) | -50 |
| Idle action | -1 |
| Database update | +10 |

## 🌐 HTTP API Reference

```bash
# Reset the environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "CheckSource", "manga_index": 0, "check_all": false}'

# Get current state
curl http://localhost:8000/state
```

## 🛠️ Deployment

### Hugging Face Spaces
Deploy to Hugging Face Spaces using the OpenEnv CLI:
```bash
openenv push
```

### Manual Docker Deployment
```bash
docker build -t manga_tracker-env:latest -f Dockerfile .
docker run -d --name manga-tracker -p 8000:8000 manga_tracker-env:latest
```

## 📜 License
BSD-style license - see LICENSE file in the repository root for details.
