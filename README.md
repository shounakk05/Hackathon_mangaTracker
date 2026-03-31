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
   cd Hackathon_mangaTracker/manga_tracker
   ```

2. **Run the evaluation grader:**
   ```bash
   uv run python grader.py
   ```

3. **Start the FastAPI Server:**
   ```bash
   uv run --project . server
   ```

## 📖 Complete Documentation

The core environment logic, detailed API references, action spaces, observation spaces, and reward functions are fully documented inside the main package folder. 

👉 **[View Full Environment Documentation](./manga_tracker/README.md)**

## 🛠️ Project Structure

```text
Hackathon_MangaTracker/
├── manga_tracker/            # Main OpenEnv package
│   ├── server/               # FastAPI application backend
│   ├── client.py             # WebSocket/HTTP RL client
│   ├── environment_spec.yaml # OpenEnv environment specifications
│   ├── grader.py             # Evaluation and grading scripts
│   ├── models.py             # Pydantic data models
│   └── README.md             # Detailed technical documentation
├── test_gym_api.py           # Gymnasium API test script
└── README.md                 # This file
```

## 📜 License

BSD-style license - see LICENSE file in the repository root for details.
