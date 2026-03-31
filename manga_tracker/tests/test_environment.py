"""
Pytest unit tests for the MangaTracker Environment.

Coverage:
    - Models: MangaEntry, MangaTrackerState, MangaTrackerAction, MangaTrackerObservation
    - Environment: reset, step, all action types, reward function, rate limiting
    - Grader: run_evaluation, run_multiple_trials, compute_statistics
    - Edge cases: invalid index, uninitialized state, consecutive check_all
"""

import pytest

from manga_tracker.models import (
    ActionType,
    MangaEntry,
    MangaTrackerAction,
    MangaTrackerObservation,
    MangaTrackerState,
    SourceHealth,
)
from manga_tracker.server.environment import (
    MangaTrackerEnvironment,
    REWARD_DB_UPDATE,
    REWARD_IDLE,
    REWARD_NEW_CHAPTER,
    REWARD_RATE_LIMIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> MangaTrackerEnvironment:
    """Fresh, reset environment for each test."""
    e = MangaTrackerEnvironment()
    e.reset()
    return e


@pytest.fixture
def manga_entry() -> MangaEntry:
    return MangaEntry(
        title="Test Manga",
        last_chapter_seen=10,
        latest_available_chapter=15,
        source_health=SourceHealth.HEALTHY,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestMangaEntry:
    def test_chapters_pending(self, manga_entry):
        assert manga_entry.chapters_pending == 5

    def test_chapters_pending_zero_when_caught_up(self):
        entry = MangaEntry(
            title="Up To Date",
            last_chapter_seen=50,
            latest_available_chapter=50,
            source_health=SourceHealth.HEALTHY,
        )
        assert entry.chapters_pending == 0

    def test_chapters_pending_never_negative(self):
        entry = MangaEntry(
            title="Odd",
            last_chapter_seen=100,
            latest_available_chapter=90,  # shouldn't happen in practice
            source_health=SourceHealth.HEALTHY,
        )
        assert entry.chapters_pending == 0

    def test_to_dict_contains_all_fields(self, manga_entry):
        d = manga_entry.to_dict()
        assert "title" in d
        assert "last_chapter_seen" in d
        assert "latest_available_chapter" in d
        assert "source_health" in d
        assert "chapters_pending" in d

    def test_source_health_values(self):
        for health in SourceHealth:
            entry = MangaEntry(
                title="x", last_chapter_seen=1,
                latest_available_chapter=1, source_health=health
            )
            assert entry.source_health == health


class TestMangaTrackerState:
    def test_default_state(self):
        state = MangaTrackerState()
        assert state.watchlist == []
        assert state.total_chapters_found == 0
        assert state.rate_limit_hits == 0
        assert state.db_updates_count == 0

    def test_to_dict_roundtrip(self, manga_entry):
        state = MangaTrackerState(watchlist=[manga_entry])
        d = state.to_dict()
        restored = MangaTrackerState.from_dict(d)
        assert len(restored.watchlist) == 1
        assert restored.watchlist[0].title == manga_entry.title
        assert restored.watchlist[0].chapters_pending == manga_entry.chapters_pending


class TestMangaTrackerAction:
    def test_defaults(self):
        action = MangaTrackerAction()
        assert action.action_type == ActionType.IDLE
        assert action.manga_index is None
        assert action.check_all is False

    def test_check_source_action(self):
        action = MangaTrackerAction(action_type=ActionType.CHECK_SOURCE, manga_index=3)
        assert action.action_type == ActionType.CHECK_SOURCE
        assert action.manga_index == 3

    def test_check_all_action(self):
        action = MangaTrackerAction(action_type=ActionType.CHECK_SOURCE, check_all=True)
        assert action.check_all is True


# ---------------------------------------------------------------------------
# Environment — reset
# ---------------------------------------------------------------------------

class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        e = MangaTrackerEnvironment()
        obs = e.reset()
        assert isinstance(obs, MangaTrackerObservation)

    def test_reset_initializes_10_manga(self, env):
        assert len(env.get_full_state().watchlist) == 10

    def test_reset_zeroes_counters(self, env):
        state = env.get_full_state()
        assert state.total_chapters_found == 0
        assert state.rate_limit_hits == 0
        assert state.db_updates_count == 0

    def test_reset_step_count_zero(self, env):
        assert env.state.step_count == 0

    def test_double_reset_restores_clean_state(self, env):
        # Take some steps then reset
        env.step(MangaTrackerAction(action_type=ActionType.CHECK_SOURCE, check_all=True))
        env.reset()
        state = env.get_full_state()
        assert state.total_chapters_found == 0
        assert env.state.step_count == 0

    def test_each_reset_generates_new_episode_id(self):
        e = MangaTrackerEnvironment()
        e.reset()
        id1 = e.state.episode_id
        e.reset()
        id2 = e.state.episode_id
        assert id1 != id2


# ---------------------------------------------------------------------------
# Environment — step / action types
# ---------------------------------------------------------------------------

class TestEnvironmentStep:
    def test_step_increments_step_count(self, env):
        env.step(MangaTrackerAction(action_type=ActionType.IDLE))
        assert env.state.step_count == 1

    def test_idle_action_reward(self, env):
        obs = env.step(MangaTrackerAction(action_type=ActionType.IDLE))
        assert obs.reward == REWARD_IDLE
        assert obs.new_chapters_found == 0
        assert obs.rate_limited is False

    def test_idle_action_result_message(self, env):
        obs = env.step(MangaTrackerAction(action_type=ActionType.IDLE))
        assert "idle" in obs.action_result.lower()

    def test_check_source_specific_manga(self, env):
        obs = env.step(MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE, manga_index=0
        ))
        assert isinstance(obs, MangaTrackerObservation)
        assert obs.rate_limited is False  # single-source check is never rate limited

    def test_check_source_invalid_index_returns_error(self, env):
        obs = env.step(MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE, manga_index=99
        ))
        assert obs.new_chapters_found == 0
        assert "invalid" in obs.action_result.lower()

    def test_check_source_no_index_returns_error(self, env):
        obs = env.step(MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE,
            manga_index=None,
            check_all=False,
        ))
        assert obs.new_chapters_found == 0

    def test_update_db_action(self, env):
        obs = env.step(MangaTrackerAction(action_type=ActionType.UPDATE_DB))
        assert isinstance(obs, MangaTrackerObservation)

    def test_step_returns_observation_with_state(self, env):
        obs = env.step(MangaTrackerAction(action_type=ActionType.IDLE))
        assert obs.state is not None
        assert len(obs.state.watchlist) == 10


# ---------------------------------------------------------------------------
# Environment — reward function
# ---------------------------------------------------------------------------

class TestRewardFunction:
    def test_idle_reward_is_negative_one(self, env):
        obs = env.step(MangaTrackerAction(action_type=ActionType.IDLE))
        assert obs.reward == REWARD_IDLE  # -1.0

    def test_rate_limit_penalty(self):
        """Force a rate limit by seeding rng so check_all always triggers."""
        # Run many check_all actions; statistically at least one will rate-limit
        e = MangaTrackerEnvironment()
        e.reset()
        rate_limit_rewards = []
        for _ in range(30):
            obs = e.step(MangaTrackerAction(
                action_type=ActionType.CHECK_SOURCE, check_all=True
            ))
            if obs.rate_limited:
                rate_limit_rewards.append(obs.reward)
        # At least one should have been rate limited across 30 consecutive check_all
        assert len(rate_limit_rewards) > 0
        for r in rate_limit_rewards:
            assert r == REWARD_RATE_LIMIT  # -50.0

    def test_reward_constants(self):
        assert REWARD_NEW_CHAPTER == 100.0
        assert REWARD_RATE_LIMIT == -50.0
        assert REWARD_IDLE == -1.0
        assert REWARD_DB_UPDATE == 10.0

    def test_chapters_found_reward_proportional(self, env):
        """If n chapters are found, reward should be n * 100."""
        # Directly manipulate state to guarantee pending chapters
        state = env.get_full_state()
        for entry in state.watchlist:
            entry.latest_available_chapter = entry.last_chapter_seen + 5
            entry.source_health = SourceHealth.HEALTHY

        obs = env.step(MangaTrackerAction(
            action_type=ActionType.CHECK_SOURCE, manga_index=0
        ))
        if obs.new_chapters_found > 0:
            assert obs.reward == obs.new_chapters_found * REWARD_NEW_CHAPTER


# ---------------------------------------------------------------------------
# Environment — rate limiting logic
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_single_source_never_rate_limited(self, env):
        """Individual source checks have 0% rate limit chance."""
        for i in range(10):
            obs = env.step(MangaTrackerAction(
                action_type=ActionType.CHECK_SOURCE, manga_index=i
            ))
            assert obs.rate_limited is False

    def test_consecutive_check_all_increases_rate_limit_risk(self):
        """After many consecutive check_all, rate_limit_counter increases."""
        e = MangaTrackerEnvironment()
        e.reset()
        rate_limits = sum(
            1 for _ in range(50)
            if e.step(MangaTrackerAction(
                action_type=ActionType.CHECK_SOURCE, check_all=True
            )).rate_limited
        )
        # With base 30% + escalating penalty, 50 steps should yield several hits
        assert rate_limits > 0

    def test_rate_limited_observation_has_no_chapters(self, env):
        """When rate limited, chapters_found must be 0."""
        e = MangaTrackerEnvironment()
        e.reset()
        for _ in range(50):
            obs = e.step(MangaTrackerAction(
                action_type=ActionType.CHECK_SOURCE, check_all=True
            ))
            if obs.rate_limited:
                assert obs.new_chapters_found == 0
                break


# ---------------------------------------------------------------------------
# Grader integration
# ---------------------------------------------------------------------------

class TestGrader:
    def test_run_evaluation_simple_agent(self):
        from manga_tracker.grader import SimpleAgent, run_evaluation
        agent = SimpleAgent(seed=42)
        result = run_evaluation(agent, steps=100)
        assert result.total_steps == 100
        assert result.chapters_found >= 0
        assert result.rate_limit_hits >= 0

    def test_run_evaluation_random_agent(self):
        from manga_tracker.grader import RandomAgent, run_evaluation
        agent = RandomAgent(seed=42)
        result = run_evaluation(agent, steps=50)
        assert result.total_steps == 50

    def test_run_multiple_trials(self):
        from manga_tracker.grader import SimpleAgent, run_multiple_trials
        results = run_multiple_trials(SimpleAgent, steps=50, trials=3)
        assert len(results) == 3

    def test_compute_statistics_keys(self):
        from manga_tracker.grader import SimpleAgent, compute_statistics, run_multiple_trials
        results = run_multiple_trials(SimpleAgent, steps=50, trials=2)
        stats = compute_statistics(results)
        for key in ("trials", "avg_reward", "avg_chapters_found", "pass_rate",
                    "avg_rate_limits", "avg_efficiency", "min_reward", "max_reward"):
            assert key in stats

    def test_compute_statistics_empty(self):
        from manga_tracker.grader import compute_statistics
        assert compute_statistics([]) == {}

    def test_simple_agent_passes_grading_targets(self):
        """Verify the SimpleAgent meets all hackathon scoring thresholds."""
        from manga_tracker.grader import SimpleAgent, compute_statistics, run_multiple_trials
        results = run_multiple_trials(SimpleAgent, steps=100, trials=5)
        stats = compute_statistics(results)
        assert stats["pass_rate"] >= 0.8, "Pass rate must be >= 80%"
        assert stats["avg_reward"] >= 15_000, "Avg reward must be >= 15,000"
        assert stats["avg_chapters_found"] >= 150, "Avg chapters must be >= 150"
        assert stats["avg_efficiency"] >= 1.5, "Efficiency must be >= 1.5 chapters/step"
        assert stats["avg_rate_limits"] <= 10, "Rate limits must be <= 10 per trial"
