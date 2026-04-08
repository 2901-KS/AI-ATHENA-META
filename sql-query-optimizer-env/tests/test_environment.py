"""
pytest test suite for SQL Query Optimizer OpenEnv.

Run: pytest tests/ -v
"""

import pytest
import sys
sys.path.insert(0, ".")

from env.environment import SQLQueryOptimizerEnv
from env.models import Action, Observation, Reward, EpisodeResult
from env.tasks.definitions import ALL_TASKS, EASY_TASK, MEDIUM_TASK, HARD_TASK
from graders import EasyGrader, MediumGrader, HardGrader, run_all_graders


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    env.reset()
    return env

@pytest.fixture
def medium_env():
    env = SQLQueryOptimizerEnv(difficulty="medium", seed=42)
    env.reset()
    return env

@pytest.fixture
def hard_env():
    env = SQLQueryOptimizerEnv(difficulty="hard", seed=42)
    env.reset()
    return env

@pytest.fixture
def reset_obs():
    env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
    return env, env.reset()


# ──────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────

class TestTaskDefinitions:
    def test_all_tasks_present(self):
        assert set(ALL_TASKS.keys()) == {"easy", "medium", "hard"}

    def test_task_fields(self):
        for diff, task in ALL_TASKS.items():
            assert task.task_id
            assert task.difficulty == diff
            assert task.baseline_query.strip().upper().startswith("SELECT")
            assert len(task.hints) >= 2
            assert task.target_speedup > 1.0
            assert task.max_steps >= 5

    def test_difficulty_ordering(self):
        assert EASY_TASK.target_speedup < MEDIUM_TASK.target_speedup < HARD_TASK.target_speedup
        assert EASY_TASK.max_steps <= MEDIUM_TASK.max_steps <= HARD_TASK.max_steps


# ──────────────────────────────────────────────
# Environment lifecycle
# ──────────────────────────────────────────────

class TestEnvironmentLifecycle:
    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            SQLQueryOptimizerEnv(difficulty="impossible")

    def test_reset_returns_observation(self, reset_obs):
        _, obs = reset_obs
        assert isinstance(obs, Observation)

    def test_reset_initializes_step_zero(self, reset_obs):
        _, obs = reset_obs
        assert obs.step_number == 0

    def test_reset_not_done(self, reset_obs):
        _, obs = reset_obs
        assert not obs.done

    def test_reset_has_baseline_metrics(self, reset_obs):
        _, obs = reset_obs
        assert obs.baseline_metrics.execution_time_ms > 0
        assert obs.baseline_metrics.result_hash != ""

    def test_double_reset_clean_state(self):
        env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
        obs1 = env.reset()
        env.reset()
        obs2 = env.reset()
        assert obs2.step_number == 0
        assert obs2.cumulative_reward == 0.0
        assert obs2.baseline_metrics.result_hash == obs1.baseline_metrics.result_hash

    def test_step_after_done_raises(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=True)
        easy_env.step(action)
        with pytest.raises(RuntimeError):
            easy_env.step(action)


# ──────────────────────────────────────────────
# Step API
# ──────────────────────────────────────────────

class TestStepAPI:
    def test_step_returns_tuple(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        result = easy_env.step(action)
        assert len(result) == 4

    def test_step_increments_step_number(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        obs, _, _, _ = easy_env.step(action)
        assert obs.step_number == 1

    def test_step_returns_typed_models(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        obs, reward, done, info = easy_env.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_declare_done_ends_episode(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=True)
        _, _, done, _ = easy_env.step(action)
        assert done is True

    def test_invalid_sql_no_crash(self, easy_env):
        action = Action(optimized_query="INVALID SQL !!!", declare_done=False)
        obs, reward, done, info = easy_env.step(action)
        assert reward.total < 0
        assert not obs.last_action_valid

    def test_non_select_rejected(self, easy_env):
        action = Action(optimized_query="DROP TABLE customers", declare_done=False)
        obs, reward, _, _ = easy_env.step(action)
        assert not obs.last_action_valid

    def test_max_steps_terminates(self):
        env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
        env.reset()
        env._task.max_steps = 1
        action = Action(optimized_query=env._current_query, declare_done=False)
        _, _, done, _ = env.step(action)
        assert done is True


# ──────────────────────────────────────────────
# Reward signal
# ──────────────────────────────────────────────

class TestRewardSignal:
    def test_reward_in_range(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        _, reward, _, _ = easy_env.step(action)
        assert -1.0 <= reward.total <= 1.0

    def test_correct_query_positive_correctness(self, easy_env):
        # Same query as baseline → result is correct
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        _, reward, _, _ = easy_env.step(action)
        assert reward.is_correct
        assert reward.correctness_score > 0

    def test_invalid_sql_negative_reward(self, easy_env):
        action = Action(optimized_query="BAD", declare_done=False)
        _, reward, _, _ = easy_env.step(action)
        assert reward.total < 0

    def test_hint_penalty_applied(self, easy_env):
        action = Action(
            optimized_query=easy_env._current_query,
            request_hint=True,
            declare_done=False,
        )
        _, reward, _, _ = easy_env.step(action)
        assert reward.hint_penalty < 0

    def test_step_penalty_always_applied(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        _, reward, _, _ = easy_env.step(action)
        assert reward.step_penalty < 0

    def test_optimized_query_better_reward(self):
        env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
        obs = env.reset()

        # Baseline query
        action_base = Action(optimized_query=obs.current_query, declare_done=False)
        _, reward_base, _, _ = env.step(action_base)
        env.reset()

        # Optimized CTE query
        optimized = """WITH order_totals AS (
            SELECT customer_id, SUM(total_amount) AS total_spend
            FROM orders
            WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01'
              AND status = 'delivered'
            GROUP BY customer_id
        )
        SELECT c.customer_id, c.name, c.country, ot.total_spend
        FROM customers c
        JOIN order_totals ot ON ot.customer_id = c.customer_id
        ORDER BY ot.total_spend DESC LIMIT 10"""

        action_opt = Action(optimized_query=optimized, declare_done=True)
        _, reward_opt, _, _ = env.step(action_opt)

        assert reward_opt.is_correct
        assert reward_opt.speedup_ratio > 1.0


# ──────────────────────────────────────────────
# State API
# ──────────────────────────────────────────────

class TestStateAPI:
    def test_state_returns_dict(self, easy_env):
        assert isinstance(easy_env.state(), dict)

    def test_state_has_required_keys(self, easy_env):
        s = easy_env.state()
        for key in ["difficulty", "task_id", "step_number", "max_steps",
                    "current_query", "cumulative_reward", "done"]:
            assert key in s, f"Missing key: {key}"

    def test_state_reflects_current_step(self, easy_env):
        action = Action(optimized_query=easy_env._current_query, declare_done=False)
        easy_env.step(action)
        assert easy_env.state()["step_number"] == 1


# ──────────────────────────────────────────────
# Hint system
# ──────────────────────────────────────────────

class TestHintSystem:
    def test_hint_returns_text(self, easy_env):
        action = Action(
            optimized_query=easy_env._current_query,
            request_hint=True,
            declare_done=False,
        )
        obs, _, _, _ = easy_env.step(action)
        assert obs.hint is not None
        assert len(obs.hint) > 10

    def test_sequential_hints_differ(self, easy_env):
        hints = []
        for _ in range(3):
            action = Action(
                optimized_query=easy_env._current_query,
                request_hint=True,
                declare_done=False,
            )
            obs, _, _, _ = easy_env.step(action)
            if obs.hint:
                hints.append(obs.hint)
        # At least the first two hints should be populated
        assert len(hints) >= 1


# ──────────────────────────────────────────────
# Graders
# ──────────────────────────────────────────────

class TestGraders:
    def _get_baseline(self, difficulty):
        env = SQLQueryOptimizerEnv(difficulty=difficulty, seed=42)
        obs = env.reset()
        return obs.current_query

    def test_easy_grader_score_in_range(self):
        q = self._get_baseline("easy")
        result = EasyGrader(seed=42).grade(q)
        assert 0.0 <= result["score"] <= 1.0

    def test_medium_grader_score_in_range(self):
        q = self._get_baseline("medium")
        result = MediumGrader(seed=42).grade(q)
        assert 0.0 <= result["score"] <= 1.0

    def test_hard_grader_score_in_range(self):
        q = self._get_baseline("hard")
        result = HardGrader(seed=42).grade(q)
        assert 0.0 <= result["score"] <= 1.0

    def test_grader_has_required_fields(self):
        q = self._get_baseline("easy")
        result = EasyGrader(seed=42).grade(q)
        for key in ["score", "task_id", "difficulty", "breakdown", "metrics", "passed"]:
            assert key in result, f"Missing: {key}"

    def test_run_all_graders(self):
        easy_q = self._get_baseline("easy")
        med_q = self._get_baseline("medium")
        hard_q = self._get_baseline("hard")
        result = run_all_graders(easy_q, med_q, hard_q, seed=42)
        assert "aggregate_score" in result
        assert 0.0 <= result["aggregate_score"] <= 1.0

    def test_grader_deterministic(self):
        q = self._get_baseline("easy")
        r1 = EasyGrader(seed=42).grade(q)
        r2 = EasyGrader(seed=42).grade(q)
        assert r1["score"] == r2["score"]

    def test_optimized_scores_higher_than_baseline(self):
        env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
        obs = env.reset()
        baseline_q = obs.current_query

        optimized_q = """WITH order_totals AS (
            SELECT customer_id, SUM(total_amount) AS total_spend
            FROM orders
            WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01'
              AND status = 'delivered'
            GROUP BY customer_id
        )
        SELECT c.customer_id, c.name, c.country, ot.total_spend
        FROM customers c
        JOIN order_totals ot ON ot.customer_id = c.customer_id
        ORDER BY ot.total_spend DESC LIMIT 10"""

        baseline_result = EasyGrader(seed=42).grade(baseline_q)
        optimized_result = EasyGrader(seed=42).grade(optimized_q)
        assert optimized_result["score"] >= baseline_result["score"]


# ──────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_baseline_hash(self):
        hashes = []
        for _ in range(3):
            env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
            obs = env.reset()
            hashes.append(obs.baseline_metrics.result_hash)
        assert len(set(hashes)) == 1

    def test_same_seed_same_execution_time_range(self):
        times = []
        for _ in range(2):
            env = SQLQueryOptimizerEnv(difficulty="easy", seed=42)
            obs = env.reset()
            times.append(obs.baseline_metrics.execution_time_ms)
        # Times should be in the same ballpark (within 10x of each other)
        assert max(times) / min(times) < 10
