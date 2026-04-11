"""
Tests for the SQL Debug Environment.
Run with: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import SQLDebugAction, SQLDebugObservation, SQLDebugState
from server.environment import SQLDebugEnvironment
from tasks import TASKS, grade_submission


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = SQLDebugEnvironment(task_name="easy_syntax_fix")
    yield env
    env.close()

@pytest.fixture
def medium_env():
    env = SQLDebugEnvironment(task_name="medium_logic_fix")
    yield env
    env.close()

@pytest.fixture
def hard_env():
    env = SQLDebugEnvironment(task_name="hard_optimization_fix")
    yield env
    env.close()


# ─── Basic API Tests ──────────────────────────────────────────────────────────

class TestOpenEnvInterface:
    """Verify OpenEnv spec compliance: reset/step/state."""

    def test_reset_returns_observation(self, easy_env):
        result = easy_env.reset()
        assert result.observation is not None
        obs = result.observation
        assert isinstance(obs, SQLDebugObservation)
        assert obs.task_id == "easy_syntax_fix"
        assert obs.task_difficulty == "easy"
        assert obs.broken_sql != ""
        assert obs.error_message != ""
        assert len(obs.db_schema) > 0

    def test_state_returns_metadata(self, easy_env):
        easy_env.reset()
        state = easy_env.state()
        assert isinstance(state, SQLDebugState)
        assert state.task_name == "easy_syntax_fix"
        assert state.current_step == 0
        assert state.done == False
        assert state.total_reward == 0.0

    def test_step_returns_valid_reward(self, easy_env):
        easy_env.reset()
        action = SQLDebugAction(corrected_sql="SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC")
        result = easy_env.step(action)
        assert 0.0 <= result.reward <= 1.0
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_increments_counter(self, easy_env):
        easy_env.reset()
        action = SQLDebugAction(corrected_sql="SELECT 1")
        easy_env.step(action)
        state = easy_env.state()
        assert state.current_step == 1

    def test_done_after_correct_answer(self, easy_env):
        easy_env.reset()
        correct_sql = TASKS["easy_syntax_fix"]["fixed_sql"]
        action = SQLDebugAction(corrected_sql=correct_sql)
        result = easy_env.step(action)
        assert result.done == True
        assert result.reward > 0.7

    def test_done_after_max_steps(self, easy_env):
        easy_env.reset()
        max_steps = TASKS["easy_syntax_fix"]["max_steps"]
        for _ in range(max_steps):
            action = SQLDebugAction(corrected_sql="SELECT 1 FROM employees")
            result = easy_env.step(action)
        assert result.done == True

    def test_error_on_step_after_done(self, easy_env):
        easy_env.reset()
        correct_sql = TASKS["easy_syntax_fix"]["fixed_sql"]
        easy_env.step(SQLDebugAction(corrected_sql=correct_sql))
        with pytest.raises(RuntimeError):
            easy_env.step(SQLDebugAction(corrected_sql="SELECT 1"))

    def test_reset_clears_state(self, easy_env):
        easy_env.reset()
        easy_env.step(SQLDebugAction(corrected_sql="SELECT 1"))
        easy_env.reset()
        state = easy_env.state()
        assert state.current_step == 0
        assert state.done == False
        assert state.total_reward == 0.0


# ─── Grader Tests ─────────────────────────────────────────────────────────────

class TestGrader:
    """Verify grader scores are in [0, 1] and deterministic."""

    def _setup_db(self, task_name: str) -> str:
        """Create a temp DB for testing."""
        import sqlite3
        import tempfile
        task = TASKS[task_name]
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        for stmt in task["create_sql"].split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
        conn.close()
        return path

    def test_wrong_sql_low_reward(self):
        db_path = self._setup_db("easy_syntax_fix")
        result = grade_submission("easy_syntax_fix", "SELECT 1", db_path, 1, 5)
        assert 0.0 <= result["reward"] <= 1.0
        assert result["results_match"] == False
        os.unlink(db_path)

    def test_syntax_error_very_low_reward(self):
        db_path = self._setup_db("easy_syntax_fix")
        result = grade_submission("easy_syntax_fix", "SELCT * FORM employees", db_path, 1, 5)
        assert result["reward"] < 0.15
        assert result["syntax_valid"] == False
        os.unlink(db_path)

    def test_correct_sql_high_reward(self):
        db_path = self._setup_db("easy_syntax_fix")
        correct = TASKS["easy_syntax_fix"]["fixed_sql"]
        result = grade_submission("easy_syntax_fix", correct, db_path, 1, 5)
        assert result["reward"] >= 0.8
        assert result["results_match"] == True
        os.unlink(db_path)

    def test_reward_in_range_for_all_tasks(self):
        for task_name in TASKS:
            db_path = self._setup_db(task_name)
            correct = TASKS[task_name]["fixed_sql"]
            result = grade_submission(task_name, correct, db_path, 1, 5)
            assert 0.0 <= result["reward"] <= 1.0, f"Reward out of range for {task_name}"
            assert result["results_match"] == True, f"Correct SQL should match for {task_name}"
            os.unlink(db_path)

    def test_grader_deterministic(self):
        """Same input should produce same score every time."""
        db_path = self._setup_db("medium_logic_fix")
        sql = TASKS["medium_logic_fix"]["fixed_sql"]
        results = [
            grade_submission("medium_logic_fix", sql, db_path, 1, 5)
            for _ in range(3)
        ]
        rewards = [r["reward"] for r in results]
        assert len(set(rewards)) == 1, f"Grader not deterministic: {rewards}"
        os.unlink(db_path)

    def test_attempt_penalty_applies(self):
        """Later attempts should yield slightly less reward for same SQL."""
        db_path = self._setup_db("easy_syntax_fix")
        sql = "SELECT name FROM employees"  # executes but doesn't fully match
        r1 = grade_submission("easy_syntax_fix", sql, db_path, 1, 5)["reward"]
        r3 = grade_submission("easy_syntax_fix", sql, db_path, 3, 5)["reward"]
        assert r3 <= r1, "Attempt penalty should reduce reward"
        os.unlink(db_path)


# ─── Task Difficulty Tests ────────────────────────────────────────────────────

class TestTaskDifficulty:
    """Ensure tasks have appropriate difficulty ordering."""

    def test_all_tasks_present(self):
        assert "easy_syntax_fix" in TASKS
        assert "medium_logic_fix" in TASKS
        assert "hard_optimization_fix" in TASKS

    def test_easy_has_fewer_max_steps(self):
        assert TASKS["easy_syntax_fix"]["max_steps"] <= TASKS["medium_logic_fix"]["max_steps"]

    def test_hard_has_most_errors(self):
        # Hard task should have more schema tables
        assert len(TASKS["hard_optimization_fix"]["schema"]) >= len(TASKS["easy_syntax_fix"]["schema"])

    def test_each_task_has_required_fields(self):
        required = ["name", "difficulty", "broken_sql", "fixed_sql", "error_message",
                    "schema", "create_sql", "expected_result_description", "max_steps"]
        for task_name, task in TASKS.items():
            for field in required:
                assert field in task, f"Task {task_name} missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
