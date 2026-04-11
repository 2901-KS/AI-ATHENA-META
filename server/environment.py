"""
SQL Debug Environment — Server-side Environment implementation.
Implements the OpenEnv interface: reset(), step(), state().
"""
import os
import sqlite3
import tempfile
import uuid
from typing import Any, Dict, Optional

# Make sure parent directory is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    SQLDebugAction,
    SQLDebugObservation,
    SQLDebugState,
    StepResult,
    ResetResult,
    TableSchema,
)
from tasks import TASKS, grade_submission


class SQLDebugEnvironment:
    """
    SQL Debugging environment.

    An AI agent receives a broken SQL query, its error message, and a DB schema.
    The agent must submit corrected SQL queries. Rewards are given for:
      - Syntax validity
      - Successful execution
      - Correct result set
      - Matching reference solution

    Supports 3 task difficulties: easy, medium, hard.
    """

    def __init__(self, task_name: str = "easy_syntax_fix"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task_name = task_name
        self.task_config = TASKS[task_name]

        # State
        self._episode_id: Optional[str] = None
        self._current_step: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._attempts: list = []
        self._db_path: Optional[str] = None
        self._db_file: Optional[Any] = None  # tempfile handle

    # ── Database setup ────────────────────────────────────────────────────────

    def _setup_database(self) -> str:
        """Create a fresh in-disk SQLite database for this episode."""
        if self._db_file is not None:
            try:
                os.unlink(self._db_file)
            except Exception:
                pass

        # Create a named temp file that persists for the episode
        fd, path = tempfile.mkstemp(suffix=".db", prefix="sql_debug_")
        os.close(fd)
        self._db_file = path

        conn = sqlite3.connect(path)
        for statement in self.task_config["create_sql"].split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()
        conn.close()
        return path

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(
        self,
        attempt_number: int = 1,
        partial_score: float = 0.0,
        last_execution_result: Optional[str] = None,
    ) -> SQLDebugObservation:
        schema_models = []
        for tbl in self.task_config["schema"]:
            schema_models.append(
                TableSchema(
                    table_name=tbl["table_name"],
                    columns=tbl["columns"],
                    sample_rows=tbl.get("sample_rows", []),
                )
            )
        return SQLDebugObservation(
            task_id=self.task_name,
            task_difficulty=self.task_config["difficulty"],
            broken_sql=self.task_config["broken_sql"].strip(),
            error_message=self.task_config["error_message"],
            db_schema=schema_models,
            expected_result_description=self.task_config["expected_result_description"],
            attempt_number=attempt_number,
            max_attempts=self.task_config["max_steps"],
            last_execution_result=last_execution_result,
            partial_score=partial_score,
        )

    # ── OpenEnv Interface ─────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Initialize a new episode. Returns the initial observation."""
        self._episode_id = str(uuid.uuid4())
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._attempts = []
        self._db_path = self._setup_database()

        obs = self._build_observation(attempt_number=1, partial_score=0.0)
        return ResetResult(
            observation=obs,
            info={"episode_id": self._episode_id, "task": self.task_name},
        )

    def step(self, action: SQLDebugAction) -> StepResult:
        """
        Execute the agent's action (corrected SQL) and return observation + reward.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._current_step += 1
        submitted_sql = action.corrected_sql.strip()

        # Grade the submission
        grade = grade_submission(
            task_name=self.task_name,
            submitted_sql=submitted_sql,
            db_path=self._db_path,
            attempt_number=self._current_step,
            max_attempts=self.task_config["max_steps"],
        )

        reward = grade["reward"]
        self._total_reward += reward

        # Record attempt
        self._attempts.append(
            {
                "step": self._current_step,
                "submitted_sql": submitted_sql,
                "reward": reward,
                "results_match": grade["results_match"],
                "feedback": grade["feedback"],
            }
        )

        # Episode is done when:
        # 1. Agent got correct results
        # 2. Max steps reached
        done = grade["results_match"] or (self._current_step >= self.task_config["max_steps"])
        self._done = done

        # Build next observation
        obs = self._build_observation(
            attempt_number=self._current_step + 1,
            partial_score=min(1.0, self._total_reward / self.task_config["max_steps"]),
            last_execution_result=grade.get("execution_result"),
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "syntax_valid": grade["syntax_valid"],
                "executes_without_error": grade["executes_without_error"],
                "results_match": grade["results_match"],
                "is_optimal": grade["is_optimal"],
                "feedback": grade["feedback"],
                "step": self._current_step,
                "episode_id": self._episode_id,
            },
        )

    def state(self) -> SQLDebugState:
        """Return current episode metadata."""
        return SQLDebugState(
            episode_id=self._episode_id or "",
            task_name=self.task_name,
            task_difficulty=self.task_config["difficulty"],
            current_step=self._current_step,
            max_steps=self.task_config["max_steps"],
            done=self._done,
            total_reward=self._total_reward,
            attempts=self._attempts,
        )

    def close(self):
        """Cleanup resources."""
        if self._db_file and os.path.exists(self._db_file):
            try:
                os.unlink(self._db_file)
            except Exception:
                pass
