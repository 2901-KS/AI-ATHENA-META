"""
SQL Query Optimizer — OpenEnv Environment

Implements the full OpenEnv interface:
  - reset() → Observation
  - step(action) → (Observation, Reward, done, info)
  - state() → dict

Real-world task: AI agents learn to optimize slow SQL queries by rewriting them
and optionally suggesting indexes, using dense reward signals tied to measurable
performance improvements.
"""

import re
import time
from typing import Any, Dict, Optional, Tuple

from .database import DatabaseEngine
from .models import Action, EpisodeResult, Observation, Reward
from .tasks.definitions import ALL_TASKS, Task


class SQLQueryOptimizerEnv:
    """
    OpenEnv-compliant environment for SQL query optimization.

    An agent receives a slow SQL query against a realistic e-commerce database
    and must iteratively rewrite it to be faster while preserving correctness.

    Reward is dense:
      +0.40 for correct result set
      +0.40 for performance improvement (scaled by speedup ratio)
      +0.20 for query plan cost reduction
      -0.10 per invalid SQL submission
      -0.05 per hint requested
      -0.01 per step (efficiency signal)

    Episodes end when:
      - Agent calls declare_done=True
      - Max steps reached
      - Agent achieves >= target_speedup with correct results
    """

    metadata = {
        "name": "sql-query-optimizer",
        "version": "1.0.0",
        "task_ids": ["easy_top_customers", "medium_product_revenue", "hard_cohort_retention"],
        "observation_type": "SQLQueryObservation",
        "action_type": "SQLQueryAction",
        "reward_range": (-1.0, 1.0),
    }

    def __init__(self, difficulty: str = "easy", db_path: str = ":memory:", seed: int = 42):
        """
        Args:
            difficulty: "easy", "medium", or "hard"
            db_path: SQLite DB path (":memory:" for ephemeral)
            seed: Random seed for reproducibility
        """
        if difficulty not in ALL_TASKS:
            raise ValueError(f"difficulty must be one of {list(ALL_TASKS.keys())}")

        self.difficulty = difficulty
        self.seed = seed
        self._task: Task = ALL_TASKS[difficulty]
        self._db = DatabaseEngine(db_path=db_path, seed=seed)

        # Episode state
        self._current_query: str = ""
        self._baseline_metrics = None
        self._current_metrics = None
        self._step_num: int = 0
        self._cumulative_reward: float = 0.0
        self._hints_used: int = 0
        self._done: bool = False
        self._best_metrics = None
        self._best_query: str = ""
        self._episode_result: Optional[EpisodeResult] = None
        self._last_feedback: str = ""
        self._last_valid: bool = True

    # ──────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns initial observation."""
        self._db.reset()

        self._task = ALL_TASKS[self.difficulty]
        self._current_query = self._task.baseline_query
        self._step_num = 0
        self._cumulative_reward = 0.0
        self._hints_used = 0
        self._done = False
        self._last_feedback = "Environment reset. Optimize the provided SQL query."
        self._last_valid = True
        self._episode_result = None

        # Measure baseline
        metrics, error = self._db.measure_query(self._task.baseline_query)
        if error or metrics is None:
            raise RuntimeError(f"Baseline query failed: {error}")

        self._baseline_metrics = metrics
        self._current_metrics = metrics
        self._best_metrics = metrics
        self._best_query = self._task.baseline_query

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action and return (observation, reward, done, info).

        Args:
            action: Action model with optimized_query, request_hint, declare_done

        Returns:
            observation: Updated state after action
            reward: Shaped reward for this step
            done: Whether the episode has ended
            info: Additional metadata
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_num += 1
        hint_penalty = 0.0
        hint_text = None

        # Handle hint request
        if action.request_hint:
            self._hints_used += 1
            hint_idx = min(self._hints_used - 1, len(self._task.hints) - 1)
            hint_text = self._task.hints[hint_idx]
            hint_penalty = -0.05

        # Try to execute the submitted query
        sql = action.optimized_query.strip()
        metrics, error = self._db.measure_query(sql)

        if error or metrics is None:
            # Invalid SQL penalty
            reward = self._build_reward(
                metrics=self._current_metrics,
                syntax_error=True,
                hint_penalty=hint_penalty,
            )
            self._last_feedback = f"Invalid SQL: {error}. Previous query retained."
            self._last_valid = False

            obs = self._build_observation(hint=hint_text)
            self._cumulative_reward += reward.total
            done = self._check_done(action.declare_done)
            self._done = done
            if done:
                self._finalize_episode()
            return obs, reward, done, self._build_info()

        # Valid query — update state
        self._last_valid = True
        self._current_query = sql
        self._current_metrics = metrics

        # Track best performing query
        if metrics.execution_time_ms < self._best_metrics.execution_time_ms:
            if metrics.result_hash == self._baseline_metrics.result_hash:
                self._best_metrics = metrics
                self._best_query = sql

        reward = self._build_reward(
            metrics=metrics,
            syntax_error=False,
            hint_penalty=hint_penalty,
        )

        speedup = self._baseline_metrics.execution_time_ms / max(metrics.execution_time_ms, 0.1)
        self._last_feedback = (
            f"Query valid. Speedup: {speedup:.2f}x | "
            f"Execution: {metrics.execution_time_ms:.1f}ms (baseline: {self._baseline_metrics.execution_time_ms:.1f}ms) | "
            f"Correct: {metrics.result_hash == self._baseline_metrics.result_hash} | "
            f"Index used: {metrics.used_index}"
        )

        self._cumulative_reward += reward.total
        done = self._check_done(action.declare_done)
        self._done = done
        if done:
            self._finalize_episode()

        obs = self._build_observation(hint=hint_text)
        return obs, reward, done, self._build_info()

    def state(self) -> Dict[str, Any]:
        """Return current environment state as a plain dict (for serialization)."""
        return {
            "difficulty": self.difficulty,
            "task_id": self._task.task_id,
            "step_number": self._step_num,
            "max_steps": self._task.max_steps,
            "current_query": self._current_query,
            "cumulative_reward": self._cumulative_reward,
            "hints_used": self._hints_used,
            "done": self._done,
            "current_execution_ms": self._current_metrics.execution_time_ms if self._current_metrics else None,
            "baseline_execution_ms": self._baseline_metrics.execution_time_ms if self._baseline_metrics else None,
            "best_execution_ms": self._best_metrics.execution_time_ms if self._best_metrics else None,
            "active_indexes": list(self._db._active_indexes),
        }

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _build_observation(self, hint: Optional[str] = None) -> Observation:
        return Observation(
            task_id=self._task.task_id,
            task_description=self._task.description,
            difficulty=self._task.difficulty,
            current_query=self._current_query,
            original_query=self._task.baseline_query,
            schema_info=self._db.get_schema_info(),
            sample_data=self._db.get_sample_data(),
            current_metrics=self._current_metrics,
            baseline_metrics=self._baseline_metrics,
            last_action_feedback=self._last_feedback,
            last_action_valid=self._last_valid,
            step_number=self._step_num,
            max_steps=self._task.max_steps,
            cumulative_reward=self._cumulative_reward,
            hint=hint,
            done=self._done,
        )

    def _build_reward(
        self,
        metrics,
        syntax_error: bool = False,
        hint_penalty: float = 0.0,
    ) -> Reward:
        step_penalty = -0.01

        if syntax_error:
            syntax_pen = -0.10
            return Reward(
                total=round(syntax_pen + hint_penalty + step_penalty, 4),
                correctness_score=0.0,
                performance_improvement=0.0,
                plan_cost_reduction=0.0,
                syntax_penalty=syntax_pen,
                hint_penalty=hint_penalty,
                step_penalty=step_penalty,
                speedup_ratio=1.0,
                is_correct=False,
                improvement_pct=0.0,
                explanation="Invalid SQL submitted. Penalty applied.",
            )

        baseline_ms = self._baseline_metrics.execution_time_ms
        current_ms = max(metrics.execution_time_ms, 0.1)
        speedup = baseline_ms / current_ms

        # Correctness (0.0 or 0.40)
        is_correct = metrics.result_hash == self._baseline_metrics.result_hash
        correctness_score = 0.40 if is_correct else 0.0

        # Performance improvement (0.0 to 0.40), scaled by log speedup
        import math
        if speedup > 1.0 and is_correct:
            # log scale: 2x=0.10, 4x=0.20, 8x=0.30, target_speedup=0.40
            perf_score = min(0.40, 0.40 * math.log(speedup, 2) / math.log(self._task.target_speedup, 2))
        elif speedup >= 1.0:
            perf_score = 0.0
        else:
            # Regression — small penalty
            perf_score = max(-0.10, -0.10 * (1.0 - speedup))

        # Plan cost reduction (0.0 to 0.20)
        baseline_cost = max(self._baseline_metrics.query_plan_cost, 1.0)
        current_cost = max(metrics.query_plan_cost, 1.0)
        cost_ratio = baseline_cost / current_cost
        plan_score = min(0.20, 0.20 * (cost_ratio - 1.0) / 4.0) if cost_ratio > 1.0 else 0.0

        total = correctness_score + perf_score + plan_score + hint_penalty + step_penalty
        total = round(max(-1.0, min(1.0, total)), 4)

        improvement_pct = (speedup - 1.0) * 100 if speedup > 1.0 else 0.0

        return Reward(
            total=total,
            correctness_score=correctness_score,
            performance_improvement=perf_score,
            plan_cost_reduction=plan_score,
            syntax_penalty=0.0,
            hint_penalty=hint_penalty,
            step_penalty=step_penalty,
            speedup_ratio=round(speedup, 3),
            is_correct=is_correct,
            improvement_pct=round(improvement_pct, 2),
            explanation=(
                f"Correctness: {correctness_score:.2f} | "
                f"Perf: {perf_score:.2f} ({speedup:.2f}x speedup) | "
                f"Plan: {plan_score:.2f} | "
                f"Hint: {hint_penalty:.2f} | "
                f"Step: {step_penalty:.2f} → Total: {total:.4f}"
            ),
        )

    def _check_done(self, declare_done: bool) -> bool:
        if declare_done:
            return True
        if self._step_num >= self._task.max_steps:
            return True
        # Auto-complete if agent achieves target speedup with correct results
        if self._current_metrics and self._baseline_metrics:
            speedup = self._baseline_metrics.execution_time_ms / max(self._current_metrics.execution_time_ms, 0.1)
            correct = self._current_metrics.result_hash == self._baseline_metrics.result_hash
            if speedup >= self._task.target_speedup and correct:
                return True
        return False

    def _finalize_episode(self) -> None:
        speedup = (
            self._baseline_metrics.execution_time_ms / max(self._best_metrics.execution_time_ms, 0.1)
            if self._best_metrics else 1.0
        )
        self._episode_result = EpisodeResult(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            final_score=0.0,  # filled by grader
            steps_taken=self._step_num,
            max_steps=self._task.max_steps,
            baseline_execution_ms=self._baseline_metrics.execution_time_ms,
            best_execution_ms=self._best_metrics.execution_time_ms,
            speedup_ratio=round(speedup, 3),
            result_correct=self._best_metrics.result_hash == self._baseline_metrics.result_hash,
            hints_used=self._hints_used,
            final_query=self._best_query,
        )

    def _build_info(self) -> Dict[str, Any]:
        info = {
            "step": self._step_num,
            "max_steps": self._task.max_steps,
            "hints_used": self._hints_used,
        }
        if self._episode_result:
            info["episode_result"] = self._episode_result.model_dump()
        return info
