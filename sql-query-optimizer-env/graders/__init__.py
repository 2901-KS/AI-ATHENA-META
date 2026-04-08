"""
Graders for the SQL Query Optimizer environment.

Each grader runs a complete episode and produces a deterministic score (0.0-1.0).
Scores are composites of:
  - Correctness (result set matches baseline)
  - Performance improvement (speedup ratio)
  - Efficiency (steps used vs max)
  - Hint usage penalty
"""

import math
from typing import Dict, Any, Optional
from ..environment import SQLQueryOptimizerEnv
from ..models import Action, EpisodeResult


class BaseGrader:
    """Base class for task graders."""

    difficulty: str = "base"
    task_id: str = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def grade(self, final_query: str, hints_used: int = 0) -> Dict[str, Any]:
        """
        Run the environment with the submitted query and return a grade dict.

        Args:
            final_query: The agent's best optimized SQL query
            hints_used: Number of hints the agent consumed

        Returns:
            dict with 'score' (0.0-1.0) and 'breakdown' details
        """
        env = SQLQueryOptimizerEnv(difficulty=self.difficulty, seed=self.seed)
        obs = env.reset()

        baseline_ms = obs.baseline_metrics.execution_time_ms

        # Submit the query directly
        action = Action(
            optimized_query=final_query,
            request_hint=False,
            declare_done=True,
        )
        obs, reward, done, info = env.step(action)

        metrics = obs.current_metrics
        is_correct = metrics.result_hash == obs.baseline_metrics.result_hash
        current_ms = metrics.execution_time_ms
        speedup = baseline_ms / max(current_ms, 0.1)

        return self._compute_score(
            is_correct=is_correct,
            speedup=speedup,
            hints_used=hints_used,
            baseline_ms=baseline_ms,
            best_ms=current_ms,
            obs=obs,
        )

    def _compute_score(
        self,
        is_correct: bool,
        speedup: float,
        hints_used: int,
        baseline_ms: float,
        best_ms: float,
        obs=None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class EasyGrader(BaseGrader):
    """
    Grader for the easy task: correlated subquery elimination.

    Scoring:
      - 0.50 correctness (must match baseline results)
      - 0.40 performance (target: 3x speedup, scales linearly up to target)
      - 0.10 efficiency (steps < half of max_steps)
    Hint penalty: -0.05 per hint used
    """
    difficulty = "easy"
    task_id = "easy_top_customers"
    TARGET_SPEEDUP = 3.0

    def _compute_score(self, is_correct, speedup, hints_used, baseline_ms, best_ms, obs=None):
        correctness = 0.50 if is_correct else 0.0

        if is_correct and speedup > 1.0:
            perf = min(0.40, 0.40 * speedup / self.TARGET_SPEEDUP)
        else:
            perf = 0.0

        # Efficiency bonus (max 0.10)
        efficiency = 0.10  # always give this since we're grading final answer

        hint_pen = min(0.15, hints_used * 0.05)

        raw_score = correctness + perf + efficiency - hint_pen
        score = round(max(0.0, min(1.0, raw_score)), 4)

        return {
            "score": score,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "breakdown": {
                "correctness": correctness,
                "performance": perf,
                "efficiency": efficiency,
                "hint_penalty": -hint_pen,
            },
            "metrics": {
                "speedup_ratio": round(speedup, 3),
                "baseline_ms": round(baseline_ms, 3),
                "optimized_ms": round(best_ms, 3),
                "is_correct": is_correct,
                "hints_used": hints_used,
            },
            "passed": score >= 0.5,
        }


class MediumGrader(BaseGrader):
    """
    Grader for the medium task: multi-table join optimization.

    Scoring:
      - 0.45 correctness
      - 0.45 performance (target: 5x speedup)
      - 0.10 bonus for using indexes (detected via plan)
    Hint penalty: -0.05 per hint
    """
    difficulty = "medium"
    task_id = "medium_product_revenue"
    TARGET_SPEEDUP = 5.0

    def _compute_score(self, is_correct, speedup, hints_used, baseline_ms, best_ms, obs=None):
        correctness = 0.45 if is_correct else 0.0

        if is_correct and speedup > 1.0:
            perf = min(0.45, 0.45 * math.log(speedup + 1, self.TARGET_SPEEDUP + 1))
        else:
            perf = 0.0

        # Index bonus
        index_bonus = 0.10 if (obs and obs.current_metrics and obs.current_metrics.used_index) else 0.0

        hint_pen = min(0.20, hints_used * 0.05)

        raw_score = correctness + perf + index_bonus - hint_pen
        score = round(max(0.0, min(1.0, raw_score)), 4)

        return {
            "score": score,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "breakdown": {
                "correctness": correctness,
                "performance": perf,
                "index_bonus": index_bonus,
                "hint_penalty": -hint_pen,
            },
            "metrics": {
                "speedup_ratio": round(speedup, 3),
                "baseline_ms": round(baseline_ms, 3),
                "optimized_ms": round(best_ms, 3),
                "is_correct": is_correct,
                "used_index": obs.current_metrics.used_index if obs else False,
                "hints_used": hints_used,
            },
            "passed": score >= 0.5,
        }


class HardGrader(BaseGrader):
    """
    Grader for the hard task: cohort retention with window functions.

    Scoring:
      - 0.40 correctness (result set matches — partial credit for row count match)
      - 0.40 performance (target: 8x speedup, log scale)
      - 0.20 technique bonus (uses WITH/CTE, avoids UNION ALL repetition)
    Hint penalty: -0.04 per hint
    """
    difficulty = "hard"
    task_id = "hard_cohort_retention"
    TARGET_SPEEDUP = 8.0

    def _compute_score(self, is_correct, speedup, hints_used, baseline_ms, best_ms, obs=None):
        correctness = 0.40 if is_correct else 0.0

        if is_correct and speedup > 1.0:
            perf = min(0.40, 0.40 * math.log(speedup, 2) / math.log(self.TARGET_SPEEDUP, 2))
        else:
            perf = 0.0

        # Technique bonus: reward use of CTEs and window functions
        technique_bonus = 0.0
        if obs and obs.current_query:
            q_upper = obs.current_query.upper()
            uses_cte = "WITH " in q_upper
            uses_window = "OVER (" in q_upper or "PARTITION BY" in q_upper
            avoids_union_repeat = q_upper.count("UNION ALL") <= 0
            if uses_cte:
                technique_bonus += 0.10
            if uses_window:
                technique_bonus += 0.07
            if avoids_union_repeat:
                technique_bonus += 0.03
        technique_bonus = min(0.20, technique_bonus)

        hint_pen = min(0.20, hints_used * 0.04)

        raw_score = correctness + perf + technique_bonus - hint_pen
        score = round(max(0.0, min(1.0, raw_score)), 4)

        return {
            "score": score,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "breakdown": {
                "correctness": correctness,
                "performance": perf,
                "technique_bonus": technique_bonus,
                "hint_penalty": -hint_pen,
            },
            "metrics": {
                "speedup_ratio": round(speedup, 3),
                "baseline_ms": round(baseline_ms, 3),
                "optimized_ms": round(best_ms, 3),
                "is_correct": is_correct,
                "hints_used": hints_used,
            },
            "passed": score >= 0.4,
        }


def run_all_graders(
    easy_query: str,
    medium_query: str,
    hard_query: str,
    hints_easy: int = 0,
    hints_medium: int = 0,
    hints_hard: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run all three graders and return aggregated results.
    Used by the baseline inference script.
    """
    easy_result = EasyGrader(seed=seed).grade(easy_query, hints_easy)
    medium_result = MediumGrader(seed=seed).grade(medium_query, hints_medium)
    hard_result = HardGrader(seed=seed).grade(hard_query, hints_hard)

    weights = {"easy": 0.25, "medium": 0.35, "hard": 0.40}
    aggregate = (
        weights["easy"] * easy_result["score"]
        + weights["medium"] * medium_result["score"]
        + weights["hard"] * hard_result["score"]
    )

    return {
        "aggregate_score": round(aggregate, 4),
        "easy": easy_result,
        "medium": medium_result,
        "hard": hard_result,
    }
