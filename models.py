"""
Typed Pydantic models for the SQL Debug Environment.
Defines Observation, Action, and Reward for the OpenEnv spec.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Action ───────────────────────────────────────────────────────────────────

class SQLDebugAction(BaseModel):
    """
    The agent's action: a corrected SQL query string.
    Optionally includes a short explanation of what was fixed.
    """
    corrected_sql: str = Field(
        ...,
        description="The agent's corrected SQL query. Must be valid SQL.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Optional brief explanation of what the agent changed.",
    )


# ─── Observation ──────────────────────────────────────────────────────────────

class TableSchema(BaseModel):
    table_name: str
    columns: List[Dict[str, str]]  # [{"name": "col", "type": "INTEGER"}, ...]
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)


class SQLDebugObservation(BaseModel):
    """
    What the agent sees each step.
    Contains the broken query, the error, and the database schema context.
    """
    task_id: str = Field(..., description="Unique task identifier")
    task_difficulty: str = Field(..., description="easy | medium | hard")
    broken_sql: str = Field(..., description="The SQL query that needs to be fixed")
    error_message: str = Field(..., description="The database error message returned")
    db_schema: List[TableSchema] = Field(..., description="Available table schemas")
    expected_result_description: str = Field(
        ..., description="Plain-English description of what the query should return"
    )
    attempt_number: int = Field(default=1, description="Which attempt this is (1-indexed)")
    max_attempts: int = Field(default=5, description="Maximum attempts allowed")
    last_execution_result: Optional[str] = Field(
        default=None,
        description="Result of executing the last submitted query (if any)",
    )
    partial_score: float = Field(
        default=0.0,
        description="Running partial score from 0.0 to 1.0",
    )


# ─── Reward ───────────────────────────────────────────────────────────────────

class SQLDebugReward(BaseModel):
    """
    Reward signal returned after each step.
    """
    value: float = Field(..., ge=0.0, le=1.0, description="Reward value in [0, 1]")
    syntax_valid: bool = Field(default=False, description="Query parsed without syntax error")
    executes_without_error: bool = Field(default=False, description="Query runs on the DB")
    results_match: bool = Field(default=False, description="Output matches expected results")
    is_optimal: bool = Field(default=False, description="Query is also efficient/clean")
    feedback: str = Field(default="", description="Human-readable feedback string")


# ─── State ────────────────────────────────────────────────────────────────────

class SQLDebugState(BaseModel):
    """
    Internal episode state returned by state().
    """
    episode_id: str
    task_name: str
    task_difficulty: str
    current_step: int
    max_steps: int
    done: bool
    total_reward: float
    attempts: List[Dict[str, Any]] = Field(default_factory=list)


# ─── Step Result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: SQLDebugObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: SQLDebugObservation
    info: Dict[str, Any] = Field(default_factory=dict)
