"""
Typed Pydantic models for the SQL Query Optimizer OpenEnv environment.
Implements the full OpenEnv spec: Observation, Action, Reward.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryMetrics(BaseModel):
    """Performance metrics for a SQL query execution."""
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")
    rows_scanned: int = Field(..., description="Number of rows scanned during execution")
    rows_returned: int = Field(..., description="Number of rows in the result set")
    query_plan_cost: float = Field(..., description="Optimizer estimated cost from EXPLAIN")
    used_index: bool = Field(..., description="Whether the query used an index")
    result_hash: str = Field(..., description="MD5 hash of result set for correctness checking")


class Observation(BaseModel):
    """
    What the agent sees at each step.
    
    Contains the current query, its performance metrics, the database schema,
    feedback on the last action taken, and step metadata.
    """
    task_id: str = Field(..., description="Unique identifier for the current task")
    task_description: str = Field(..., description="Natural language description of the optimization goal")
    difficulty: str = Field(..., description="Task difficulty: easy | medium | hard")
    
    current_query: str = Field(..., description="The SQL query to optimize")
    original_query: str = Field(..., description="The baseline query (for comparison)")
    
    schema_info: Dict[str, Any] = Field(..., description="Database schema: tables, columns, indexes, row counts")
    sample_data: Dict[str, List[Dict]] = Field(default_factory=dict, description="Sample rows per table (up to 5)")
    
    current_metrics: QueryMetrics = Field(..., description="Performance metrics of current_query")
    baseline_metrics: QueryMetrics = Field(..., description="Performance metrics of the original baseline query")
    
    last_action_feedback: str = Field(default="", description="Human-readable feedback on the previous action")
    last_action_valid: bool = Field(default=True, description="Whether the last submitted query was syntactically valid")
    
    step_number: int = Field(..., description="Current step within the episode (0-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated so far")
    
    hint: Optional[str] = Field(default=None, description="Optional hint available (costs -0.05 reward to use)")
    done: bool = Field(default=False, description="Whether the episode has ended")


class Action(BaseModel):
    """
    What the agent can do at each step.
    
    The agent submits an optimized SQL query, optionally requests a hint,
    and can declare it's finished optimizing.
    """
    optimized_query: str = Field(..., description="The SQL query the agent wants to try")
    request_hint: bool = Field(default=False, description="Whether to request a hint (costs -0.05 reward)")
    declare_done: bool = Field(default=False, description="Agent signals it's satisfied with current optimization")
    reasoning: Optional[str] = Field(default=None, description="Agent's explanation of what optimization was applied")


class Reward(BaseModel):
    """
    Shaped reward signal with partial progress components.
    
    Rewards are dense across the trajectory, not just binary end-of-episode.
    """
    total: float = Field(..., description="Total reward for this step (-1.0 to 1.0)")
    
    # Partial reward components
    correctness_score: float = Field(..., description="Is the result set identical to baseline? (0.0 or 0.4)")
    performance_improvement: float = Field(..., description="Normalized speedup reward (0.0 to 0.4)")
    plan_cost_reduction: float = Field(..., description="Query plan cost improvement (0.0 to 0.2)")
    
    # Penalties
    syntax_penalty: float = Field(default=0.0, description="Penalty for invalid SQL (-0.1)")
    hint_penalty: float = Field(default=0.0, description="Penalty for using a hint (-0.05)")
    step_penalty: float = Field(default=-0.01, description="Small per-step cost to encourage efficiency")
    
    # Metadata
    speedup_ratio: float = Field(..., description="execution_time_baseline / execution_time_current")
    is_correct: bool = Field(..., description="Whether result set matches the baseline")
    improvement_pct: float = Field(..., description="Percentage improvement over baseline")
    
    explanation: str = Field(..., description="Human-readable reward breakdown")


class EpisodeResult(BaseModel):
    """Summary of a completed episode, used by graders."""
    task_id: str
    difficulty: str
    final_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score 0.0-1.0")
    
    steps_taken: int
    max_steps: int
    
    baseline_execution_ms: float
    best_execution_ms: float
    speedup_ratio: float
    
    result_correct: bool
    hints_used: int
    
    final_query: str
    optimization_techniques: List[str] = Field(default_factory=list)
    
    grade_breakdown: Dict[str, float] = Field(default_factory=dict)
