"""
FastAPI server for the SQL Query Optimizer OpenEnv environment.

Exposes the standard OpenEnv HTTP API:
  POST /reset          → initial observation
  POST /step           → (observation, reward, done, info)
  GET  /state          → current state dict
  GET  /health         → liveness check
  GET  /tasks          → list of available tasks
  POST /grade          → run a grader against a submitted query
"""

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import SQLQueryOptimizerEnv
from env.models import Action
from graders import run_all_graders

app = FastAPI(
    title="SQL Query Optimizer OpenEnv",
    description="Real-world OpenEnv environment: teach AI agents to optimize slow SQL queries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store (in-memory; use Redis for multi-worker production)
_sessions: Dict[str, SQLQueryOptimizerEnv] = {}


# ──────────────────────────────────────────────
# Request/Response models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: int = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    optimized_query: str
    request_hint: bool = False
    declare_done: bool = False
    reasoning: Optional[str] = None


class GradeRequest(BaseModel):
    easy_query: str
    medium_query: str
    hard_query: str
    hints_easy: int = 0
    hints_medium: int = 0
    hints_hard: int = 0
    seed: int = 42


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "sql-query-optimizer", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "easy_top_customers",
                "difficulty": "easy",
                "description": "Eliminate correlated subquery to find top customers by spend.",
                "max_steps": 8,
                "target_speedup": "3x",
            },
            {
                "task_id": "medium_product_revenue",
                "difficulty": "medium",
                "description": "Optimize multi-table join with missing indexes and redundant subqueries.",
                "max_steps": 12,
                "target_speedup": "5x",
            },
            {
                "task_id": "hard_cohort_retention",
                "difficulty": "hard",
                "description": "Rewrite a quadratic cohort retention query using CTEs and window functions.",
                "max_steps": 15,
                "target_speedup": "8x",
            },
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest):
    if request.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")

    session_id = request.session_id or str(uuid.uuid4())
    env = SQLQueryOptimizerEnv(difficulty=request.difficulty, seed=request.seed)
    obs = env.reset()
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(request: StepRequest):
    env = _sessions.get(request.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found. Call /reset first.")

    action = Action(
        optimized_query=request.optimized_query,
        request_hint=request.request_hint,
        declare_done=request.declare_done,
        reasoning=request.reasoning,
    )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return env.state()


@app.post("/grade")
def grade(request: GradeRequest):
    """Run all graders and return aggregate score. Used by judges."""
    results = run_all_graders(
        easy_query=request.easy_query,
        medium_query=request.medium_query,
        hard_query=request.hard_query,
        hints_easy=request.hints_easy,
        hints_medium=request.hints_medium,
        hints_hard=request.hints_hard,
        seed=request.seed,
    )
    return results


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    return {"deleted": False}
