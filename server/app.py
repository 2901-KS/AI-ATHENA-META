"""
FastAPI server for the SQL Debug Environment.
Exposes the OpenEnv HTTP API: POST /reset, POST /step, GET /state, GET /health
"""
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SQLDebugAction
from server.environment import SQLDebugEnvironment
from tasks import TASKS


# ─── Session store (simple in-memory, keyed by session_id) ───────────────────

_sessions: Dict[str, SQLDebugEnvironment] = {}
_default_task = os.getenv("SQL_DEBUG_TASK", "easy_syntax_fix")


def _get_or_create_session(session_id: Optional[str], task_name: str) -> tuple[str, SQLDebugEnvironment]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = session_id or "default"
    env = SQLDebugEnvironment(task_name=task_name)
    _sessions[sid] = env
    return sid, env


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SQL Debug Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents debug broken SQL queries. "
        "Supports 3 difficulty levels: easy, medium, hard."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    corrected_sql: str
    explanation: Optional[str] = None
    session_id: Optional[str] = None
    task_name: Optional[str] = None


class StateRequest(BaseModel):
    session_id: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — validators ping this to confirm the Space is live."""
    return {"status": "ok", "environment": "sql-debug-env", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Initialize or reset an episode.
    Returns: initial observation.
    """
    if request is None:
        request = ResetRequest()

    task_name = (request.task_name or _default_task).strip()
    if task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid tasks: {list(TASKS.keys())}",
        )

    session_id = request.session_id or "default"
    env = SQLDebugEnvironment(task_name=task_name)
    _sessions[session_id] = env

    result = env.reset()
    return {
        "session_id": session_id,
        "observation": result.observation.model_dump(),
        "info": result.info,
    }


@app.post("/step")
async def step(request: StepRequest):
    """
    Submit a corrected SQL action and get back the next observation + reward.
    """
    session_id = request.session_id or "default"

    if session_id not in _sessions:
        # Auto-create session with default task
        task_name = request.task_name or _default_task
        env = SQLDebugEnvironment(task_name=task_name)
        env.reset()
        _sessions[session_id] = env

    env = _sessions[session_id]

    try:
        action = SQLDebugAction(
            corrected_sql=request.corrected_sql,
            explanation=request.explanation,
        )
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "session_id": session_id,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
@app.post("/state")
async def state(session_id: Optional[str] = Query(default=None)):
    """Return current episode state metadata."""
    sid = session_id or "default"
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail=f"No active session '{sid}'. Call /reset first.")
    env = _sessions[sid]
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with their metadata."""
    return {
        name: {
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "broken_sql": cfg["broken_sql"].strip(),
            "error_message": cfg["error_message"],
        }
        for name, cfg in TASKS.items()
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Web UI with interactive demo for the HF Space."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SQL Debug Environment — OpenEnv</title>
<style>
  body { font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }
  h1 { color: #58a6ff; }
  h2 { color: #8b949e; font-size: 1rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
  select, textarea, button { width: 100%; padding: 0.5rem; margin: 0.4rem 0; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #c9d1d9; font-family: monospace; }
  button { background: #238636; border-color: #2ea043; cursor: pointer; font-size: 1rem; }
  button:hover { background: #2ea043; }
  pre { background: #0d1117; padding: 1rem; border-radius: 6px; overflow-x: auto; border: 1px solid #30363d; white-space: pre-wrap; }
  .reward { color: #3fb950; font-size: 1.5rem; font-weight: bold; }
  .error { color: #f85149; }
  label { color: #8b949e; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>🗄️ SQL Debug Environment</h1>
<h2>OpenEnv — Real-world SQL debugging for AI agents</h2>

<div class="card">
  <label>Select Task:</label>
  <select id="taskSelect">
    <option value="easy_syntax_fix">Easy: Syntax Fix</option>
    <option value="medium_logic_fix">Medium: Logic Fix</option>
    <option value="hard_optimization_fix">Hard: Multi-Error Fix</option>
  </select>
  <button onclick="doReset()">🔄 Reset / Start Episode</button>
</div>

<div class="card" id="obsCard" style="display:none">
  <h2>📋 Current Observation</h2>
  <pre id="brokenSql"></pre>
  <pre id="errorMsg" class="error"></pre>
  <label>Your Corrected SQL:</label>
  <textarea id="sqlInput" rows="6" placeholder="Write your corrected SQL here..."></textarea>
  <button onclick="doStep()">▶ Submit Correction</button>
</div>

<div class="card" id="resultCard" style="display:none">
  <h2>📊 Result</h2>
  <div class="reward" id="rewardDisplay"></div>
  <pre id="feedbackDisplay"></pre>
</div>

<script>
let sessionId = 'web-' + Math.random().toString(36).substr(2,9);

async function doReset() {
  const task = document.getElementById('taskSelect').value;
  const res = await fetch('/reset', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task_name: task, session_id: sessionId})
  });
  const data = await res.json();
  const obs = data.observation;
  document.getElementById('brokenSql').textContent = 
    'BROKEN SQL:\\n' + obs.broken_sql + '\\n\\nERROR: ' + obs.error_message;
  document.getElementById('errorMsg').textContent = 
    'Goal: ' + obs.expected_result_description;
  document.getElementById('obsCard').style.display = 'block';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('sqlInput').value = '';
}

async function doStep() {
  const sql = document.getElementById('sqlInput').value;
  const res = await fetch('/step', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({corrected_sql: sql, session_id: sessionId})
  });
  const data = await res.json();
  document.getElementById('rewardDisplay').textContent = 
    'Reward: ' + data.reward.toFixed(3) + (data.done ? ' ✅ Done!' : '');
  document.getElementById('feedbackDisplay').textContent = 
    JSON.stringify(data.info, null, 2);
  document.getElementById('resultCard').style.display = 'block';
  if (!data.done) {
    const obs = data.observation;
    document.getElementById('brokenSql').textContent = 
      'BROKEN SQL:\\n' + obs.broken_sql + '\\n\\nAttempt: ' + obs.attempt_number + '/' + obs.max_attempts;
  }
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
