"""
Gradio Demo UI for SQL Query Optimizer OpenEnv

This runs alongside the FastAPI server on HuggingFace Spaces,
giving judges an interactive way to explore the environment.

Judges can:
  - Pick a difficulty and reset the environment
  - See the baseline query and its performance metrics
  - Submit optimized queries and watch the reward update live
  - View the reward breakdown per step
  - Run the grader on any query they type

Launch:  python demo.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

try:
    import gradio as gr
except ImportError:
    print("gradio not installed — run: pip install gradio")
    sys.exit(1)

from env.environment import SQLQueryOptimizerEnv
from env.models import Action
from graders import EasyGrader, MediumGrader, HardGrader

# ── Global session (single-user demo) ──────────────
_env: SQLQueryOptimizerEnv = None
_obs = None
_history = []  # list of step dicts


# ──────────────────────────────────────────────
# Core handlers
# ──────────────────────────────────────────────

def do_reset(difficulty: str):
    global _env, _obs, _history
    _env = SQLQueryOptimizerEnv(difficulty=difficulty, seed=42)
    _obs = _env.reset()
    _history = []

    schema_md = _build_schema_md(_obs.schema_info)
    metrics_md = _build_metrics_md(_obs.baseline_metrics, _obs.baseline_metrics, step=0)

    return (
        _obs.current_query,           # baseline query box
        _obs.task_description,        # task description
        schema_md,                    # schema panel
        metrics_md,                   # metrics panel
        _obs.current_query,           # editable query box (pre-filled)
        _build_history_md([]),        # step history
        f"✅ Environment reset — {difficulty} task loaded. Baseline: {_obs.baseline_metrics.execution_time_ms:.1f}ms",
    )


def do_step(query: str, request_hint: bool, declare_done: bool):
    global _env, _obs, _history

    if _env is None or _obs is None:
        return ("", "", "", "", query, "", "⚠️  Please reset the environment first.")

    if _obs.done:
        return ("", _obs.task_description, "", "", query, _build_history_md(_history),
                "🏁 Episode is done. Click Reset to start a new episode.")

    action = Action(
        optimized_query=query.strip(),
        request_hint=request_hint,
        declare_done=declare_done,
    )
    _obs, reward, done, info = _env.step(action)

    step_info = {
        "step": _obs.step_number,
        "speedup": reward.speedup_ratio,
        "correct": reward.is_correct,
        "reward": reward.total,
        "exec_ms": _obs.current_metrics.execution_time_ms,
        "query_preview": query[:60].replace("\n", " ") + ("..." if len(query) > 60 else ""),
    }
    _history.append(step_info)

    metrics_md = _build_metrics_md(_obs.current_metrics, _obs.baseline_metrics, _obs.step_number)
    hint_text = f"\n\n💡 **Hint**: {_obs.hint}" if _obs.hint else ""

    status_icon = "✅" if reward.is_correct else "❌"
    status = (
        f"{status_icon} Step {_obs.step_number}/{_obs.max_steps} | "
        f"Speedup: **{reward.speedup_ratio:.2f}x** | "
        f"Reward: **{reward.total:+.4f}** | "
        f"Correct: {reward.is_correct} | "
        f"Cumulative: {_obs.cumulative_reward:.4f}"
        + hint_text
    )
    if done:
        status += "\n\n🏁 **Episode complete!**"

    return (
        _obs.original_query,
        _obs.task_description,
        _build_schema_md(_obs.schema_info),
        metrics_md,
        _obs.current_query,
        _build_history_md(_history),
        status,
    )


def do_grade(easy_q: str, medium_q: str, hard_q: str):
    graders = {
        "easy": EasyGrader(seed=42),
        "medium": MediumGrader(seed=42),
        "hard": HardGrader(seed=42),
    }
    queries = {"easy": easy_q, "medium": medium_q, "hard": hard_q}
    results = {}
    for diff, grader in graders.items():
        q = queries[diff].strip()
        if q:
            results[diff] = grader.grade(q)
        else:
            results[diff] = {"score": 0.0, "passed": False, "breakdown": {}}

    weights = {"easy": 0.25, "medium": 0.35, "hard": 0.40}
    agg = sum(weights[d] * results[d]["score"] for d in weights)

    lines = ["## 📊 Grader Results\n"]
    for diff in ["easy", "medium", "hard"]:
        r = results[diff]
        icon = "✅" if r["passed"] else "❌"
        lines.append(f"### {icon} {diff.title()} — Score: `{r['score']:.4f}`")
        for k, v in r.get("breakdown", {}).items():
            lines.append(f"- **{k}**: {v:.4f}")
        lines.append("")

    lines.append(f"---\n### 🏆 Aggregate Score: `{agg:.4f}`")
    lines.append(f"*(weights: easy×0.25 + medium×0.35 + hard×0.40)*")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Markdown builders
# ──────────────────────────────────────────────

def _build_schema_md(schema_info: dict) -> str:
    lines = ["### 🗄️ Database Schema\n"]
    for table, info in schema_info.items():
        if table.startswith("_"):
            continue
        cols = ", ".join(c["name"] for c in info["columns"])
        lines.append(f"**{table}** ({info['row_count']:,} rows)  \n`{cols}`")
        if info["indexes"]:
            lines.append(f"  *indexes: {', '.join(info['indexes'])}*")
        lines.append("")

    available = schema_info.get("_available_indexes", [])
    active = set(schema_info.get("_active_indexes", []))
    inactive = [i for i in available if i not in active]
    if inactive:
        lines.append(f"**Available indexes (not created):** `{', '.join(inactive)}`")
    if active:
        lines.append(f"**Active indexes:** `{', '.join(active)}`")
    return "\n".join(lines)


def _build_metrics_md(current, baseline, step: int) -> str:
    if current is None or baseline is None:
        return "No metrics yet."
    speedup = baseline.execution_time_ms / max(current.execution_time_ms, 0.1)
    correct = current.result_hash == baseline.result_hash
    return (
        f"### ⚡ Performance Metrics (Step {step})\n\n"
        f"| Metric | Baseline | Current | Δ |\n"
        f"|---|---|---|---|\n"
        f"| Execution time | {baseline.execution_time_ms:.1f}ms | **{current.execution_time_ms:.1f}ms** | {speedup:.2f}x |\n"
        f"| Rows scanned | {baseline.rows_scanned:,} | {current.rows_scanned:,} | {'↓' if current.rows_scanned < baseline.rows_scanned else '→'} |\n"
        f"| Query plan cost | {baseline.query_plan_cost:.0f} | {current.query_plan_cost:.0f} | {'↓' if current.query_plan_cost < baseline.query_plan_cost else '→'} |\n"
        f"| Index used | {baseline.used_index} | **{current.used_index}** | |\n"
        f"| Result correct | — | **{correct}** | |\n"
    )


def _build_history_md(history: list) -> str:
    if not history:
        return "*No steps taken yet.*"
    lines = ["### 📈 Step History\n", "| Step | Query Preview | Exec (ms) | Speedup | Correct | Reward |",
             "|---|---|---|---|---|---|"]
    for h in history:
        icon = "✅" if h["correct"] else "❌"
        lines.append(
            f"| {h['step']} | `{h['query_preview']}` | "
            f"{h['exec_ms']:.1f} | {h['speedup']:.2f}x | {icon} | {h['reward']:+.4f} |"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

EXAMPLE_EASY_OPT = """WITH order_totals AS (
    SELECT customer_id, SUM(total_amount) AS total_spend
    FROM orders
    WHERE order_date >= '2023-01-01'
      AND order_date < '2024-01-01'
      AND status = 'delivered'
    GROUP BY customer_id
)
SELECT c.customer_id, c.name, c.country, ot.total_spend
FROM customers c
JOIN order_totals ot ON ot.customer_id = c.customer_id
ORDER BY ot.total_spend DESC
LIMIT 10"""

CSS = """
.reward-box { font-size: 1.1em; padding: 12px; border-radius: 8px; }
.metric-green { color: #22c55e; font-weight: bold; }
footer { display: none !important; }
"""

with gr.Blocks(title="SQL Query Optimizer — OpenEnv", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🗄️ SQL Query Optimizer — OpenEnv Environment
**Real-world task**: Teach AI agents to rewrite slow SQL queries for maximum speed while preserving correctness.
    """)

    with gr.Tabs():

        # ── Tab 1: Interactive Environment ─────────────
        with gr.TabItem("🎮 Interactive Demo"):
            with gr.Row():
                diff_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Task Difficulty",
                    scale=1,
                )
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary", scale=2)

            task_desc = gr.Markdown("*Click Reset to load a task.*")

            with gr.Row():
                with gr.Column(scale=1):
                    schema_panel = gr.Markdown("*Schema loads on reset.*")
                with gr.Column(scale=2):
                    metrics_panel = gr.Markdown("*Metrics load on reset.*")

            gr.Markdown("---")
            gr.Markdown("### ✏️ Submit Your Optimized Query")

            baseline_box = gr.Code(
                label="📋 Baseline Query (read-only)",
                language="sql",
                interactive=False,
                lines=8,
            )
            query_box = gr.Code(
                label="🚀 Your Optimized Query",
                language="sql",
                lines=12,
                value=EXAMPLE_EASY_OPT,
            )

            with gr.Row():
                hint_cb = gr.Checkbox(label="💡 Request hint (−0.05 reward)", value=False)
                done_cb = gr.Checkbox(label="🏁 Declare done", value=False)
                step_btn = gr.Button("▶️ Submit Step", variant="primary")

            status_box = gr.Markdown("*Reset the environment to begin.*", elem_classes=["reward-box"])
            history_panel = gr.Markdown("*Step history appears here.*")

        # ── Tab 2: Grader ───────────────────────────────
        with gr.TabItem("📊 Grader"):
            gr.Markdown("""
Submit your best queries for all three tasks and get a full score breakdown.
This is exactly what the judges run in Phase 1 automated evaluation.
            """)
            easy_input = gr.Code(label="Easy Task Query", language="sql", lines=6)
            medium_input = gr.Code(label="Medium Task Query", language="sql", lines=6)
            hard_input = gr.Code(label="Hard Task Query", language="sql", lines=6)
            grade_btn = gr.Button("🏆 Run Grader", variant="primary")
            grade_output = gr.Markdown("*Submit queries above to see scores.*")

        # ── Tab 3: Environment Spec ─────────────────────
        with gr.TabItem("📖 Environment Spec"):
            gr.Markdown("""
## Observation Space
| Field | Type | Description |
|---|---|---|
| `task_id` | str | e.g. `"easy_top_customers"` |
| `current_query` | str | SQL to optimize |
| `schema_info` | dict | Tables, columns, row counts, indexes |
| `current_metrics` | QueryMetrics | execution_time_ms, rows_scanned, used_index |
| `baseline_metrics` | QueryMetrics | Original query metrics |
| `step_number` / `max_steps` | int | Episode progress |
| `hint` | str? | Hint text (if requested) |

## Action Space
| Field | Type | Description |
|---|---|---|
| `optimized_query` | str | Your rewritten SQL SELECT |
| `request_hint` | bool | Get a hint (costs −0.05) |
| `declare_done` | bool | End the episode |

## Reward Components
| Component | Max | Description |
|---|---|---|
| Correctness | +0.40 | Result set matches baseline |
| Performance | +0.40 | Log-scaled speedup ratio |
| Plan cost | +0.20 | EXPLAIN cost reduction |
| Syntax error | −0.10 | Invalid SQL |
| Hint | −0.05 | Per hint used |
| Step cost | −0.01 | Per step efficiency signal |

## API Endpoints
```
POST /reset   → { session_id, observation }
POST /step    → { observation, reward, done, info }
GET  /state   → { current state dict }
POST /grade   → { aggregate_score, easy, medium, hard }
GET  /health  → { status: "ok" }
GET  /tasks   → [ task list ]
```

## HTTP Example
```python
import requests, json

# Reset
r = requests.post("http://localhost:7860/reset", json={"difficulty": "easy"})
session_id = r.json()["session_id"]

# Step
r = requests.post("http://localhost:7860/step", json={
    "session_id": session_id,
    "optimized_query": "SELECT ...",
    "declare_done": True,
})
print(r.json()["reward"]["total"])
```
            """)

    # ── Event wiring ────────────────────────────────────
    reset_outputs = [baseline_box, task_desc, schema_panel, metrics_panel,
                     query_box, history_panel, status_box]

    reset_btn.click(
        fn=do_reset,
        inputs=[diff_dropdown],
        outputs=reset_outputs,
    )

    step_btn.click(
        fn=do_step,
        inputs=[query_box, hint_cb, done_cb],
        outputs=[baseline_box, task_desc, schema_panel, metrics_panel,
                 query_box, history_panel, status_box],
    )

    grade_btn.click(
        fn=do_grade,
        inputs=[easy_input, medium_input, hard_input],
        outputs=[grade_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
