# SQL Query Optimizer — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-ready-green)](./Dockerfile)
[![Tasks](https://img.shields.io/badge/Tasks-3%20(easy→hard)-orange)](./openenv.yaml)

A **real-world OpenEnv environment** where AI agents learn to optimize slow SQL queries
against a realistic e-commerce database — a task that costs companies millions of dollars
in cloud infrastructure annually.

---

## 🎯 Motivation

Database query optimization is one of the most impactful real-world engineering tasks:
- A single slow query can cost thousands of dollars per day in cloud costs
- DBAs spend 30-40% of their time identifying and fixing query performance issues
- No existing OpenEnv benchmark covers this domain

This environment trains agents to **identify bottlenecks** (full table scans, correlated
subqueries, missing indexes) and **rewrite queries** to be dramatically faster while
preserving 100% result correctness.

---

## 🏗️ Environment Overview

### Database Schema (E-Commerce)

```
customers (2,000 rows)    — customer_id, name, email, country, signup_date, tier
products  (500 rows)      — product_id, name, category, price, stock, supplier_id
orders    (10,000 rows)   — order_id, customer_id, order_date, status, total_amount
order_items (30,000 rows) — item_id, order_id, product_id, quantity, unit_price
reviews   (8,000 rows)    — review_id, product_id, customer_id, rating, review_date
```

### Action Space

```python
class Action(BaseModel):
    optimized_query: str       # The agent's rewritten SQL SELECT statement
    request_hint: bool         # Request a hint (costs -0.05 reward)
    declare_done: bool         # Signal the agent is satisfied
    reasoning: Optional[str]   # Agent's explanation (logged, not evaluated)
```

### Observation Space

```python
class Observation(BaseModel):
    task_id: str               # e.g. "easy_top_customers"
    task_description: str      # Natural language goal
    current_query: str         # Query to optimize
    original_query: str        # Unchanged baseline
    schema_info: dict          # Tables, columns, indexes, row counts
    sample_data: dict          # 3 sample rows per table
    current_metrics: QueryMetrics   # execution_time_ms, rows_scanned, used_index...
    baseline_metrics: QueryMetrics  # Original query metrics (for comparison)
    step_number: int
    max_steps: int
    cumulative_reward: float
    hint: Optional[str]        # Hint text if requested
    done: bool
```

### Reward Function (Dense, Shaped)

| Component | Weight | Description |
|---|---|---|
| Correctness | +0.40 | Result set identical to baseline (MD5 hash match) |
| Performance | +0.40 | Log-scaled speedup toward target (2x→0.10, 4x→0.20, 8x→0.30) |
| Plan cost | +0.20 | Query plan cost reduction from EXPLAIN |
| Syntax error | -0.10 | Penalty per invalid SQL submission |
| Hint used | -0.05 | Penalty per hint consumed |
| Step cost | -0.01 | Small per-step penalty for efficiency |

**Key design**: Reward is dense across the full trajectory — agents get signal at every step,
not just at episode end. Partial credit rewards partial progress.

---

## 📋 Tasks

### Task 1: Easy — Top Customers (Correlated Subquery)
**Target speedup**: 3x | **Max steps**: 8

The baseline query executes a correlated subquery twice per customer row to find top
spenders in 2023. With 2,000 customers, this is 4,000+ redundant subquery executions.

**Optimal solution**: Replace with a single CTE that pre-aggregates orders, then JOIN.

Expected techniques: `CTE`, `GROUP BY pre-aggregation`, `JOIN instead of subquery`

---

### Task 2: Medium — Product Revenue (Multi-Table Join)
**Target speedup**: 5x | **Max steps**: 12

Find top 20 products by revenue in two categories where average rating ≥ 4.0. The baseline
uses correlated subqueries for ratings and does full table scans on all joins.

**Optimal solution**: Pre-aggregate reviews in a CTE, add indexes, reorder joins to filter early.

Expected techniques: `CTE`, `pre-aggregation`, `HAVING clause`, `indexes`, `early filtering`

---

### Task 3: Hard — Cohort Retention (Window Functions)
**Target speedup**: 8x | **Max steps**: 15

Compute monthly cohort retention rates using repeated UNION ALL blocks with nested correlated
subqueries — causing quadratic complexity. This is a frontier-model-level challenge.

**Optimal solution**: Single-pass CTE for cohorts, month-offset computation, window functions
for cohort sizes, eliminating all UNION ALL repetition.

Expected techniques: `CTEs`, `window functions`, `PARTITION BY`, `single-pass aggregation`

---

## 🚀 Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-query-optimizer
cd sql-query-optimizer
pip install -r requirements.txt

# Run validation (OpenEnv compliance check)
python validate.py --verbose

# Run tests
pytest tests/ -v

# Start the Gradio demo UI
python demo.py

# Or start the FastAPI server only
uvicorn server:app --port 7860

# Or use the environment directly
python -c "
from env import SQLQueryOptimizerEnv
from env.models import Action

env = SQLQueryOptimizerEnv(difficulty='easy')
obs = env.reset()
print(obs.task_description)

action = Action(optimized_query='SELECT ...', declare_done=True)
obs, reward, done, info = env.step(action)
print(reward.explanation)
"
```

### Docker

```bash
docker build -t sql-query-optimizer .
docker run -p 7860:7860 sql-query-optimizer
```

### HTTP API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "optimized_query": "SELECT c.customer_id, c.name, c.country, ot.total_spend FROM customers c JOIN (SELECT customer_id, SUM(total_amount) AS total_spend FROM orders WHERE order_date >= '\''2023-01-01'\'' AND status = '\''delivered'\'' GROUP BY customer_id) ot ON ot.customer_id = c.customer_id ORDER BY ot.total_spend DESC LIMIT 10",
    "declare_done": true
  }'

# Get state
curl "http://localhost:7860/state?session_id=YOUR_SESSION_ID"

# Grade a submission
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{"easy_query": "...", "medium_query": "...", "hard_query": "..."}'
```

---

## 📊 Baseline Scores

Run the baseline inference script to reproduce scores:

```bash
export OPENAI_API_KEY=sk-...
python baseline_inference.py --model gpt-4o-mini --seed 42 --verbose
```

| Model | Easy | Medium | Hard | Aggregate |
|---|---|---|---|---|
| gpt-4o-mini | 0.72 | 0.48 | 0.21 | 0.45 |
| gpt-4o | 0.85 | 0.63 | 0.35 | 0.59 |

*(Scores are reproducible with `--seed 42`)*

---

## 📁 Project Structure

```
sql-query-optimizer/
├── env/
│   ├── __init__.py
│   ├── environment.py      # Core OpenEnv class (reset/step/state)
│   ├── models.py           # Pydantic models: Observation, Action, Reward
│   ├── database.py         # SQLite engine, schema, query execution
│   └── tasks/
│       ├── __init__.py
│       └── definitions.py  # Easy, Medium, Hard task specs
├── graders/
│   └── __init__.py         # EasyGrader, MediumGrader, HardGrader + run_all_graders
├── tests/
│   └── test_environment.py # Full pytest suite (40+ tests)
├── server.py               # FastAPI HTTP server
├── demo.py                 # Gradio interactive demo UI
├── validate.py             # OpenEnv compliance validation script
├── baseline_inference.py   # OpenAI API baseline agent
├── openenv.yaml            # Environment metadata
├── Dockerfile              # Container for HF Spaces
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🔬 Design Decisions

**Why SQLite?** Zero infrastructure dependencies — the environment runs anywhere with
no external database server. The in-memory DB ensures reproducibility across machines.

**Why shaped rewards?** Binary end-of-episode rewards create sparse gradients that slow
learning. Our reward gives signal at every step: correctness, speedup, and plan cost
all contribute independently, so partial progress is always rewarded.

**Why these three tasks?** They cover the three most common real-world SQL anti-patterns:
(1) correlated subqueries, (2) missing indexes on joins, (3) analytical queries that need
window functions. Together they span a genuine difficulty range from junior-engineer-level
to senior-DBA-level optimization challenges.

---

## License

MIT
