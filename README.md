---
title: AI ATHENA META
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
---
# 🗄️ SQL Debug Environment

**An OpenEnv-compliant real-world environment for training AI agents to debug SQL queries.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://hub.docker.com)

---

## 🎯 What Is This?

SQL debugging is one of the most frequent real-world tasks data engineers, analysts, and backend developers face daily. A broken query with a cryptic error message is a perfect environment for testing an agent's ability to:

- Read and understand a relational schema
- Interpret database error messages
- Reason about SQL semantics (JOINs, GROUP BY, subqueries)
- Iteratively correct code with partial feedback

This environment simulates exactly that. An agent receives a **broken SQL query**, the **error message**, and the **database schema**, and must submit corrected SQL. Rewards are given for progressive improvements: syntax validity → execution → correct results.

---

## 📋 Environment Description

| Property | Value |
|---|---|
| Domain | Data Engineering / SQL |
| Interface | OpenEnv HTTP API (FastAPI) |
| Tasks | 3 (Easy → Medium → Hard) |
| Reward Type | Continuous [0.0, 1.0] per step |
| Max Steps | 5–7 per episode |
| Backend | SQLite (isolated per episode) |
| Port | 7860 |

---

## 🧩 Action Space

```python
class SQLDebugAction(BaseModel):
    corrected_sql: str   # Required: the agent's corrected SQL query
    explanation: str     # Optional: brief description of changes made
```

**Example action:**
```json
{
  "corrected_sql": "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC",
  "explanation": "Added missing FROM keyword and ORDER BY keyword"
}
```

---

## 👁️ Observation Space

```python
class SQLDebugObservation(BaseModel):
    task_id: str                          # Task name identifier
    task_difficulty: str                  # "easy" | "medium" | "hard"
    broken_sql: str                       # The SQL query with bugs
    error_message: str                    # Error from DB execution
    db_schema: List[TableSchema]          # Available tables & columns
    expected_result_description: str      # Plain-English goal
    attempt_number: int                   # Current attempt (1-indexed)
    max_attempts: int                     # Episode step limit
    last_execution_result: Optional[str]  # Output of last query attempt
    partial_score: float                  # Running score [0.0, 1.0]
```

---

## 📊 Reward Function

The reward function provides **continuous signal at every step**, not just at episode end:

| Milestone | Reward Contribution |
|---|---|
| SQL syntax valid | +0.10 |
| Query executes without error | +0.20 |
| Result set matches expected | +0.55 |
| Key clauses present (structural) | +0.10 |
| Exact match to reference solution | +0.05 bonus |
| Later attempt penalty | −5% per extra attempt |

This design encourages agents to:
1. Fix syntax first (quick wins)
2. Achieve execution without crashes
3. Get the logic right
4. Do it in fewer attempts

**Reward is always in [0.0, 1.0].** Penalizes brute-force guessing via attempt decay.

---

## 🎮 Tasks

### Task 1: `easy_syntax_fix` (Easy)
- **What's broken:** Missing `FROM` keyword, missing `ORDER BY` keyword
- **Schema:** `employees(id, name, department, salary)`
- **Goal:** Return Engineering employees sorted by salary descending
- **Max steps:** 5
- **Expected frontier model score:** ~0.95

**Broken SQL:**
```sql
SELECT name, salary employees WHERE department = 'Engineering' ORDER salary DESC;
```
**Error:** `near "employees": syntax error`

---

### Task 2: `medium_logic_fix` (Medium)
- **What's broken:** `GROUP BY` uses `order_id` instead of `customer_id` — query runs but produces wrong results (each row = one order, not customer total)
- **Schema:** `customers(customer_id, name, email)` + `orders(order_id, customer_id, amount, status)`
- **Goal:** Total spending per customer on completed orders, sorted descending
- **Max steps:** 5
- **Expected frontier model score:** ~0.75

**Broken SQL:**
```sql
SELECT c.name, SUM(o.amount) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY o.order_id   -- BUG: should be c.customer_id, c.name
ORDER BY total_spent DESC;
```
**Error:** Executes but returns wrong results (one row per order, not per customer)

---

### Task 3: `hard_optimization_fix` (Hard)
- **What's broken:** 3 column name errors (`p.name` → `p.product_name`, `s.qty` → `s.quantity`, `p.id` → `p.product_id`)
- **Schema:** `products`, `sales`, `inventory` (3-table join)
- **Goal:** Revenue by product for categories with >1 product, with inventory data
- **Max steps:** 7
- **Expected frontier model score:** ~0.60

**Broken SQL has 3 simultaneous errors** — tests whether agents can find and fix all of them, not just the first one.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Development

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-debug-env
cd sql-debug-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 4. Test it works
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_syntax_fix"}'
```

### Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

### Docker

```bash
# Build
docker build -t sql-debug-env .

# Run
docker run -p 7860:7860 sql-debug-env

# Test the running container
curl http://localhost:7860/health
```

---

## 🤖 HTTP API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/reset` | POST | Start a new episode, returns initial observation |
| `/step` | POST | Submit corrected SQL, returns reward + next obs |
| `/state` | GET/POST | Get current episode metadata |
| `/tasks` | GET | List all available tasks |
| `/` | GET | Interactive web demo UI |

### Example: Full Episode via curl

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_syntax_fix", "session_id": "my-session"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "corrected_sql": "SELECT name, salary FROM employees WHERE department = '\''Engineering'\'' ORDER BY salary DESC"
  }'
```

---

## 🚀 Baseline Inference Script

```bash
# Set environment variables
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

# Run against all 3 tasks
python inference.py
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score | Notes |
|---|---|---|
| `easy_syntax_fix` | ~0.92 | Frontier models solve easily |
| `medium_logic_fix` | ~0.71 | Logic errors harder to spot |
| `hard_optimization_fix` | ~0.58 | Multiple simultaneous bugs are challenging |
| **Average** | **~0.74** | |

---

## 📁 Project Structure

```
sql-debug-env/
├── openenv.yaml          # OpenEnv spec metadata
├── models.py             # Pydantic typed models (Action, Observation, Reward, State)
├── tasks.py              # Task definitions + grader functions
├── inference.py          # Baseline inference script (OpenAI client)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── README.md             # This file
├── server/
│   ├── __init__.py
│   ├── app.py            # FastAPI HTTP server
│   └── environment.py    # Core Environment class (reset/step/state)
└── tests/
    ├── __init__.py
    └── test_environment.py  # 18 unit tests
```

---

## 🏆 Why This Environment?

1. **Novel domain** — SQL debugging isn't in the current OpenEnv catalog
2. **Real daily task** — Every data engineer debugs SQL queries constantly
3. **Rich reward signal** — 5-tier grading rewards partial progress, not just pass/fail
4. **Meaningful difficulty curve** — Easy (syntax), Medium (logic), Hard (multi-bug)
5. **Deterministic grading** — SQLite execution guarantees reproducible scores
6. **Isolated episodes** — Each episode gets its own fresh SQLite DB file

---

## 📄 License

Apache 2.0
