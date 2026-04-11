"""
Task definitions for the SQL Debug Environment.
Each task has:
  - broken_sql: the query that needs fixing
  - fixed_sql: the reference correct answer
  - schema: table definitions
  - grader: function returning float in [0.0, 1.0]
  - difficulty: easy | medium | hard
"""
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


# ─── Schema helpers ───────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, dict] = {}


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, collapse whitespace."""
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    # remove trailing semicolons for comparison
    sql = sql.rstrip(";").strip()
    return sql


def _run_sql_on_db(db_path: str, sql: str) -> Tuple[bool, Optional[List], Optional[str]]:
    """
    Run a SQL query against a sqlite DB.
    Returns (success, rows, error_message).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


# ─── TASK 1: Easy – Syntax Fix ────────────────────────────────────────────────

TASK_EASY = {
    "name": "easy_syntax_fix",
    "difficulty": "easy",
    "description": (
        "Fix a SQL query with simple syntax errors: a missing keyword and "
        "incorrect use of single-equals for alias."
    ),
    "schema": [
        {
            "table_name": "employees",
            "columns": [
                {"name": "id", "type": "INTEGER PRIMARY KEY"},
                {"name": "name", "type": "TEXT"},
                {"name": "department", "type": "TEXT"},
                {"name": "salary", "type": "REAL"},
            ],
            "sample_rows": [
                {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000},
                {"id": 2, "name": "Bob", "department": "Marketing", "salary": 72000},
                {"id": 3, "name": "Carol", "department": "Engineering", "salary": 88000},
            ],
        }
    ],
    "create_sql": """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary REAL
        );
        INSERT OR IGNORE INTO employees VALUES (1, 'Alice', 'Engineering', 95000);
        INSERT OR IGNORE INTO employees VALUES (2, 'Bob', 'Marketing', 72000);
        INSERT OR IGNORE INTO employees VALUES (3, 'Carol', 'Engineering', 88000);
        INSERT OR IGNORE INTO employees VALUES (4, 'Dave', 'HR', 65000);
        INSERT OR IGNORE INTO employees VALUES (5, 'Eve', 'Engineering', 102000);
    """,
    # BROKEN: missing FROM, = instead of AS
    "broken_sql": "SELECT name, salary employees WHERE department = 'Engineering' ORDER salary DESC;",
    "error_message": "near \"employees\": syntax error",
    "fixed_sql": "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC",
    "expected_result_description": (
        "Return the name and salary of all Engineering department employees, "
        "sorted by salary in descending order."
    ),
    "expected_rows": [
        ("Eve", 102000),
        ("Alice", 95000),
        ("Carol", 88000),
    ],
    "max_steps": 5,
}


# ─── TASK 2: Medium – Logic Fix ───────────────────────────────────────────────

TASK_MEDIUM = {
    "name": "medium_logic_fix",
    "difficulty": "medium",
    "description": (
        "Fix a SQL query with logic errors: wrong JOIN type causes missing rows, "
        "and the GROUP BY is incorrect."
    ),
    "schema": [
        {
            "table_name": "orders",
            "columns": [
                {"name": "order_id", "type": "INTEGER PRIMARY KEY"},
                {"name": "customer_id", "type": "INTEGER"},
                {"name": "amount", "type": "REAL"},
                {"name": "status", "type": "TEXT"},
            ],
            "sample_rows": [
                {"order_id": 1, "customer_id": 1, "amount": 250.0, "status": "completed"},
                {"order_id": 2, "customer_id": 2, "amount": 89.5, "status": "pending"},
                {"order_id": 3, "customer_id": 1, "amount": 430.0, "status": "completed"},
            ],
        },
        {
            "table_name": "customers",
            "columns": [
                {"name": "customer_id", "type": "INTEGER PRIMARY KEY"},
                {"name": "name", "type": "TEXT"},
                {"name": "email", "type": "TEXT"},
            ],
            "sample_rows": [
                {"customer_id": 1, "name": "Alice", "email": "alice@example.com"},
                {"customer_id": 2, "name": "Bob", "email": "bob@example.com"},
                {"customer_id": 3, "name": "Carol", "email": "carol@example.com"},
            ],
        },
    ],
    "create_sql": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount REAL,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        INSERT OR IGNORE INTO customers VALUES (1, 'Alice', 'alice@example.com');
        INSERT OR IGNORE INTO customers VALUES (2, 'Bob', 'bob@example.com');
        INSERT OR IGNORE INTO customers VALUES (3, 'Carol', 'carol@example.com');
        INSERT OR IGNORE INTO orders VALUES (1, 1, 250.00, 'completed');
        INSERT OR IGNORE INTO orders VALUES (2, 2, 89.50, 'pending');
        INSERT OR IGNORE INTO orders VALUES (3, 1, 430.00, 'completed');
        INSERT OR IGNORE INTO orders VALUES (4, 3, 175.00, 'completed');
        INSERT OR IGNORE INTO orders VALUES (5, 2, 320.00, 'completed');
    """,
    # BROKEN: INNER JOIN excludes Carol (no orders for carol not in completed);
    # also GROUP BY order_id instead of customer_id; also sums all not just completed
    "broken_sql": """
        SELECT c.name, SUM(o.amount) AS total_spent
        FROM customers c
        INNER JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.status = 'completed'
        GROUP BY o.order_id
        ORDER BY total_spent DESC;
    """,
    "error_message": (
        "Query executes but returns wrong results: groups by order_id instead of customer, "
        "so each row is a single order, not the customer total. Carol is missing (no completed orders? "
        "Actually Carol has order 4 completed — she should appear). Also customers with 0 completed "
        "orders should be excluded but Carol should be included."
    ),
    "fixed_sql": (
        "SELECT c.name, SUM(o.amount) AS total_spent "
        "FROM customers c "
        "INNER JOIN orders o ON c.customer_id = o.customer_id "
        "WHERE o.status = 'completed' "
        "GROUP BY c.customer_id, c.name "
        "ORDER BY total_spent DESC"
    ),
    "expected_result_description": (
        "Return each customer's name and their total spending on completed orders only, "
        "grouped by customer (not by order), sorted by total_spent descending. "
        "Customers with no completed orders should be excluded."
    ),
    "expected_rows": [
        ("Alice", 680.0),
        ("Bob", 320.0),
        ("Carol", 175.0),
    ],
    "max_steps": 5,
}


# ─── TASK 3: Hard – Multi-Error + Subquery Fix ────────────────────────────────

TASK_HARD = {
    "name": "hard_optimization_fix",
    "difficulty": "hard",
    "description": (
        "Fix a complex SQL query with multiple errors: wrong column name, "
        "incorrect subquery correlation, missing HAVING clause, and a schema mismatch."
    ),
    "schema": [
        {
            "table_name": "products",
            "columns": [
                {"name": "product_id", "type": "INTEGER PRIMARY KEY"},
                {"name": "product_name", "type": "TEXT"},
                {"name": "category", "type": "TEXT"},
                {"name": "price", "type": "REAL"},
            ],
            "sample_rows": [
                {"product_id": 1, "product_name": "Laptop", "category": "Electronics", "price": 999.99},
                {"product_id": 2, "product_name": "Desk Chair", "category": "Furniture", "price": 249.99},
            ],
        },
        {
            "table_name": "sales",
            "columns": [
                {"name": "sale_id", "type": "INTEGER PRIMARY KEY"},
                {"name": "product_id", "type": "INTEGER"},
                {"name": "quantity", "type": "INTEGER"},
                {"name": "sale_date", "type": "TEXT"},
                {"name": "revenue", "type": "REAL"},
            ],
            "sample_rows": [
                {"sale_id": 1, "product_id": 1, "quantity": 5, "sale_date": "2024-01-15", "revenue": 4999.95},
            ],
        },
        {
            "table_name": "inventory",
            "columns": [
                {"name": "product_id", "type": "INTEGER PRIMARY KEY"},
                {"name": "stock_quantity", "type": "INTEGER"},
                {"name": "warehouse", "type": "TEXT"},
                {"name": "last_updated", "type": "TEXT"},
            ],
            "sample_rows": [
                {"product_id": 1, "stock_quantity": 45, "warehouse": "WH-A", "last_updated": "2024-01-20"},
            ],
        },
    ],
    "create_sql": """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            price REAL
        );
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            sale_date TEXT,
            revenue REAL
        );
        CREATE TABLE IF NOT EXISTS inventory (
            product_id INTEGER PRIMARY KEY,
            stock_quantity INTEGER,
            warehouse TEXT,
            last_updated TEXT
        );
        INSERT OR IGNORE INTO products VALUES (1, 'Laptop', 'Electronics', 999.99);
        INSERT OR IGNORE INTO products VALUES (2, 'Desk Chair', 'Furniture', 249.99);
        INSERT OR IGNORE INTO products VALUES (3, 'Monitor', 'Electronics', 399.99);
        INSERT OR IGNORE INTO products VALUES (4, 'Keyboard', 'Electronics', 79.99);
        INSERT OR IGNORE INTO products VALUES (5, 'Standing Desk', 'Furniture', 599.99);
        INSERT OR IGNORE INTO sales VALUES (1, 1, 5, '2024-01-15', 4999.95);
        INSERT OR IGNORE INTO sales VALUES (2, 1, 3, '2024-02-10', 2999.97);
        INSERT OR IGNORE INTO sales VALUES (3, 2, 2, '2024-01-20', 499.98);
        INSERT OR IGNORE INTO sales VALUES (4, 3, 8, '2024-01-25', 3199.92);
        INSERT OR IGNORE INTO sales VALUES (5, 3, 4, '2024-02-05', 1599.96);
        INSERT OR IGNORE INTO sales VALUES (6, 4, 15, '2024-01-30', 1199.85);
        INSERT OR IGNORE INTO sales VALUES (7, 4, 10, '2024-02-12', 799.90);
        INSERT OR IGNORE INTO inventory VALUES (1, 45, 'WH-A', '2024-01-20');
        INSERT OR IGNORE INTO inventory VALUES (2, 12, 'WH-B', '2024-01-21');
        INSERT OR IGNORE INTO inventory VALUES (3, 8, 'WH-A', '2024-01-22');
        INSERT OR IGNORE INTO inventory VALUES (4, 120, 'WH-C', '2024-01-23');
        INSERT OR IGNORE INTO inventory VALUES (5, 3, 'WH-B', '2024-01-24');
    """,
    # BROKEN: 4 bugs:
    # 1. p.name should be p.product_name
    # 2. SUM(s.qty) should be SUM(s.quantity)
    # 3. Subquery references i.product_id = p.id (wrong: should be p.product_id)
    # 4. Missing HAVING to filter categories with >1 product sold
    "broken_sql": """
        SELECT
            p.category,
            p.name AS product_name,
            SUM(s.qty) AS total_units_sold,
            SUM(s.revenue) AS total_revenue,
            i.stock_quantity
        FROM products p
        JOIN sales s ON p.product_id = s.product_id
        JOIN inventory i ON i.product_id = p.id
        WHERE p.category IN (
            SELECT category FROM products
            GROUP BY category
            HAVING COUNT(*) > 1
        )
        GROUP BY p.category, p.name, i.stock_quantity
        ORDER BY total_revenue DESC;
    """,
    "error_message": (
        "no such column: p.name — also 'qty' is not a column in sales (should be 'quantity'), "
        "and 'p.id' is not a valid column (should be 'p.product_id')"
    ),
    "fixed_sql": (
        "SELECT p.category, p.product_name, SUM(s.quantity) AS total_units_sold, "
        "SUM(s.revenue) AS total_revenue, i.stock_quantity "
        "FROM products p "
        "JOIN sales s ON p.product_id = s.product_id "
        "JOIN inventory i ON i.product_id = p.product_id "
        "WHERE p.category IN ("
        "SELECT category FROM products GROUP BY category HAVING COUNT(*) > 1"
        ") "
        "GROUP BY p.category, p.product_name, i.stock_quantity "
        "ORDER BY total_revenue DESC"
    ),
    "expected_result_description": (
        "Return category, product name, total units sold, total revenue, and current stock "
        "for all products in categories that have more than one product. "
        "Join products with sales and inventory. Order by total revenue descending."
    ),
    "expected_rows": None,  # Checked by execution match logic below
    "max_steps": 7,
}

TASKS = {
    "easy_syntax_fix": TASK_EASY,
    "medium_logic_fix": TASK_MEDIUM,
    "hard_optimization_fix": TASK_HARD,
}


# ─── Grader ───────────────────────────────────────────────────────────────────

def grade_submission(
    task_name: str,
    submitted_sql: str,
    db_path: str,
    attempt_number: int,
    max_attempts: int,
) -> Dict[str, Any]:
    """
    Grade a submitted SQL query against the task specification.
    Returns a dict with: reward (float), feedback, and individual signal flags.
    """
    task = TASKS[task_name]
    fixed_sql = task["fixed_sql"]
    difficulty = task["difficulty"]

    syntax_valid = False
    executes = False
    results_match = False
    is_optimal = False
    feedback_parts = []

    # Attempt penalty: reward decays slightly each attempt to encourage getting it right early
    attempt_penalty = max(0.0, (attempt_number - 1) * 0.05)

    # ── Step 1: Basic syntax check ───────────────────────────────────────────
    # We attempt to compile the query on the real DB using EXPLAIN QUERY PLAN.
    # This validates syntax without caring about table existence at parse time.
    # We use a different approach: try to prepare the statement via sqlite3.
    try:
        conn = sqlite3.connect(db_path)
        # EXPLAIN QUERY PLAN parses and plans without returning rows
        conn.execute(f"EXPLAIN QUERY PLAN {submitted_sql}")
        conn.close()
        syntax_valid = True
        feedback_parts.append("✓ SQL syntax is valid.")
    except sqlite3.OperationalError as e:
        err_str = str(e)
        # If error is "no such table" or "no such column" that means syntax was OK
        # but schema mismatch — still counts as a syntax-level pass
        if any(phrase in err_str for phrase in ["no such table", "no such column", "ambiguous"]):
            syntax_valid = True
            feedback_parts.append(f"✓ SQL parses correctly (schema issue: {err_str}).")
        else:
            feedback_parts.append(f"✗ Syntax error: {err_str}")
            raw_reward = max(0.0, 0.05 - attempt_penalty)
            return {
                "reward": raw_reward,
                "syntax_valid": False,
                "executes_without_error": False,
                "results_match": False,
                "is_optimal": False,
                "feedback": " ".join(feedback_parts),
                "execution_result": err_str,
            }
    except Exception as e:
        feedback_parts.append(f"✗ Syntax error: {e}")
        raw_reward = max(0.0, 0.05 - attempt_penalty)
        return {
            "reward": raw_reward,
            "syntax_valid": False,
            "executes_without_error": False,
            "results_match": False,
            "is_optimal": False,
            "feedback": " ".join(feedback_parts),
            "execution_result": str(e),
        }

    # ── Step 2: Execute against real DB ─────────────────────────────────────
    success, submitted_rows, exec_error = _run_sql_on_db(db_path, submitted_sql)
    if not success:
        feedback_parts.append(f"✗ Execution error: {exec_error}")
        raw_reward = max(0.0, 0.20 - attempt_penalty)  # syntax ok but runtime error
        return {
            "reward": raw_reward,
            "syntax_valid": True,
            "executes_without_error": False,
            "results_match": False,
            "is_optimal": False,
            "feedback": " ".join(feedback_parts),
            "execution_result": str(exec_error),
        }

    executes = True
    feedback_parts.append("✓ Query executes without error.")

    # ── Step 3: Compare results against reference query ──────────────────────
    _, reference_rows, _ = _run_sql_on_db(db_path, fixed_sql)

    # Normalize both for comparison (convert to sorted list of tuples)
    def normalize_rows(rows):
        if rows is None:
            return []
        return sorted([tuple(str(v) for v in row) for row in rows])

    submitted_norm = normalize_rows(submitted_rows)
    reference_norm = normalize_rows(reference_rows)

    if submitted_norm == reference_norm:
        results_match = True
        feedback_parts.append("✓ Results match expected output exactly.")
    elif len(submitted_norm) == len(reference_norm):
        feedback_parts.append(
            f"~ Correct row count ({len(submitted_norm)}) but values differ. "
            "Check aggregations or column selection."
        )
    else:
        feedback_parts.append(
            f"✗ Result mismatch: got {len(submitted_norm)} rows, expected {len(reference_norm)} rows."
        )

    # ── Step 4: Structural similarity to reference (optional bonus) ──────────
    submitted_norm_sql = _normalize_sql(submitted_sql)
    reference_norm_sql = _normalize_sql(fixed_sql)

    # Check if key clauses are present
    key_tokens_correct = 0
    key_tokens_total = 0
    for token in ["join", "group by", "order by", "having", "where"]:
        if token in reference_norm_sql:
            key_tokens_total += 1
            if token in submitted_norm_sql:
                key_tokens_correct += 1

    structural_score = key_tokens_correct / key_tokens_total if key_tokens_total > 0 else 1.0

    if submitted_norm_sql == reference_norm_sql:
        is_optimal = True
        feedback_parts.append("✓ Query matches reference solution exactly.")

    # ── Compute final reward ─────────────────────────────────────────────────
    # Reward breakdown:
    #   - syntax valid:    0.10
    #   - executes:        0.20
    #   - results match:   0.55
    #   - structural fit:  0.10 (partial)
    #   - optimal match:   0.05 bonus
    raw_reward = 0.0
    raw_reward += 0.10  # already passed syntax
    raw_reward += 0.20  # already passed execution
    if results_match:
        raw_reward += 0.55
    else:
        # partial credit for structural correctness
        raw_reward += 0.15 * structural_score
    raw_reward += 0.10 * structural_score
    if is_optimal:
        raw_reward += 0.05

    # Apply attempt penalty (max 25% reduction)
    raw_reward = raw_reward * max(0.75, 1.0 - attempt_penalty)

    # Difficulty multiplier — hard tasks earn same max but graded tighter
    if difficulty == "hard" and not results_match:
        raw_reward *= 0.9

    final_reward = round(min(1.0, max(0.0, raw_reward)), 4)

    return {
        "reward": final_reward,
        "syntax_valid": syntax_valid,
        "executes_without_error": executes,
        "results_match": results_match,
        "is_optimal": is_optimal,
        "feedback": " ".join(feedback_parts),
        "execution_result": str(submitted_rows[:5]) if submitted_rows else "[]",
    }
