"""
Task definitions for the SQL Query Optimizer environment.

Each task defines:
  - A baseline (unoptimized) SQL query
  - A description of what needs to be optimized
  - Hints for the agent
  - The target speedup to achieve full score
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Task:
    task_id: str
    difficulty: str          # easy | medium | hard
    description: str
    baseline_query: str
    hints: List[str]
    target_speedup: float    # speedup ratio needed for full performance score
    max_steps: int
    optimization_techniques: List[str]  # what a perfect solution uses


# ──────────────────────────────────────────────
# EASY TASK
# N+1 style query: find top customers with subquery in SELECT
# ──────────────────────────────────────────────

EASY_TASK = Task(
    task_id="easy_top_customers",
    difficulty="easy",
    description=(
        "Find the top 10 customers by total order value in 2023, "
        "showing customer name, country, and their total spend. "
        "The current query uses a correlated subquery in the SELECT clause "
        "which executes once per customer row — optimize it."
    ),
    baseline_query="""
SELECT 
    c.customer_id,
    c.name,
    c.country,
    (
        SELECT SUM(o.total_amount)
        FROM orders o
        WHERE o.customer_id = c.customer_id
          AND o.order_date >= '2023-01-01'
          AND o.order_date < '2024-01-01'
          AND o.status = 'delivered'
    ) AS total_spend
FROM customers c
WHERE (
    SELECT SUM(o.total_amount)
    FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.order_date >= '2023-01-01'
      AND o.order_date < '2024-01-01'
      AND o.status = 'delivered'
) IS NOT NULL
ORDER BY total_spend DESC
LIMIT 10
""".strip(),
    hints=[
        "Correlated subqueries execute once per outer row. Consider rewriting using a JOIN.",
        "Aggregate orders first using GROUP BY, then join back to customers.",
        "Try: WITH order_totals AS (SELECT customer_id, SUM(...) FROM orders GROUP BY ...) SELECT ... JOIN ...",
    ],
    target_speedup=3.0,
    max_steps=8,
    optimization_techniques=["replace_correlated_subquery", "use_cte", "add_index"],
)


# ──────────────────────────────────────────────
# MEDIUM TASK
# Multi-table join with missing indexes + bad filter ordering
# ──────────────────────────────────────────────

MEDIUM_TASK = Task(
    task_id="medium_product_revenue",
    difficulty="medium",
    description=(
        "Find the top 20 products by total revenue where average rating >= 4.0, "
        "only for 'Electronics' and 'Sports' categories, ordered by revenue descending. "
        "The current query does a full scan of all tables with inefficient join order "
        "and computes ratings with a subquery. Optimize it to run as fast as possible."
    ),
    baseline_query="""
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    (
        SELECT AVG(r.rating)
        FROM reviews r
        WHERE r.product_id = p.product_id
    ) AS avg_rating,
    COUNT(DISTINCT oi.order_id) AS num_orders
FROM products p
JOIN order_items oi ON oi.product_id = p.product_id
JOIN orders o ON o.order_id = oi.order_id
WHERE p.category IN ('Electronics', 'Sports')
  AND o.status = 'delivered'
  AND (
        SELECT AVG(r.rating)
        FROM reviews r
        WHERE r.product_id = p.product_id
      ) >= 4.0
GROUP BY p.product_id, p.name, p.category, p.price
ORDER BY total_revenue DESC
LIMIT 20
""".strip(),
    hints=[
        "The correlated subquery for avg_rating runs thousands of times. Pre-aggregate reviews.",
        "Filter products by category early (before joining) to reduce the join size.",
        "Consider adding indexes on order_items(product_id) and reviews(product_id).",
        "Move the HAVING clause logic to a pre-aggregated CTE to avoid repeated computation.",
    ],
    target_speedup=5.0,
    max_steps=12,
    optimization_techniques=[
        "pre_aggregate_reviews",
        "early_filter",
        "replace_correlated_subquery",
        "add_indexes",
        "use_having_instead_of_where_subquery",
    ],
)


# ──────────────────────────────────────────────
# HARD TASK
# Complex analytics: cohort retention with window functions
# ──────────────────────────────────────────────

HARD_TASK = Task(
    task_id="hard_cohort_retention",
    difficulty="hard",
    description=(
        "Compute monthly cohort retention: for each month customers first ordered "
        "(their 'cohort month'), find what percentage also ordered in subsequent months "
        "(months 1, 2, 3 after cohort). Return cohort_month, retention_month (0,1,2,3), "
        "cohort_size, retained_count, and retention_rate. "
        "The current query uses multiple nested subqueries and self-joins that cause "
        "quadratic blowup. Optimize using window functions and/or CTEs. "
        "This is a frontier-model-level challenge requiring deep SQL optimization knowledge."
    ),
    baseline_query="""
SELECT 
    cohort.cohort_month,
    0 AS retention_month,
    COUNT(DISTINCT cohort.customer_id) AS cohort_size,
    COUNT(DISTINCT cohort.customer_id) AS retained_count,
    1.0 AS retention_rate
FROM (
    SELECT 
        c.customer_id,
        SUBSTR(MIN(o.order_date), 1, 7) AS cohort_month
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status != 'cancelled'
    GROUP BY c.customer_id
) cohort
GROUP BY cohort.cohort_month

UNION ALL

SELECT 
    cohort.cohort_month,
    1 AS retention_month,
    (
        SELECT COUNT(DISTINCT c2.customer_id)
        FROM customers c2
        JOIN orders o2 ON o2.customer_id = c2.customer_id
        WHERE o2.status != 'cancelled'
        GROUP BY c2.customer_id
        HAVING SUBSTR(MIN(o2.order_date), 1, 7) = cohort.cohort_month
    ) AS cohort_size,
    COUNT(DISTINCT retained.customer_id) AS retained_count,
    CAST(COUNT(DISTINCT retained.customer_id) AS REAL) / (
        SELECT COUNT(DISTINCT c3.customer_id)
        FROM customers c3
        JOIN orders o3 ON o3.customer_id = c3.customer_id
        WHERE o3.status != 'cancelled'
        GROUP BY c3.customer_id
        HAVING SUBSTR(MIN(o3.order_date), 1, 7) = cohort.cohort_month
    ) AS retention_rate
FROM (
    SELECT 
        c.customer_id,
        SUBSTR(MIN(o.order_date), 1, 7) AS cohort_month
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status != 'cancelled'
    GROUP BY c.customer_id
) cohort
JOIN orders retained_orders ON retained_orders.customer_id = cohort.customer_id
JOIN customers retained ON retained.customer_id = retained_orders.customer_id
WHERE retained_orders.status != 'cancelled'
  AND SUBSTR(retained_orders.order_date, 1, 7) > cohort.cohort_month
  AND (
    CAST(SUBSTR(retained_orders.order_date, 6, 2) AS INTEGER) 
    - CAST(SUBSTR(cohort.cohort_month, 6, 2) AS INTEGER) 
    + 12 * (CAST(SUBSTR(retained_orders.order_date, 1, 4) AS INTEGER) 
    - CAST(SUBSTR(cohort.cohort_month, 1, 4) AS INTEGER))
  ) = 1
GROUP BY cohort.cohort_month

ORDER BY cohort_month, retention_month
""".strip(),
    hints=[
        "Build the cohort CTE once: SELECT customer_id, MIN(order_date) as first_order grouped by customer_id.",
        "Use a second CTE to compute month-offset between cohort month and each subsequent order month.",
        "Use GROUP BY cohort_month, month_offset with COUNT(DISTINCT customer_id) to get retained counts.",
        "Compute cohort_size using a window function: COUNT(*) OVER (PARTITION BY cohort_month).",
        "Replace the UNION ALL with a single query using CASE or a cross-join on retention month values.",
    ],
    target_speedup=8.0,
    max_steps=15,
    optimization_techniques=[
        "use_cte_for_cohort",
        "eliminate_repeated_subqueries",
        "use_window_functions",
        "single_pass_aggregation",
        "add_indexes",
    ],
)


ALL_TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}
