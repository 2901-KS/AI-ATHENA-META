"""
Database engine for the SQL Query Optimizer environment.

Uses SQLite (in-memory or file-based) with a realistic e-commerce schema.
Provides query execution, EXPLAIN plan parsing, and result hashing.
"""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from .models import QueryMetrics


# ──────────────────────────────────────────────
# Schema Definition
# ──────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    country       TEXT NOT NULL,
    signup_date   TEXT NOT NULL,
    tier          TEXT NOT NULL DEFAULT 'standard'  -- standard, premium, enterprise
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    price         REAL NOT NULL,
    stock         INTEGER NOT NULL DEFAULT 0,
    supplier_id   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending, shipped, delivered, cancelled
    total_amount  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_date   TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0
);

-- No indexes by default (agent must discover that indexes help)
"""

INDEXES_SQL = """
-- These are NOT created initially; the agent can suggest them
-- CREATE INDEX idx_orders_customer ON orders(customer_id);
-- CREATE INDEX idx_orders_date ON orders(order_date);
-- CREATE INDEX idx_order_items_order ON order_items(order_id);
-- CREATE INDEX idx_order_items_product ON order_items(product_id);
-- CREATE INDEX idx_reviews_product ON reviews(product_id);
-- CREATE INDEX idx_customers_country ON customers(country);
-- CREATE INDEX idx_products_category ON products(category);
"""

AVAILABLE_INDEXES = {
    "idx_orders_customer": "CREATE INDEX idx_orders_customer ON orders(customer_id)",
    "idx_orders_date": "CREATE INDEX idx_orders_date ON orders(order_date)",
    "idx_order_items_order": "CREATE INDEX idx_order_items_order ON order_items(order_id)",
    "idx_order_items_product": "CREATE INDEX idx_order_items_product ON order_items(product_id)",
    "idx_reviews_product": "CREATE INDEX idx_reviews_product ON reviews(product_id)",
    "idx_customers_country": "CREATE INDEX idx_customers_country ON customers(country)",
    "idx_products_category": "CREATE INDEX idx_products_category ON products(category)",
}


# ──────────────────────────────────────────────
# Data Generation
# ──────────────────────────────────────────────

def _generate_data(conn: sqlite3.Connection, seed: int = 42) -> None:
    """Populate DB with realistic synthetic e-commerce data."""
    import random
    rng = random.Random(seed)

    countries = ["US", "UK", "DE", "FR", "IN", "CA", "AU", "JP", "BR", "SG"]
    tiers = ["standard", "standard", "standard", "premium", "enterprise"]
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty", "Toys"]
    statuses = ["pending", "shipped", "delivered", "delivered", "delivered", "cancelled"]

    # Customers (2000)
    customers = []
    for i in range(1, 2001):
        year = rng.randint(2018, 2023)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        customers.append((
            i, f"Customer {i}", f"customer{i}@example.com",
            rng.choice(countries), f"{year}-{month:02d}-{day:02d}",
            rng.choice(tiers)
        ))
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?)", customers
    )

    # Products (500)
    products = []
    for i in range(1, 501):
        products.append((
            i, f"Product {i}", rng.choice(categories),
            round(rng.uniform(5.0, 999.99), 2),
            rng.randint(0, 500), rng.randint(1, 50)
        ))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

    # Orders (10000)
    orders = []
    for i in range(1, 10001):
        year = rng.randint(2020, 2024)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        orders.append((
            i, rng.randint(1, 2000), f"{year}-{month:02d}-{day:02d}",
            rng.choice(statuses), round(rng.uniform(10.0, 5000.0), 2)
        ))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)

    # Order Items (30000)
    items = []
    for i in range(1, 30001):
        items.append((
            i, rng.randint(1, 10000), rng.randint(1, 500),
            rng.randint(1, 5), round(rng.uniform(5.0, 999.99), 2)
        ))
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    # Reviews (8000)
    reviews = []
    for i in range(1, 8001):
        year = rng.randint(2020, 2024)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        reviews.append((
            i, rng.randint(1, 500), rng.randint(1, 2000),
            rng.randint(1, 5), f"{year}-{month:02d}-{day:02d}",
            rng.randint(0, 1)
        ))
    conn.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?)", reviews)

    conn.commit()


# ──────────────────────────────────────────────
# Database Engine
# ──────────────────────────────────────────────

class DatabaseEngine:
    """
    Manages the SQLite connection pool and provides query execution,
    EXPLAIN plan analysis, and result hashing utilities.
    """

    def __init__(self, db_path: str = ":memory:", seed: int = 42):
        self.db_path = db_path
        self.seed = seed
        self._conn: Optional[sqlite3.Connection] = None
        self._active_indexes: set = set()
        self._initialize()

    def _initialize(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=10000")
        self._conn.executescript(SCHEMA_SQL)
        _generate_data(self._conn, self.seed)

    def reset(self) -> None:
        """Drop all user-created indexes (resets environment state)."""
        for idx_name in list(self._active_indexes):
            try:
                self._conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
            except Exception:
                pass
        self._active_indexes.clear()
        self._conn.commit()

    def apply_index(self, index_name: str) -> bool:
        """Apply one of the predefined indexes. Returns True if successful."""
        if index_name not in AVAILABLE_INDEXES:
            return False
        if index_name in self._active_indexes:
            return True  # already applied
        try:
            self._conn.execute(AVAILABLE_INDEXES[index_name])
            self._conn.commit()
            self._active_indexes.add(index_name)
            return True
        except Exception:
            return False

    def execute_query(self, sql: str, timeout_ms: float = 5000) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Execute a SQL query and return (rows, error).
        Returns None rows on error.
        """
        try:
            sql_clean = sql.strip()
            # Safety: only allow SELECT statements
            if not re.match(r"^\s*SELECT", sql_clean, re.IGNORECASE):
                return None, "Only SELECT statements are allowed."
            cursor = self._conn.execute(sql_clean)
            rows = [dict(row) for row in cursor.fetchall()]
            return rows, None
        except sqlite3.OperationalError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)

    def measure_query(self, sql: str, runs: int = 3) -> Tuple[Optional[QueryMetrics], Optional[str]]:
        """
        Execute query multiple times and return averaged performance metrics.
        """
        rows, error = self.execute_query(sql)
        if error:
            return None, error

        # Measure execution time (average of `runs`)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self._conn.execute(sql).fetchall()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        # EXPLAIN QUERY PLAN for cost & index usage
        plan_cost, used_index, rows_scanned = self._analyze_plan(sql)

        # Hash result set for correctness checking
        result_hash = self._hash_results(rows)

        return QueryMetrics(
            execution_time_ms=round(avg_time, 3),
            rows_scanned=rows_scanned,
            rows_returned=len(rows),
            query_plan_cost=plan_cost,
            used_index=used_index,
            result_hash=result_hash,
        ), None

    def _analyze_plan(self, sql: str) -> Tuple[float, bool, int]:
        """Parse EXPLAIN QUERY PLAN output."""
        try:
            cursor = self._conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_rows = cursor.fetchall()
            plan_text = " ".join(str(dict(r)) for r in plan_rows)

            used_index = "USING INDEX" in plan_text or "USING COVERING INDEX" in plan_text
            # Estimate cost from scan mentions
            full_scans = plan_text.count("SCAN")
            index_scans = plan_text.count("SEARCH")

            # Heuristic cost: full scans are expensive
            cost = full_scans * 1000.0 + index_scans * 10.0

            # Estimate rows scanned (heuristic)
            rows_scanned = full_scans * 5000 + index_scans * 100

            return cost, used_index, rows_scanned
        except Exception:
            return 9999.0, False, 99999

    def _hash_results(self, rows: List[Dict]) -> str:
        """Produce a deterministic hash of a result set."""
        serialized = json.dumps(rows, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get_schema_info(self) -> Dict[str, Any]:
        """Return schema metadata for the observation."""
        schema = {}
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            col_cursor = self._conn.execute(f"PRAGMA table_info({table})")
            columns = [
                {"name": row[1], "type": row[2], "not_null": bool(row[3]), "pk": bool(row[5])}
                for row in col_cursor.fetchall()
            ]
            count_cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = count_cursor.fetchone()[0]

            idx_cursor = self._conn.execute(f"PRAGMA index_list({table})")
            indexes = [row[1] for row in idx_cursor.fetchall()]

            schema[table] = {
                "columns": columns,
                "row_count": row_count,
                "indexes": indexes,
            }

        schema["_available_indexes"] = list(AVAILABLE_INDEXES.keys())
        schema["_active_indexes"] = list(self._active_indexes)
        return schema

    def get_sample_data(self) -> Dict[str, List[Dict]]:
        """Return up to 3 rows per table for agent context."""
        samples = {}
        for table in ["customers", "products", "orders", "order_items", "reviews"]:
            rows, _ = self.execute_query(f"SELECT * FROM {table} LIMIT 3")
            samples[table] = rows or []
        return samples

    def close(self) -> None:
        if self._conn:
            self._conn.close()
