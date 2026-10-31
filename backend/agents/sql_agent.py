"""
agents/sql_agent.py
--------------------
Translates a natural-language question into a safe SQL SELECT query,
executes it against the local SQLite database, and returns a plain-English answer.

Flow:
  1. Ask LLM to generate SQL from the question + schema description
  2. Validate: only allow SELECT statements
  3. Execute on SQLite
  4. Ask LLM to turn the raw result into a friendly sentence
"""
import re
import sqlite3
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend.config.settings import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Path to the SQLite database created by database/sample_data.py
DB_PATH = settings.DATA_DIR / "business.db"

# Human-readable schema description given to the LLM so it knows what tables exist
SCHEMA_DESCRIPTION = """
Database: business.db (SQLite)

Tables:
1. customers(id, name, email, city, country, created_at)
2. products(id, name, category, price, stock_quantity)
3. sales(id, customer_id, product_id, quantity, total_amount, sale_date)
4. employees(id, name, department, salary, hire_date, manager_id)

Relationships:
- sales.customer_id  → customers.id
- sales.product_id   → products.id
- employees.manager_id → employees.id (self-referencing)

Notes:
- Dates are stored as TEXT in ISO-8601 format (YYYY-MM-DD).
- total_amount = quantity × price at the time of sale.
"""


# ─── Safety ──────────────────────────────────────────────────────────────────

DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def _is_safe_query(sql: str) -> bool:
    """Return True only if the query is a plain SELECT with no dangerous keywords."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return False
    if DANGEROUS_KEYWORDS.search(sql):
        return False
    return True


def _extract_sql(raw: str) -> str:
    """Pull the SQL out of a markdown code block if the LLM wrapped it."""
    # Match ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Otherwise assume the whole string is SQL
    return raw.strip()


# ─── Core functions ──────────────────────────────────────────────────────────

def generate_sql(question: str) -> str:
    """Ask the LLM to produce a SQL SELECT query for the given question."""
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert SQL assistant. Given a natural-language question "
                    "and a database schema, produce ONE valid SQLite SELECT query that answers it. "
                    "Return ONLY the SQL query — no explanations, no markdown, no comments.\n\n"
                    f"Schema:\n{SCHEMA_DESCRIPTION}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return _extract_sql(raw)


def execute_sql(sql: str) -> list[dict[str, Any]]:
    """Run the SQL query against the SQLite database and return rows as dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def explain_results(question: str, sql: str, rows: list[dict]) -> str:
    """Ask the LLM to convert raw SQL results into a human-friendly answer."""
    if not rows:
        return "The query returned no results."

    # Limit to first 20 rows to keep the prompt manageable
    rows_preview = rows[:20]
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful data analyst. Given a user question, the SQL query used, "
                    "and its result rows (as JSON), write a concise, clear natural-language answer. "
                    "Mention key numbers and insights. Do not show raw JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"SQL used:\n{sql}\n\n"
                    f"Result rows (up to 20):\n{rows_preview}"
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ─── Main entry point ────────────────────────────────────────────────────────

def run(question: str) -> dict:
    """
    Full SQL-agent pipeline.

    Returns:
      {
        "answer":        str,   # plain-English response
        "sql_query":     str,   # the generated SQL (shown in UI)
        "raw_results":   list,  # raw rows from DB (shown in UI)
        "error":         str | None
      }
    """
    try:
        sql = generate_sql(question)

        if not _is_safe_query(sql):
            return {
                "answer": (
                    "I generated a query that isn't a safe SELECT statement. "
                    "For safety, only read-only SELECT queries are allowed."
                ),
                "sql_query": sql,
                "raw_results": [],
                "error": "Unsafe query blocked",
            }

        rows = execute_sql(sql)
        answer = explain_results(question, sql, rows)

        return {
            "answer": answer,
            "sql_query": sql,
            "raw_results": rows,
            "error": None,
        }

    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error while processing your question: {e}",
            "sql_query": "",
            "raw_results": [],
            "error": str(e),
        }
