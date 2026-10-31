"""Tests for the SQL agent — focuses on safety validation and query extraction."""
import pytest
from backend.agents.sql_agent import _is_safe_query, _extract_sql


# ── Safety validation tests ───────────────────────────────────────────────────

def test_select_is_safe():
    assert _is_safe_query("SELECT * FROM customers") is True


def test_select_with_join_is_safe():
    sql = "SELECT c.name, SUM(s.total_amount) FROM customers c JOIN sales s ON c.id=s.customer_id GROUP BY c.id"
    assert _is_safe_query(sql) is True


def test_drop_is_blocked():
    assert _is_safe_query("DROP TABLE customers") is False


def test_delete_is_blocked():
    assert _is_safe_query("DELETE FROM sales WHERE id=1") is False


def test_update_is_blocked():
    assert _is_safe_query("UPDATE products SET price=0") is False


def test_insert_is_blocked():
    assert _is_safe_query("INSERT INTO customers VALUES (99,'Hacker','h@x.com','','')") is False


def test_non_select_start_is_blocked():
    assert _is_safe_query("PRAGMA table_info(customers)") is False


# ── SQL extraction tests ──────────────────────────────────────────────────────

def test_extract_sql_from_markdown_block():
    raw = "Here is the query:\n```sql\nSELECT * FROM products;\n```"
    assert _extract_sql(raw) == "SELECT * FROM products;"


def test_extract_sql_plain():
    raw = "SELECT id, name FROM employees WHERE department='Engineering'"
    assert _extract_sql(raw) == raw


def test_extract_sql_generic_code_block():
    raw = "```\nSELECT COUNT(*) FROM sales\n```"
    assert _extract_sql(raw) == "SELECT COUNT(*) FROM sales"
