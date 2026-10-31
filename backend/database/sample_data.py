"""
database/sample_data.py
------------------------
Creates and seeds the SQLite business database with realistic sample data.
Run once with:  python -m backend.database.sample_data
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from backend.config.settings import settings

DB_PATH = settings.DATA_DIR / "business.db"


def seed_database(force: bool = False) -> None:
    """Create tables and insert sample data. Skip if DB already exists unless force=True."""
    if DB_PATH.exists() and not force:
        print(f"Database already exists at {DB_PATH}. Use force=True to re-seed.")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Schema ────────────────────────────────────────────────────────────────
    cur.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS employees;

        CREATE TABLE customers (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            email      TEXT UNIQUE,
            city       TEXT,
            country    TEXT,
            created_at TEXT
        );

        CREATE TABLE products (
            id             INTEGER PRIMARY KEY,
            name           TEXT NOT NULL,
            category       TEXT,
            price          REAL,
            stock_quantity INTEGER
        );

        CREATE TABLE sales (
            id           INTEGER PRIMARY KEY,
            customer_id  INTEGER REFERENCES customers(id),
            product_id   INTEGER REFERENCES products(id),
            quantity     INTEGER,
            total_amount REAL,
            sale_date    TEXT
        );

        CREATE TABLE employees (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            department TEXT,
            salary     REAL,
            hire_date  TEXT,
            manager_id INTEGER REFERENCES employees(id)
        );
    """)

    # ── Customers ─────────────────────────────────────────────────────────────
    customers = [
        ("Alice Johnson", "alice@example.com", "New York", "USA"),
        ("Bob Smith", "bob@example.com", "London", "UK"),
        ("Carlos Ruiz", "carlos@example.com", "Madrid", "Spain"),
        ("Diana Lee", "diana@example.com", "Tokyo", "Japan"),
        ("Eva Müller", "eva@example.com", "Berlin", "Germany"),
        ("Frank Chen", "frank@example.com", "Shanghai", "China"),
        ("Grace Patel", "grace@example.com", "Mumbai", "India"),
        ("Henry Brown", "henry@example.com", "Sydney", "Australia"),
        ("Isabel Ferreira", "isabel@example.com", "São Paulo", "Brazil"),
        ("Jack Wilson", "jack@example.com", "Toronto", "Canada"),
    ]
    base_date = date(2022, 1, 1)
    for i, (name, email, city, country) in enumerate(customers, 1):
        created = base_date + timedelta(days=i * 30)
        cur.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?)",
            (i, name, email, city, country, str(created)),
        )

    # ── Products ──────────────────────────────────────────────────────────────
    products = [
        ("Laptop Pro 15", "Electronics", 1299.99, 50),
        ("Wireless Headphones", "Electronics", 199.99, 200),
        ("Standing Desk", "Furniture", 549.00, 30),
        ("Ergonomic Chair", "Furniture", 429.00, 40),
        ("Python Crash Course (Book)", "Books", 39.99, 500),
        ("AI Engineering Handbook", "Books", 54.99, 300),
        ("USB-C Hub 7-in-1", "Accessories", 49.99, 400),
        ("4K Webcam", "Electronics", 149.99, 150),
        ("Mechanical Keyboard", "Accessories", 129.99, 250),
        ("Monitor 27\" 4K", "Electronics", 699.99, 60),
    ]
    for i, (name, cat, price, qty) in enumerate(products, 1):
        cur.execute("INSERT INTO products VALUES (?,?,?,?,?)", (i, name, cat, price, qty))

    # ── Sales (random, last 12 months) ────────────────────────────────────────
    random.seed(42)
    sale_id = 1
    for _ in range(200):
        cust_id = random.randint(1, 10)
        prod_id = random.randint(1, 10)
        price = products[prod_id - 1][2]
        qty = random.randint(1, 5)
        total = round(price * qty, 2)
        days_ago = random.randint(0, 365)
        sale_date = date.today() - timedelta(days=days_ago)
        cur.execute(
            "INSERT INTO sales VALUES (?,?,?,?,?,?)",
            (sale_id, cust_id, prod_id, qty, total, str(sale_date)),
        )
        sale_id += 1

    # ── Employees ─────────────────────────────────────────────────────────────
    employees = [
        (1, "Sarah Connor", "Engineering", 120000, "2020-03-01", None),
        (2, "John Martinez", "Engineering", 95000, "2021-06-15", 1),
        (3, "Emily Zhang", "Engineering", 92000, "2021-09-01", 1),
        (4, "Michael Scott", "Sales", 85000, "2019-01-10", None),
        (5, "Pam Beesly", "Sales", 72000, "2020-05-20", 4),
        (6, "Dwight Schrute", "Sales", 78000, "2019-07-04", 4),
        (7, "Angela Martin", "Finance", 98000, "2018-11-01", None),
        (8, "Oscar Nunez", "Finance", 82000, "2020-02-14", 7),
        (9, "Kevin Malone", "Finance", 70000, "2021-03-31", 7),
        (10, "Ryan Howard", "Marketing", 88000, "2022-01-17", None),
    ]
    for emp in employees:
        cur.execute("INSERT INTO employees VALUES (?,?,?,?,?,?)", emp)

    conn.commit()
    conn.close()
    print(f"✅  Database seeded at {DB_PATH}")


if __name__ == "__main__":
    seed_database(force=True)
