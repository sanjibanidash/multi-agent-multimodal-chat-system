"""
database/database.py
---------------------
SQLAlchemy engine setup for the SQLite business database.
The database file is created automatically in the /data directory.
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config.settings import settings

DB_PATH = settings.DATA_DIR / "business.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass
