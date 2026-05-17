"""
EverLearn Vision – Database Connection
========================================
SQLAlchemy engine, session factory, and declarative base.

Supports both SQLite (default, zero-config) and PostgreSQL (production).

Database selection:
    - By default, uses a local SQLite file (everlearn.db) — works
      on any machine with no setup required.
    - To use PostgreSQL, set the DATABASE_URL environment variable:
        export DATABASE_URL=postgresql://user:pass@localhost:5432/everlearn_vision

Usage:
    from app.database import get_db, engine, Base

    # In FastAPI lifespan:
    Base.metadata.create_all(bind=engine)

    # As a route dependency:
    @app.get("/items")
    def read_items(db: Session = Depends(get_db)):
        ...
"""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# ── Connection URL ────────────────────────────────────────────────────────────
# Default: local SQLite file (zero-config, works everywhere).
# Override in production via the DATABASE_URL environment variable
# to use PostgreSQL or any other SQLAlchemy-supported database.
_SQLITE_DEFAULT = "sqlite:///everlearn.db"
DATABASE_URL = os.getenv("DATABASE_URL", _SQLITE_DEFAULT)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True tests connections before handing them out, avoiding
# stale-connection errors after database restarts.
# connect_args={"check_same_thread": False} is required for SQLite only
# (SQLite doesn't allow multi-threaded access by default).
_engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# ── Session factory ───────────────────────────────────────────────────────────
# autocommit=False  → we control when commits happen (explicit is better)
# autoflush=False   → prevents unexpected flushes before queries
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Declarative Base ──────────────────────────────────────────────────────────
# All ORM models inherit from this Base.
Base = declarative_base()


# ── Dependency ────────────────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.

    The session is automatically closed after the request finishes,
    even if an exception occurs (thanks to the finally block).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
