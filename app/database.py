"""
EverLearn Vision – Database Connection
========================================
SQLAlchemy engine, session factory, and declarative base for PostgreSQL.

Reads the DATABASE_URL from an environment variable, falling back to
a local PostgreSQL instance at localhost:5432/everlearn_vision.

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
# Default: local PostgreSQL with no password (Homebrew macOS default).
# Override in production via the DATABASE_URL environment variable.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/everlearn_vision",
)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True tests connections before handing them out, avoiding
# stale-connection errors after PostgreSQL restarts.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

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
