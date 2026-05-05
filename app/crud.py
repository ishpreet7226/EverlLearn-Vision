"""
EverLearn Vision – CRUD Operations
=====================================
Database read/write operations for the feedback table.

Separating CRUD from route handlers keeps the code clean:
  - Routes handle HTTP concerns (status codes, headers, validation)
  - CRUD handles database concerns (queries, inserts, commits)

This makes it easy to reuse the same DB operations in scripts,
background tasks, or tests without importing FastAPI.
"""

from sqlalchemy.orm import Session

from app.models import Feedback
from app.schemas import FeedbackCreate


def create_feedback(db: Session, feedback_data: FeedbackCreate) -> Feedback:
    """
    Insert a new feedback row into the database.

    Args:
        db            : Active SQLAlchemy session
        feedback_data : Validated Pydantic model with the feedback fields

    Returns:
        The created Feedback ORM instance (with id and created_at populated)
    """
    row = Feedback(
        image_name=feedback_data.image_name,
        predicted_label=feedback_data.predicted_label,
        actual_label=feedback_data.actual_label,
        confidence=feedback_data.confidence,
    )
    db.add(row)
    db.commit()
    db.refresh(row)  # Populate id + created_at from the database
    return row


def get_all_feedback(
    db: Session,
    skip: int = 0,
    limit: int = 50,
) -> list[Feedback]:
    """
    Retrieve feedback entries, newest first, with pagination.

    Args:
        db    : Active SQLAlchemy session
        skip  : Number of rows to skip (for pagination)
        limit : Maximum number of rows to return

    Returns:
        List of Feedback ORM instances
    """
    return (
        db.query(Feedback)
        .order_by(Feedback.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_feedback_count(db: Session) -> int:
    """Return the total number of feedback entries in the database."""
    return db.query(Feedback).count()
