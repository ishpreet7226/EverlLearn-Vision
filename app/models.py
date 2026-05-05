"""
EverLearn Vision – ORM Models
===============================
SQLAlchemy models that map to PostgreSQL tables.

Each class here corresponds to a database table. SQLAlchemy handles
the translation between Python objects and SQL rows automatically.

Table: feedback
    Stores user corrections to model predictions. Each row represents
    one instance where a user said "the model predicted X but the
    answer is actually Y". This data can be used for:
      - Tracking model accuracy over time
      - Identifying weak spots (which classes get confused)
      - Building a curated dataset for model retraining
"""

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Feedback(Base):
    """
    ORM model for the 'feedback' table.

    Columns:
        id              — Auto-incrementing primary key
        image_name      — Original filename of the uploaded image
        predicted_label — What the model predicted (e.g. "dog")
        actual_label    — The user's correction (e.g. "cat")
        confidence      — Model's softmax confidence for its prediction
        created_at      — Timestamp, auto-set on insert
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    predicted_label = Column(String(100), nullable=False)
    actual_label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Feedback id={self.id} predicted='{self.predicted_label}' "
            f"actual='{self.actual_label}' confidence={self.confidence:.2f}>"
        )
