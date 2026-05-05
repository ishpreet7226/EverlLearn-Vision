"""
EverLearn Vision – Pydantic Schemas
=====================================
Request/response validation models for the feedback API endpoints.

These Pydantic models ensure that:
  - Incoming JSON has all required fields with correct types
  - Outgoing JSON has a consistent, documented shape
  - FastAPI auto-generates accurate OpenAPI/Swagger docs

Note: These are separate from the SQLAlchemy ORM models in models.py.
  - ORM models  → talk to the database
  - Pydantic    → talk to the API consumer
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    """
    Request body for POST /feedback.

    Sent by the frontend when a user corrects a prediction.
    """

    image_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename of the uploaded image",
        examples=["cat_photo.jpg"],
    )
    predicted_label: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The label the model predicted",
        examples=["dog"],
    )
    actual_label: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The correct label provided by the user",
        examples=["cat"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's softmax confidence for its prediction (0-1)",
        examples=[0.72],
    )


class FeedbackResponse(BaseModel):
    """
    Response body returned after creating or reading a feedback entry.

    Includes the auto-generated `id` and `created_at` from the database.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    image_name: str
    predicted_label: str
    actual_label: str
    confidence: float
    created_at: datetime


class FeedbackListResponse(BaseModel):
    """Response body for GET /feedback — a paginated list with total count."""

    count: int = Field(..., description="Total number of feedback entries")
    feedback: list[FeedbackResponse]
