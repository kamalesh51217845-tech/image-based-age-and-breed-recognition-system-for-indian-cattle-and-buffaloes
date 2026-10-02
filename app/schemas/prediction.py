"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BreedPrediction(BaseModel):
    """Schema for individual breed prediction."""
    breed: str
    confidence: float = Field(..., ge=0, le=1)


class AgePrediction(BaseModel):
    """Schema for individual age prediction."""
    age_group: str
    confidence: float = Field(..., ge=0, le=1)


class AnimalTypePrediction(BaseModel):
    """Schema for animal type prediction."""
    prediction: str
    confidence: float = Field(..., ge=0, le=1)


class BreedResult(BaseModel):
    """Schema for breed classification result."""
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    top_predictions: List[BreedPrediction]


class AgeResult(BaseModel):
    """Schema for age estimation result."""
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    all_predictions: List[AgePrediction]


class PredictionResponse(BaseModel):
    """Complete prediction response schema."""
    success: bool
    prediction_id: Optional[int] = None
    image_path: Optional[str] = None
    animal_type: AnimalTypePrediction
    breed: BreedResult
    age: AgeResult
    breed_info: Optional[dict] = None
    message: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    """Schema for prediction history item."""
    id: int
    image_path: str
    original_filename: Optional[str]
    animal_type: str
    animal_type_confidence: float
    predicted_breed: str
    breed_confidence: float
    predicted_age: str
    age_confidence: float
    created_at: datetime
    user_feedback: Optional[str]

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    """Schema for user feedback on predictions."""
    prediction_id: int
    feedback: str = Field(..., pattern="^(correct|incorrect)$")
    actual_breed: Optional[str] = None
    actual_age: Optional[str] = None


class BreedInfoCreate(BaseModel):
    """Schema for creating new breed info."""
    name: str
    animal_type: str = Field(..., pattern="^(Cattle|Buffalo)$")
    origin: Optional[str] = None
    breed_type: Optional[str] = None
    characteristics: Optional[str] = None
    average_milk_yield: Optional[str] = None
    image_url: Optional[str] = None


class BreedInfoResponse(BaseModel):
    """Schema for breed info response."""
    id: int
    name: str
    animal_type: str
    origin: Optional[str]
    breed_type: Optional[str]
    characteristics: Optional[str]
    average_milk_yield: Optional[str]
    image_url: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Schema for statistics response."""
    total_predictions: int
    breed_distribution: dict
    animal_type_distribution: dict


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    success: bool = False
    error: str
    detail: Optional[str] = None
