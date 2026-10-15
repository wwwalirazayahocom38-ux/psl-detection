from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class SignItem(BaseModel):
    id: str
    english: str
    urdu: str
    phonetic: str
    category: str


class DetectionResult(BaseModel):
    sign_key: str
    english: str
    urdu: str
    phonetic: str
    category: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DetectionLogResponse(BaseModel):
    id: int
    sign_key: str
    english_label: str
    urdu_label: str
    phonetic: Optional[str]
    category: Optional[str]
    confidence: float
    timestamp: datetime
    session_id: Optional[str]

    class Config:
        from_attributes = True


class AnalyticsSummary(BaseModel):
    total_detections: int
    today_detections: int
    avg_confidence: float
    unique_signs_detected: int
    letters_count: int
    digits_count: int
    phrases_count: int
    popular_signs: List[dict]
    recent_activity: List[DetectionLogResponse]
    daily_trend: List[dict]


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
