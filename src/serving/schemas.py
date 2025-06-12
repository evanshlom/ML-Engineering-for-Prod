"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """API health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class NICUFeatures(BaseModel):
    """Input features for prediction request."""
    
    heart_rate: float = Field(
        ...,
        ge=50.0,
        le=200.0,
        description="Heart rate in beats per minute"
    )
    oxygen_saturation: float = Field(
        ...,
        ge=70.0,
        le=100.0,
        description="Oxygen saturation percentage"
    )
    respiratory_rate: float = Field(
        ...,
        ge=20.0,
        le=80.0,
        description="Respiratory rate in breaths per minute"
    )
    weight_grams: float = Field(
        ...,
        ge=500.0,
        le=5000.0,
        description="Infant weight in grams"
    )
    temperature_celsius: float = Field(
        ...,
        ge=35.0,
        le=38.5,
        description="Body temperature in Celsius"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "heart_rate": 120.0,
                "oxygen_saturation": 96.5,
                "respiratory_rate": 35.0,
                "weight_grams": 2100.0,
                "temperature_celsius": 36.8
            }
        }
    )
    
    @field_validator('oxygen_saturation')
    def validate_oxygen_percentage(cls, v):
        """Ensure oxygen saturation is a valid percentage."""
        if not 0 <= v <= 100:
            raise ValueError('Oxygen saturation must be between 0 and 100')
        return v


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    features: NICUFeatures
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    request_id: Optional[str] = Field(None, description="Optional request tracking ID")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "heart_rate": 120.0,
                    "oxygen_saturation": 96.5,
                    "respiratory_rate": 35.0,
                    "weight_grams": 2100.0,
                    "temperature_celsius": 36.8
                },
                "patient_id": "NICU-2024-001",
                "request_id": "req-123456"
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    instances: List[NICUFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of feature sets for batch prediction"
    )
    request_id: Optional[str] = Field(None, description="Optional batch request tracking ID")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    suitable_for_kangaroo_care: bool = Field(
        ...,
        description="Binary prediction of kangaroo care readiness"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being suitable for kangaroo care"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Relative importance of each feature in the prediction"
    )
    model_version: str = Field(..., description="Version of the model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patient_id: Optional[str] = None
    request_id: Optional[str] = None
    
    @field_validator('confidence')
    def compute_confidence(cls, v, info):
        """Compute confidence level based on probability."""
        prob = info.data.get('probability', 0.5)
        if prob > 0.85 or prob < 0.15:
            return 'high'
        elif prob > 0.7 or prob < 0.3:
            return 'medium'
        else:
            return 'low'


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_count: int
    suitable_count: int
    average_probability: float
    model_version: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    auc_roc: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluation_date: datetime
    test_set_size: int


class ModelInfo(BaseModel):
    """Model information and metadata."""
    
    model_name: str = "NICUReadinessClassifier"
    model_version: str
    training_date: datetime
    feature_names: List[str]
    model_type: str = "binary_classifier"
    framework: str = "pytorch"
    metrics: ModelMetrics
    hyperparameters: Dict[str, Union[int, float, str, List]]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: HealthStatus
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    last_prediction_time: Optional[datetime]
    total_predictions: int
    checks: Dict[str, bool] = Field(
        default_factory=lambda: {
            "database": True,
            "model": True,
            "memory": True,
            "disk": True
        }
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    error_type: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ExplainabilityRequest(BaseModel):
    """Request for model explainability analysis."""
    
    features: NICUFeatures
    explanation_type: str = Field(
        default="feature_importance",
        pattern="^(feature_importance|decision_boundary|counterfactual)$"
    )
    target_feature: Optional[str] = Field(
        None,
        description="Feature to analyze for decision boundary"
    )


class ExplainabilityResponse(BaseModel):
    """Response with model explanation."""
    
    prediction: PredictionResponse
    explanation_type: str
    feature_importance: Dict[str, float]
    decision_boundaries: Optional[Dict[str, List[float]]] = None
    counterfactual_examples: Optional[List[NICUFeatures]] = None
    interpretation: str = Field(
        ...,
        description="Human-readable interpretation of the prediction"
    )